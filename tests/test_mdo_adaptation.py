"""CPU-only fixtures for split isolation, probe restoration, and exact resume.

These deliberately tiny synthetic fixtures are unit tests, never training data
for exported AeroTransformer checkpoints. Public training rejects their shape
and provenance, and requires the official Large weights and frozen full split.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from mdo_demo.mdo_adaptation import (
    DEFAULT_LOSS, FROZEN_SPLIT_SHA256, LOSS_KEYS, _seed,
    _validate_partition_metadata, fit_mdo_epochs, resolve_mdo_split, resource_probe,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

ROOT = Path(__file__).resolve().parents[1]


def small_metadata():
    raw = {"dataset": {"total_samples": 401, "total_geometries": 4}, "splits": {}}
    for name, shape, ids in (("train", 0, [100, 101]), ("dev", 1, [200]),
                             ("calibration", 2, [300]), ("test", 3, [400])):
        raw["splits"][name] = {"geometry_ids": [shape], "source_sample_ids": ids,
                              "n_geometries": 1, "n_samples": len(ids)}
    index = np.zeros((3, 12))
    index[:, 0] = [0, 0, 1]
    dataset = SimpleNamespace(sample_ids=[100, 101, 200], index=index)
    return raw, dataset


class SplitIsolationTests(unittest.TestCase):
    def test_compact_dataset_needs_no_calibration_or_test_fields(self):
        raw, dataset = small_metadata()
        dataset.sample = lambda row: self.fail("Split validation must never read flow fields")
        result = _validate_partition_metadata(raw, dataset)
        self.assertEqual(result["train_rows"], [0, 1])
        self.assertEqual(result["dev_rows"], [2])
        self.assertEqual(result["calibration_sample_ids"], [300])
        self.assertEqual(result["test_sample_ids"], [400])

    def test_sample_and_geometry_overlap_are_independently_rejected(self):
        raw, dataset = small_metadata()
        raw["splits"]["test"]["source_sample_ids"] = [300]
        with self.assertRaisesRegex(ValueError, "sample IDs overlap"):
            _validate_partition_metadata(raw, dataset)
        raw, dataset = small_metadata()
        raw["splits"]["test"]["geometry_ids"] = [2]
        with self.assertRaisesRegex(ValueError, "geometry IDs overlap"):
            _validate_partition_metadata(raw, dataset)

    def test_missing_training_id_and_wrong_geometry_are_rejected(self):
        raw, dataset = small_metadata()
        dataset.sample_ids = [100, 999, 200]
        with self.assertRaisesRegex(ValueError, "Missing exact train IDs"):
            _validate_partition_metadata(raw, dataset)
        raw, dataset = small_metadata()
        dataset.index[1, 0] = 1
        with self.assertRaisesRegex(ValueError, "mapping disagrees"):
            _validate_partition_metadata(raw, dataset)

    def test_full_metadata_checks_exact_local_rows(self):
        raw, dataset = small_metadata()
        full_index = np.zeros((401, 12))
        full_index[200, 0], full_index[300, 0], full_index[400, 0] = 1, 2, 3
        _validate_partition_metadata(raw, dataset, full_index)
        dataset.index[0, 3] = 0.2
        with self.assertRaisesRegex(ValueError, "Local index values differ"):
            _validate_partition_metadata(raw, dataset, full_index)

    def test_public_helper_accepts_only_exact_frozen_partition(self):
        raw = json.loads((ROOT / "configs/protocol_v1_split.json").read_text(encoding="utf-8"))
        ids, shapes = [], []
        for name in ("train", "dev"):
            group = raw["splits"][name]
            ids.extend(group["source_sample_ids"])
            # Metadata fixture only: covers each declared geometry. No flow data.
            shapes.extend(group["geometry_ids"][i % group["n_geometries"]]
                          for i in range(group["n_samples"]))
        dataset = SimpleNamespace(sample_ids=ids, index=np.zeros((len(ids), 12)))
        dataset.index[:, 0] = shapes
        dataset.sample = lambda row: self.fail("No flow-field access permitted")
        result = resolve_mdo_split(dataset, raw)
        self.assertEqual(result["split_sha256"], FROZEN_SPLIT_SHA256)
        self.assertEqual(len(result["train_rows"]), 450)
        self.assertEqual(len(result["dev_rows"]), 350)
        self.assertEqual(result["metadata_validation"], "frozen_manifest_and_available_index_rows")
        changed = deepcopy(raw)
        changed["splits"]["train"]["source_sample_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "differs from the frozen"):
            resolve_mdo_split(dataset, changed)
        dataset.sample_ids.append(raw["splits"]["calibration"]["source_sample_ids"][0])
        extra = np.zeros((1, 12))
        extra[0, 0] = raw["splits"]["calibration"]["geometry_ids"][0]
        dataset.index = np.concatenate((dataset.index, extra))
        with self.assertRaisesRegex(ValueError, "exactly frozen train\\+dev"):
            resolve_mdo_split(dataset, raw)


class TinyTrainingData:
    """Only two explicitly authorized unit-test rows may be sampled."""
    def __init__(self):
        z, x = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 6), indexing="ij")
        y = 0.04 * np.sin(x * np.pi) + 0.01 * z * x
        self.vertices = np.stack((x + 0.2 * z, y, z)).astype(np.float32)
        self.geometry = (self.vertices[:, :-1, :-1] + self.vertices[:, 1:, :-1]
                         + self.vertices[:, :-1, 1:] + self.vertices[:, 1:, 1:]) / 4
        self.accesses = []

    def sample(self, row):
        if row not in (0, 1):
            raise AssertionError("Development/calibration/test fields were accessed")
        self.accesses.append(row)
        fields = self.geometry * np.float32(0.15) - np.float32(0.2 + row * 0.01)
        return {"sample_id": 100 + row, "shape_id": 0,
                "geometry": self.geometry, "original_geometry": self.vertices,
                "condition": np.array([3 + row, .8], dtype=np.float32),
                "ref_area": 0.9, "fields": fields}


if HAS_TORCH:
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.dropout = torch.nn.Dropout(p=0.1)
            self.head = torch.nn.Conv2d(4, 3, kernel_size=1)
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

        def forward(self, geometry, code):
            self.calls.add_(1)
            return self.head(self.dropout(torch.tanh(self.backbone(geometry))))


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch dependency is not installed")
class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        _seed(torch, 19)
        self.device = torch.device("cpu")
        self.data = TinyTrainingData()
        self.model = TinyModel()

    def test_probe_restores_all_weights_buffers_rng_and_training_flags(self):
        self.model.eval()
        next(self.model.parameters()).requires_grad_(False)
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            result = resource_probe(self.model, self.data, [0, 1], torch=torch,
                                    device=self.device, output=Path(temporary),
                                    warmup_steps=1, measured_steps=2)
            records = [json.loads(line) for line in (Path(temporary) / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(result["measured_source_sample_ids"], [101, 100])
            self.assertGreater(result["median_step_wall_s"], 0)
            self.assertFalse((Path(temporary) / "latest_training.pt").exists())
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertFalse(self.model.training)
        self.assertFalse(next(self.model.parameters()).requires_grad)

    def test_full_model_updates_and_logs_all_components_without_holdout_access(self):
        before = self.model.backbone.weight.detach().clone()
        next(self.model.parameters()).requires_grad_(False)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = fit_mdo_epochs(self.model, self.data, [0, 1], torch=torch,
                                    device=self.device, output=output, epochs=1, seed=19)
            saved = torch.load(output / "latest_training.pt", weights_only=True)
            records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertFalse(torch.equal(self.model.backbone.weight, before))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
        self.assertEqual(result["completed_epochs"], 1)
        self.assertEqual(saved["completed_epochs"], 1)
        self.assertEqual(set(result["history"][0]["training_loss_online"]), set(LOSS_KEYS))
        steps = [record for record in records if record["stage"] == "train_step"]
        self.assertEqual({record["source_sample_id"] for record in steps}, {100, 101})
        for record in steps:
            self.assertEqual(set(record["losses"]), set(LOSS_KEYS))
            self.assertEqual(record["seed"], 19)
        self.assertEqual(set(self.data.accesses), {0, 1})

    def test_epoch_resume_matches_uninterrupted_training_including_dropout(self):
        initial = deepcopy(self.model.state_dict())
        contract = {"seed": 19, "loss": DEFAULT_LOSS, "test_fixture": True}
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            _seed(torch, 19)
            fit_mdo_epochs(self.model, self.data, [0, 1], torch=torch, device=self.device,
                           output=Path(left), epochs=2, seed=19, contract=contract)
            expected = deepcopy(self.model.state_dict())
            interrupted = TinyModel()
            interrupted.load_state_dict(initial)
            _seed(torch, 19)
            fit_mdo_epochs(interrupted, self.data, [0, 1], torch=torch, device=self.device,
                           output=Path(right), epochs=1, seed=19, contract=contract)
            resumed = TinyModel()  # RNG is deliberately advanced before restoring.
            result = fit_mdo_epochs(resumed, self.data, [0, 1], torch=torch, device=self.device,
                                    output=Path(right), epochs=2, seed=19, contract=contract, resume=True)
            self.assertEqual(result["resumed_from_epoch"], 1)
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            # A preemption after committing the last epoch but before exporting
            # final weights can resume finalization without repeating any step.
            accesses_before = len(self.data.accesses)
            finalized = fit_mdo_epochs(resumed, self.data, [0, 1], torch=torch, device=self.device,
                                       output=Path(right), epochs=2, seed=19, contract=contract, resume=True)
            self.assertEqual(finalized["completed_epochs"], 2)
            self.assertEqual(len(self.data.accesses), accesses_before)
            with self.assertRaisesRegex(ValueError, "Resume contract differs"):
                fit_mdo_epochs(resumed, self.data, [0, 1], torch=torch, device=self.device,
                               output=Path(right), epochs=3, contract={"wrong": True}, resume=True)

    def test_nonfinite_prediction_aborts_before_committing_checkpoint(self):
        with torch.no_grad():
            self.model.head.bias.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "Invalid surface prediction"):
                fit_mdo_epochs(self.model, self.data, [0, 1], torch=torch, device=self.device,
                               output=Path(temporary), epochs=1)
            self.assertFalse((Path(temporary) / "latest_training.pt").exists())


if __name__ == "__main__":
    unittest.main()
