"""Exercise workflow partitioning and provenance without optional torch/CFD."""

import copy
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mdo_demo.confidence import fit_calibration, split_geometry_groups
from mdo_demo.aerotransformer import prediction_fingerprint
from mdo_demo.dataset import load_dataset
from mdo_demo.experiment import augment_split, run_experiment, verify_dataset
from mdo_demo.io import read_json, sha256_file, write_json


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # Mock call history can hold mmap arrays in cycles; release before
        # TemporaryDirectory removes open files on Windows.
        self.addCleanup(gc.collect)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.shape_ids = list(range(10, 49, 3))
        self.sample_ids = list(range(200, 291, 7))
        index = np.zeros((13, 12))
        index[:, 0] = self.shape_ids
        index[:, 2:4] = [2., .8]
        index[:, 4:6] = [10., 6.]
        index[:, 6:9] = index[:, 9:12] = [.5, .02, -.1]
        span, chord = np.meshgrid(np.linspace(0, 6, 4), np.linspace(0, 1, 6), indexing="ij")
        geometry = np.stack((chord, .1 * np.sin(chord * np.pi), span))
        arrays = {"index.npy": index, "data.npy": np.zeros((13, 3, 4, 6)),
                  "geom0.npy": np.repeat(geometry[None], 13, axis=0),
                  "origingeom.npy": np.zeros((13, 3, 5, 7))}
        for name, array in arrays.items():
            np.save(self.data / name, array, allow_pickle=False)
        # A temporary schema fixture, never a downloaded or reported experiment.
        manifest = {"sample_ids": self.sample_ids, "geometry_shape_ids": self.shape_ids,
                    "dataset": "thuerey-group/CRMpert", "split": "unit_test_only",
                    "revision": "88ece28b846fd1d9870933252db556cf97d30ae0",
                    "field_scales": [1., 150., 300.],
                    "files": {name: {"local_sha256": sha256_file(self.data / name)} for name in arrays}}
        write_json(self.data / "manifest.json", manifest)

    def test_dataset_integrity_rejects_changed_array(self):
        path = self.data / "data.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 0, 0, 0] = 1
        np.save(path, values, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch: data.npy"):
            verify_dataset(load_dataset(self.data))

    def test_dataset_integrity_requires_every_array(self):
        manifest = read_json(self.data / "manifest.json")
        del manifest["files"]["origingeom.npy"]
        write_json(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "verified fetch_assets"):
            verify_dataset(load_dataset(self.data))

    def test_custom_or_synthetic_dataset_is_not_called_pinned_cfd(self):
        manifest = read_json(self.data / "manifest.json")
        manifest["dataset"] = "synthetic_fixture"
        write_json(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "pinned CRMpert"):
            verify_dataset(load_dataset(self.data))

    def test_split_augmentation_preserves_source_identity(self):
        dataset = load_dataset(self.data)
        split = split_geometry_groups(self.shape_ids, seed=7,
                                      counts={"train": 2, "calibration": 9, "test": 2})
        original = copy.deepcopy(split)
        result = augment_split(dataset, split)
        sample_to_shape = dict(zip(self.sample_ids, map(str, self.shape_ids)))
        consumed = []
        for partition in ("train", "calibration", "test"):
            self.assertEqual(result[partition], original[partition])
            selected = result[partition + "_sample_ids"]
            self.assertEqual({sample_to_shape[sid] for sid in selected}, set(original[partition]))
            consumed.extend(selected)
        self.assertEqual(result["fingerprint"], original["fingerprint"])
        self.assertEqual(sorted(consumed), self.sample_ids)

    def test_model_identity_changes_with_config_and_postprocessing(self):
        provenance = {"weights_sha256": "weights", "config_sha256": "config", "adapter_sha256": "adapter",
                      "flogen": {"actual_revision": "model-rev", "module_sha256": "model-code",
                                 "tracked_files_modified": False},
                      "cfdpost": {"actual_revision": "post-rev", "module_sha256": "post-code",
                                  "tracked_files_modified": False}}
        original = prediction_fingerprint(provenance)
        for key in ("weights_sha256", "config_sha256", "adapter_sha256"):
            changed = copy.deepcopy(provenance)
            changed[key] = "changed"
            self.assertNotEqual(prediction_fingerprint(changed), original)
        for source in ("flogen", "cfdpost"):
            changed = copy.deepcopy(provenance)
            changed[source]["module_sha256"] = "changed"
            self.assertNotEqual(prediction_fingerprint(changed), original)

    def test_workflow_calibrates_and_reports_disjoint_partitions(self):
        """Fake only model fitting/inference; run real splitting, gate and report."""
        checkpoint = self.root / "base"
        checkpoint.mkdir()
        (checkpoint / "best_model_weights").write_bytes(b"unit-test-not-a-model")
        (checkpoint / "model_config").write_text("{}", encoding="utf-8")
        config = {"seed": 7, "shape_counts": {"train": 2, "calibration": 9, "test": 2},
                  "adaptation": {"mode": "head", "epochs": 1, "batch_size": 1, "lr": .0001},
                  "alpha": .1, "tolerances": {"CL": .01, "CD": .0005}}
        config_path = self.root / "config.json"
        write_json(config_path, config)
        output = self.root / "result"

        class FakePredictor:
            def __init__(self, model_path, device):
                self.provenance = {"weights_sha256": str(Path(model_path).name),
                                   "config_sha256": "test-config",
                                   "adapter_sha256": "test-adapter",
                                   "flogen": {"actual_revision": "test-source", "module_sha256": "test-module"},
                                   "cfdpost": {"actual_revision": "test-post", "module_sha256": "test-post-module"}}

        def fake_train(data_dir, base, split_path, target, **kwargs):
            saved_split = read_json(split_path)
            target.mkdir()
            manifest = {"train_sample_ids": saved_split["train_sample_ids"],
                        "train_shape_ids": saved_split["train"], "mode": kwargs["mode"],
                        "hyperparameters": kwargs, "fit_wall_s": .01, "total_wall_s": .02}
            write_json(target / "manifest.json", manifest)
            return manifest

        def fake_evaluate(dataset, predictor, directory, limit, warmup):
            directory.mkdir()
            cases = []
            for row in range(limit):
                sample = dataset.sample(row)
                filename = f"case_{sample['sample_id']}.npz"
                np.savez(directory / filename, geometry=sample["geometry"],
                         truth=sample["fields"], prediction=sample["fields"] + .01)
                cases.append({"sample_id": sample["sample_id"], "shape_id": sample["shape_id"],
                              "condition": {"alpha_deg": 2., "mach": .8},
                              "coefficients": {"CL": .5002, "CD": .02005, "CM": -.1},
                              "reference_coefficients": sample["reference_coefficients"],
                              "field_errors": {key: {"rmse": .01} for key in ("Cp", "Cf_stream", "Cf_span")},
                              "end_to_end_prediction_s": .01, "array_file": filename})
            # Mock the production evaluator's contract; all files remain in
            # this temporary unittest directory and are discarded afterward.
            return {"mode": "real_checkpoint_dataset_evaluation", "cases": cases}

        with patch("mdo_demo.adaptation.train_model", side_effect=fake_train) as training, \
             patch("mdo_demo.aerotransformer.AeroTransformerPredictor", FakePredictor), \
             patch("mdo_demo.evaluation.evaluate", side_effect=fake_evaluate), \
             patch("mdo_demo.confidence.fit_calibration", wraps=fit_calibration) as calibrating:
            summary = run_experiment(self.data, checkpoint, config_path, output)
        split = summary["split"]
        self.assertEqual(training.call_count, 1)
        self.assertEqual(summary["training"]["train_sample_ids"], split["train_sample_ids"])
        for call in calibrating.call_args_list:
            self.assertEqual({str(case["shape_id"]) for case in call.args[0]}, set(split["calibration"]))
            self.assertEqual({str(case["shape_id"]) for case in call.kwargs["support_cases"]}, set(split["train"]))
        for result in summary["results"].values():
            self.assertEqual(result["metrics"]["samples"], 2)
            self.assertEqual({case["sample_id"] for case in result["heldout_cases"]}, set(split["test_sample_ids"]))
            self.assertEqual(result["calibration"]["calibration_group_count"], 9)
            self.assertEqual(result["identity"]["dataset"], sha256_file(self.data / "manifest.json"))
        self.assertIsNone(summary["live_cfd_speedup"])
        self.assertTrue((output / "report.html").is_file())
        previous = (output / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "new experiment output"):
            run_experiment(self.data, checkpoint, config_path, output)
        self.assertEqual((output / "summary.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
