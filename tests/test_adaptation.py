"""Leakage and frozen-parameter contracts; tiny torch fixture is not CFD evidence."""
import copy
import importlib.util
import unittest

import numpy as np

from mdo_demo.adaptation import configure_trainable, resolve_training_split


class MetadataOnlyDataset:
    sample_ids = [100, 101, 200, 300]
    index = np.array([[8], [8], [9], [10]])

    def sample(self, index):
        raise AssertionError("Split validation must not read flow fields")


class SplitTests(unittest.TestCase):
    def test_metadata_only_split_preserves_source_ids(self):
        split = resolve_training_split(MetadataOnlyDataset(), {
            "train_sample_ids": [100, 101], "calibration_sample_ids": [200],
            "test_sample_ids": [300], "train": ["8"], "calibration": ["9"], "test": ["10"],
            "fingerprint": "root-generated-fingerprint"})
        self.assertEqual(split["train_rows"], [0, 1])
        self.assertEqual(split["train_shape_ids"], [8])
        self.assertEqual(split["source_split_fingerprint"], "root-generated-fingerprint")
        changed = resolve_training_split(MetadataOnlyDataset(), [100])
        self.assertNotEqual(split["split_sha256"], changed["split_sha256"])

    def test_reject_geometry_leakage_even_with_disjoint_samples(self):
        with self.assertRaisesRegex(ValueError, "geometry IDs overlap"):
            resolve_training_split(MetadataOnlyDataset(), {"train_sample_ids": [100], "test_sample_ids": [101]})

    def test_reject_unknown_duplicate_and_missing_ids(self):
        for raw in ([999], [100, 100], [], [True], {"test_sample_ids": [300]},
                    {"train_sample_ids": [100], "train": [9]}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolve_training_split(MetadataOnlyDataset(), raw)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch training environment unavailable")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(7)
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
                self.final_layer = torch.nn.Module()
                self.final_layer.out_proj = torch.nn.Conv2d(4, 3, 1)

            def forward(self, geometry, code):
                return [self.final_layer.out_proj(self.backbone(geometry))]
        self.model = TinyModel()
        self.samples = [{"geometry": np.ones((3, 2, 4), dtype=np.float32),
                         "condition": np.array([2., .8], dtype=np.float32),
                         "fields": np.zeros((3, 2, 4), dtype=np.float32)}]

    def fit(self, mode="head"):
        from mdo_demo.adaptation import _fit_model
        return _fit_model(self.model, self.samples, torch=self.torch, device=self.torch.device("cpu"),
                          epochs=2, batch_size=1, lr=1e-3, seed=7, mode=mode)

    def test_only_output_projection_updates_and_backbone_buffers_stay_fixed(self):
        before = copy.deepcopy(self.model.state_dict())
        result = self.fit()
        self.assertTrue(result["frozen_state_verified_unchanged"])
        self.assertTrue(all(name.startswith("final_layer.out_proj.") for name in result["trainable_parameter_names"]))
        self.assertFalse(self.torch.equal(before["final_layer.out_proj.weight"], self.model.final_layer.out_proj.weight))
        for name, value in self.model.state_dict().items():
            if not name.startswith("final_layer.out_proj."):
                self.assertTrue(self.torch.equal(before[name], value), name)
        self.assertEqual(len(result["history"]), 2)

    def test_full_mode_enables_backbone(self):
        names = configure_trainable(self.model, "full")
        self.assertIn("backbone.0.weight", names)
        self.assertTrue(self.model.training)

    def test_nonfinite_predictions_abort(self):
        self.samples[0]["geometry"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "invalid surface"):
            self.fit()


if __name__ == "__main__":
    unittest.main()
