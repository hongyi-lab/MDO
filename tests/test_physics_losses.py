"""Meaningful checks of numerical conventions, conservation, and gradients."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from mdo_demo.physics_losses import mdo_loss, surface_loads

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "Optional model dependencies not installed")
class PhysicsLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        x = torch.linspace(0, 1, 6, dtype=torch.float64)
        z = torch.tensor([0.0, 0.2, 0.6, 1.0], dtype=torch.float64)
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        yy = 0.04 * torch.sin(xx * torch.pi) + 0.01 * zz * xx
        self.geometry = torch.stack((xx + 0.2 * zz, yy, zz)).unsqueeze(0)
        generator = torch.Generator().manual_seed(432)
        self.fields = torch.randn((1, 3, 3, 5), dtype=torch.float64, generator=generator) * 0.05
        self.fields[:, 0] -= 0.5

    def test_integral_of_load_density_matches_global_coefficients(self):
        output = surface_loads(self.fields, self.geometry, [3.0], [0.9])
        torch.testing.assert_close(output["spanwise_width"].sum(dim=1), torch.ones(1, dtype=torch.float64))
        for channel, name in enumerate(("spanwise_lift", "spanwise_drag")):
            reconstructed = (output[name] * output["spanwise_width"]).sum(dim=1)
            torch.testing.assert_close(reconstructed, output["coefficients"][:, channel], rtol=1e-14, atol=1e-14)

    def test_zero_loss_on_identical_fields_and_detached_target(self):
        predicted = self.fields.clone().requires_grad_()
        target = self.fields.clone().requires_grad_()
        result = mdo_loss(predicted, target, self.geometry, 3.0, 0.9)
        self.assertEqual(set(result), {"total", "field", "cl", "cd", "spanwise"})
        for term in result.values():
            self.assertEqual(float(term.detach()), 0.0)
        result["total"].backward()
        self.assertIsNotNone(predicted.grad)
        self.assertIsNone(target.grad)

    def test_friction_and_pressure_gradients_match_central_differences(self):
        prediction = (self.fields + 0.013).requires_grad_()
        loss = mdo_loss(prediction, self.fields, self.geometry, 3.0, 0.9)["total"]
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        eps = 1e-6
        for channel in range(3):
            position = (0, channel, 1, 2)
            plus, minus = prediction.detach().clone(), prediction.detach().clone()
            plus[position] += eps
            minus[position] -= eps
            value_plus = mdo_loss(plus, self.fields, self.geometry, 3.0, 0.9)["total"]
            value_minus = mdo_loss(minus, self.fields, self.geometry, 3.0, 0.9)["total"]
            finite_difference = float((value_plus - value_minus) / (2 * eps))
            self.assertAlmostEqual(float(prediction.grad[position]), finite_difference, delta=1e-7)

    def test_batched_angles_areas_and_geometries_match_single_calls(self):
        fields = torch.cat((self.fields, self.fields * 1.2))
        vertices = torch.cat((self.geometry, self.geometry * 1.3))
        result = surface_loads(fields, vertices, [2.0, 5.0], [0.9, 1.4])
        for row, (angle, area) in enumerate(((2.0, 0.9), (5.0, 1.4))):
            single = surface_loads(fields[row:row + 1], vertices[row:row + 1], angle, area)
            for key in result:
                torch.testing.assert_close(result[key][row:row + 1], single[key])

    def test_invalid_metadata_is_rejected(self):
        for area in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                surface_loads(self.fields, self.geometry, 3.0, area)
        with self.assertRaisesRegex(ValueError, "vertex_geometry"):
            surface_loads(self.fields, self.geometry[:, :, :-1, :-1], 3.0, 1.0)
        with self.assertRaisesRegex(ValueError, "increasing"):
            surface_loads(self.fields, self.geometry.flip(2), 3.0, 1.0)
        with self.assertRaisesRegex(ValueError, "scales"):
            mdo_loss(self.fields, self.fields, self.geometry, 3.0, 1.0, coefficient_scales=(0, .01))

    def test_against_pinned_numpy_postprocessor_using_training_only(self):
        dataset_dir = ROOT / "assets" / "CRMpert"
        split_path = ROOT / "configs" / "protocol_v1_split.json"
        if not dataset_dir.is_dir() or not split_path.is_file() or not (ROOT / "external" / "cfdpost").is_dir():
            self.skipTest("Real training subset and pinned cfdpost are optional integration assets")
        from mdo_demo.dataset import load_dataset
        from mdo_demo.aerotransformer import integrate_coefficients
        split = json.loads(split_path.read_text(encoding="utf-8"))
        training_ids = set(split["splits"]["train"]["source_sample_ids"])
        dataset = load_dataset(dataset_dir)
        rows = [row for row, sample_id in enumerate(dataset.sample_ids) if sample_id in training_ids]
        self.assertGreaterEqual(len(rows), 2, "Need at least two explicit training samples for integration check")
        # Do not read, tune against, or even load any calibration/holdout fields.
        for row in rows[:4]:
            sample = dataset.sample(row)
            fields = np.array(dataset.fields[row], dtype=np.float64)
            expected = integrate_coefficients(sample, fields)
            result = surface_loads(
                torch.tensor(fields).unsqueeze(0),
                torch.tensor(sample["original_geometry"]).unsqueeze(0),
                float(sample["condition"][0]), sample["ref_area"])
            np.testing.assert_allclose(result["coefficients"][0].numpy(),
                                       [expected["CL"], expected["CD"]], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(result["CM"][0].numpy(), expected["CM"], rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
