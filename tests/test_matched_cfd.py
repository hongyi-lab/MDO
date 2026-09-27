import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from mdo_demo.io import sha256_file, write_json
from mdo_demo.matched_cfd import canonical_hash, load_bundle, mesh_options, mesh_quality_from_root, paired_diagnostics, reference_from_native_blocks, reusable_mesh, validate_native_input


def native_input():
    return {"schema_version": 1, "case_name": "test", "condition": {"mach": .8, "alpha_deg": 2.},
            "geometry": {"SA": 35., "half_span": 3., "chords": [1.] * 7, "twists": [0.] * 7,
                         "DAz": [0.] * 8, "cst_u": [[.1] * 10] * 7, "cst_l": [[-.1] * 10] * 7}}


class MatchedCFDTests(unittest.TestCase):
    def test_mesh_reuse_requires_same_geometry_and_settings_even_if_flow_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh = root / "old.cgns"
            mesh.write_bytes(b"test identity only")
            identity = {"surface_sha256": "a", "condition": {"mach": .8, "reynolds": 2e7,
                        "reynolds_length_m": 1., "temperature_k": 300., "alpha_deg": 2.}, "reference": {"area_m2": 1.5}}
            options = mesh_options(root/"new.xyz", .8)
            source = {"status": "not_converged", "bundle_identity": identity, "mesh_options": options,
                      "mesh_quality": {"positive_volume_and_quality": True, "minimum_volume": 1e-14, "minimum_quality": .16},
                      "solver": {"options": {"gridFile": str(mesh)}}, "identity": {"volume_mesh_sha256": sha256_file(mesh)}}
            path = root/"result.json"
            write_json(path, source)
            self.assertEqual(reusable_mesh({"identity": identity}, path, options)[0], mesh)
            different = {**identity, "surface_sha256": "b"}
            with self.assertRaisesRegex(ValueError, "surface hash"):
                reusable_mesh({"identity": different}, path, options)
            with self.assertRaisesRegex(ValueError, "pyHyp settings"):
                reusable_mesh({"identity": identity}, path, {**options, "N": 41})
            mesh.write_bytes(b"tampered mesh")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reusable_mesh({"identity": identity}, path, options)

    def test_mesh_minima_broadcast_root_reduction_not_nonroot_uninitialized_values(self):
        root_values = (1e-13, .02)
        class FakeComm:
            def __init__(self, rank):
                self.rank = rank
            def bcast(self, values, root):
                if self.rank == 0:
                    self_value = values
                    self_test.assertEqual(self_value, root_values)
                else:
                    self_test.assertIsNone(values)
                return root_values
        self_test = self
        hyp = SimpleNamespace(hyp=SimpleNamespace(hypdata=SimpleNamespace(
            minvolumeoverall=root_values[0], minqualityoverall=root_values[1])))
        self.assertEqual(mesh_quality_from_root(hyp, FakeComm(0)), root_values)
        self.assertEqual(mesh_quality_from_root(None, FakeComm(7)), root_values)
        root_values = (0., .02)
        with self.assertRaisesRegex(RuntimeError, "Invalid generated mesh"):
            mesh_quality_from_root(None, FakeComm(7))

    def test_missing_native_constants_fail(self):
        raw = native_input()
        del raw["geometry"]["SA"]
        with self.assertRaisesRegex(ValueError, "Missing native"):
            validate_native_input(raw)

    def test_bad_geometry_rejected(self):
        for key, value in (("chords", [1.] * 3), ("twists", [float("nan")] * 7), ("half_span", -3.)):
            raw = native_input()
            raw["geometry"][key] = value
            with self.assertRaises(ValueError):
                validate_native_input(raw)

    def test_conditions_are_not_inferred(self):
        raw = native_input()
        raw["condition"] = {"mach": .8}
        with self.assertRaises(ValueError):
            validate_native_input(raw)

    def test_author_mesh_settings_and_layer_constraints(self):
        opts = mesh_options(Path("wing.xyz"), .85)
        self.assertEqual(opts["N"], 81)
        self.assertEqual(opts["BC"][1], {"jLow": "zSymm"})
        self.assertEqual(opts["families"][1], "mainwing")
        self.assertEqual(opts["families"][4], "trailingedge")
        self.assertEqual(opts["families"][9], "tip")
        self.assertGreater(opts["s0"], 0)
        self.assertLess(opts["s0"], 1e-4)
        for bad in (8, 12, 0):
            with self.assertRaises(ValueError):
                mesh_options(Path("wing.xyz"), .85, wall_normal_layers=bad)

    def test_modified_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "request.json"
            write_json(p, {"schema_version": 1, "identity": {"a": 1}, "case_sha256": canonical_hash({"a": 2})})
            with self.assertRaisesRegex(ValueError, "identity hash mismatch"):
                load_bundle(p)

    def test_modified_surface_does_not_keep_valid_case_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {}
            request = {"schema_version": 1, "identity": identity}
            for key, filename, hash_key in (("surface_path", "wing.xyz", "surface_sha256"),
                                            ("fm_input_path", "fm_input.npz", "fm_input_sha256"),
                                            ("native_input_path", "native_input.json", "native_input_sha256")):
                (root / filename).write_bytes(b"test fixture")
                identity[hash_key] = sha256_file(root / filename)
                request[key] = filename
            request["case_sha256"] = canonical_hash(identity)
            write_json(root / "request.json", request)
            self.assertEqual(load_bundle(root / "request.json")["case_sha256"], request["case_sha256"])
            (root / "wing.xyz").write_bytes(b"changed geometry")
            with self.assertRaisesRegex(ValueError, "surface_path"):
                load_bundle(root / "request.json")

    def test_surface_sampling_preserves_span_and_foil_winding(self):
        angle = np.linspace(0, 2*np.pi, 233)
        blocks = []
        for spans in (np.linspace(.3, 1.2, 5), np.linspace(1.2, 3., 9)):
            ring = np.empty((233, len(spans), 1, 3))
            ring[:, :, 0, 0] = (.5 + .5*np.cos(angle))[:, None]
            ring[:, :, 0, 1] = (-.08*np.sin(angle))[:, None]
            ring[:, :, 0, 2] = -spans[None, :]
            blocks.extend([ring[:113], ring[112:121], ring[120:], ring[-2:]])
        blocks += [blocks[0]] * 5  # tip is deliberately not consumed by the FM sampler
        geometry, metadata = reference_from_native_blocks(blocks)
        self.assertEqual(geometry.shape, (3,129,257))
        self.assertTrue(np.all(np.diff(geometry[2,:,0]) > 0))
        self.assertLess(float(geometry[1,64,64]), 0.)
        self.assertGreater(float(geometry[1,64,192]), 0.)
        self.assertFalse(metadata["calibration_transfer_allowed"])
        self.assertAlmostEqual(float(geometry[0,64,128]), 0.)

    def test_pair_requires_identity_and_convergence(self):
        prediction = {"case_sha256": "a", "coefficients": {"CL": .5, "CD": .03, "CM": -.1}, "timing": {},
                      "frame_contract": {"verified": True}, "sampling_contract": {"verified": True}}
        cfd = {"case_sha256": "a", "status": "ok", "convergence": {"converged": True},
               "coefficients": {"CL": .51, "CD": .031, "CM": -.11}, "timing": {},
               "fm_validation_status": "aligned_verified"}
        cfd["group_coefficients"] = {"mainwing": dict(cfd["coefficients"])}
        result = paired_diagnostics(prediction, cfd)
        self.assertAlmostEqual(result["coefficient_absolute_difference"]["CL"], .01)
        self.assertIsNone(result["speedup"])
        self.assertFalse(result["equal_accuracy_verified"])
        cfd["fm_validation_status"] = "native_to_model_frame_and_sampling_unaligned"
        with self.assertRaisesRegex(ValueError, "explicitly unaligned"):
            paired_diagnostics(prediction, cfd)
        cfd.pop("fm_validation_status")
        unknown = paired_diagnostics(prediction, cfd)
        self.assertIsNone(unknown["coefficient_absolute_difference"])
        self.assertFalse(unknown["coordinate_sampling_contract_verified"])
        self.assertFalse(unknown["accuracy_eligible"])
        cfd["fm_validation_status"] = "aligned_verified"
        prediction.pop("sampling_contract")
        self.assertIsNone(paired_diagnostics(prediction, cfd)["coefficient_absolute_difference"])
        cfd["case_sha256"] = "b"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            paired_diagnostics(prediction, cfd)
        cfd["case_sha256"] = "a"
        cfd["convergence"]["converged"] = False
        with self.assertRaisesRegex(ValueError, "Nonconverged"):
            paired_diagnostics(prediction, cfd)


if __name__ == "__main__":
    unittest.main()
