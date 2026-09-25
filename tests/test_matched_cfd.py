import tempfile
from pathlib import Path
import unittest

import numpy as np

from mdo_demo.io import sha256_file, write_json
from mdo_demo.matched_cfd import canonical_hash, load_bundle, mesh_options, paired_diagnostics, reference_from_native_blocks, validate_native_input


def native_input():
    return {"schema_version": 1, "case_name": "test", "condition": {"mach": .8, "alpha_deg": 2.},
            "geometry": {"SA": 35., "half_span": 3., "chords": [1.] * 7, "twists": [0.] * 7,
                         "DAz": [0.] * 8, "cst_u": [[.1] * 10] * 7, "cst_l": [[-.1] * 10] * 7}}


class MatchedCFDTests(unittest.TestCase):
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
        prediction = {"case_sha256": "a", "coefficients": {"CL": .5, "CD": .03, "CM": -.1}, "timing": {}}
        cfd = {"case_sha256": "a", "status": "ok", "convergence": {"converged": True},
               "coefficients": {"CL": .51, "CD": .031, "CM": -.11}, "timing": {}}
        cfd["group_coefficients"] = {"mainwing": dict(cfd["coefficients"])}
        result = paired_diagnostics(prediction, cfd)
        self.assertAlmostEqual(result["coefficient_absolute_difference"]["CL"], .01)
        self.assertIsNone(result["speedup"])
        self.assertFalse(result["equal_accuracy_verified"])
        cfd["case_sha256"] = "b"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            paired_diagnostics(prediction, cfd)
        cfd["case_sha256"] = "a"
        cfd["convergence"]["converged"] = False
        with self.assertRaisesRegex(ValueError, "Nonconverged"):
            paired_diagnostics(prediction, cfd)


if __name__ == "__main__":
    unittest.main()
