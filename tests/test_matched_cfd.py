import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from mdo_demo.io import read_json, sha256_file, write_json
from mdo_demo.matched_cfd import (AEROTRANSFORMER_REVISION, BASELINE_TWIST_DEGREES,
                                 FRAME_TRANSFORM_VERSION, MODEL_INPUT_FRAME,
                                 NATIVE_INPUT_FRAME, NATIVE_SAMPLING_METHOD,
                                 canonical_hash, load_bundle, mesh_options, mesh_quality_from_root,
                                 native_to_model_sample, paired_diagnostics, prediction_sample,
                                 reference_from_native_blocks, reusable_mesh, validate_native_input)


def native_input():
    return {"schema_version": 1, "case_name": "test", "condition": {"mach": .8, "alpha_deg": 2.},
            "geometry": {"SA": 35., "half_span": 3., "chords": [1.] * 7, "twists": [0.] * 7,
                         "DAz": [0.] * 8, "cst_u": [[.1] * 10] * 7, "cst_l": [[-.1] * 10] * 7}}


def native_frame_sample():
    angle = np.linspace(0, 2*np.pi, 257)
    points = np.empty((3, 129, 257))
    points[0] = .5 + .5*np.cos(angle)
    points[1] = -.08*np.sin(angle)
    points[2] = np.linspace(.3, 3., 129)[:, None]
    centers = .25 * (points[:, 1:, 1:] + points[:, 1:, :-1]
                      + points[:, :-1, 1:] + points[:, :-1, :-1])
    return {"original_geometry": points, "geometry": centers.astype(np.float32),
            "condition": np.array([2., .8], dtype=np.float32), "ref_area": 1.5}


class MatchedCFDTests(unittest.TestCase):
    def test_native_joint_rotation_copies_input_and_preserves_wind_axis_forces(self):
        source = native_frame_sample()
        before = {key: value.copy() for key, value in source.items() if isinstance(value, np.ndarray)}
        result = native_to_model_sample(source, input_frame=NATIVE_INPUT_FRAME,
                                       native_reference={"area_m2": 1.5, "moment_center_m": [.25, 0, 0]})
        self.assertAlmostEqual(float(result["condition"][0]), 8.7166, places=5)
        self.assertEqual(float(result["condition"][1]), float(source["condition"][1]))
        frame = result["frame_contract"]
        rotation = np.asarray(frame["rotation_matrix_native_to_model"])
        np.testing.assert_allclose(np.einsum("ab,bij->aij", rotation.T, result["original_geometry"]),
                                   source["original_geometry"], rtol=1e-12, atol=1e-12)
        force = np.array([.03, .5, .08])
        def wind_axes(vector, alpha):
            a = np.radians(alpha)
            return np.array([np.cos(a)*vector[0]+np.sin(a)*vector[1],
                             -np.sin(a)*vector[0]+np.cos(a)*vector[1]])
        np.testing.assert_allclose(wind_axes(force, 2.),
                                   wind_axes(rotation @ force, 2. + BASELINE_TWIST_DEGREES), atol=1e-14)
        np.testing.assert_allclose(frame["equivalent_model_moment_center_m"], rotation @ [.25, 0, 0])
        self.assertFalse(frame["CM_comparable"])
        self.assertFalse(frame["physics_changed"])
        self.assertEqual(frame["version"], FRAME_TRANSFORM_VERSION)
        for key, array in before.items():
            np.testing.assert_array_equal(source[key], array)
        np.testing.assert_array_equal(result["native_geometry"], source["geometry"])

    def test_native_rotation_does_not_reinterpret_671_as_model_671(self):
        source = native_frame_sample()
        source["condition"][0] = 6.71
        result = native_to_model_sample(source, input_frame=NATIVE_INPUT_FRAME,
                                       native_reference={"area_m2": 1.5, "moment_center_m": [.25, 0, 0]})
        self.assertAlmostEqual(float(result["condition"][0]), 13.4266, places=5)

    def test_released_unknown_and_double_transformed_inputs_are_rejected(self):
        source = native_frame_sample()
        reference = {"area_m2": 1.5, "moment_center_m": [.25, 0, 0]}
        for wrong_frame in (MODEL_INPUT_FRAME, "released_CRMpert", "unknown"):
            with self.assertRaisesRegex(ValueError, "unknown or already transformed"):
                native_to_model_sample(source, input_frame=wrong_frame, native_reference=reference)
        result = native_to_model_sample(source, input_frame=NATIVE_INPUT_FRAME, native_reference=reference)
        with self.assertRaisesRegex(ValueError, "already transformed"):
            native_to_model_sample(result, input_frame=NATIVE_INPUT_FRAME, native_reference=reference)
        source["geometry"][0, 0, 0] += .1
        with self.assertRaisesRegex(ValueError, "does not match"):
            native_to_model_sample(source, input_frame=NATIVE_INPUT_FRAME, native_reference=reference)

    def test_prediction_time_transform_keeps_legacy_cfd_bundle_hash_and_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = native_frame_sample()
            np.savez(root / "fm_input.npz", **raw)
            (root / "wing.xyz").write_bytes(b"fixture only, not a physical mesh")
            write_json(root / "native_input.json", native_input())
            identity = {"surface_sha256": sha256_file(root / "wing.xyz"),
                        "fm_input_sha256": sha256_file(root / "fm_input.npz"),
                        "native_input_sha256": sha256_file(root / "native_input.json"),
                        "condition": {"alpha_deg": 2., "mach": float(raw["condition"][1])},
                        "reference": {"area_m2": 1.5, "moment_center_m": [.25, 0, 0]},
                        "surface_sampling": {"method": NATIVE_SAMPLING_METHOD},
                        "recipe_revision": AEROTRANSFORMER_REVISION}
            request = {"schema_version": 1, "case_sha256": canonical_hash(identity), "identity": identity,
                       "fm_input_path": "fm_input.npz", "surface_path": "wing.xyz",
                       "native_input_path": "native_input.json"}
            path = root / "request.json"
            write_json(path, request)
            before = {item.name: sha256_file(item) for item in root.iterdir()}
            sample, manifest = prediction_sample(path)
            self.assertEqual(manifest["case_sha256"], request["case_sha256"])
            self.assertEqual(read_json(path), request)
            self.assertEqual(before, {item.name: sha256_file(item) for item in root.iterdir()})
            self.assertTrue(sample["frame_contract"]["verified"])
            self.assertTrue(sample["frame_contract"]["legacy_native_frame_identified_from_pinned_recipe"])
            self.assertFalse(sample["sampling_contract"]["verified"])
            self.assertFalse(sample["sampling_contract"]["accuracy_eligible"])
            self.assertEqual(manifest["fm_validation_status"], "frame_aligned_sampling_unverified")
            self.assertAlmostEqual(float(sample["condition"][0]), 8.7166, places=5)
            self.assertEqual(float(sample["native_condition"][0]), 2.)
            # Self-consistent outer hashes do not excuse a physical condition mismatch.
            request["identity"]["condition"]["alpha_deg"] = 4.
            request["case_sha256"] = canonical_hash(request["identity"])
            write_json(path, request)
            with self.assertRaisesRegex(ValueError, "condition identity differ"):
                prediction_sample(path)

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

    def test_new_frame_prediction_reuses_cfd_but_never_certifies_unknown_sampling(self):
        prediction = {"case_sha256": "same-physical-case", "coefficients": {"CL": .5, "CD": .03},
                      "timing": {}, "fm_validation_status": "frame_aligned_sampling_unverified",
                      "frame_contract": {"verified": True, "version": FRAME_TRANSFORM_VERSION,
                                         "source_bundle_case_sha256": "same-physical-case"},
                      "sampling_contract": {"verified": False, "physical_surface_family": "mainwing"}}
        cfd = {"case_sha256": "same-physical-case", "status": "ok", "convergence": {"converged": True},
               "coefficients": {"CL": .52, "CD": .033}, "timing": {},
               "group_coefficients": {"mainwing": {"CL": .51, "CD": .031}},
               "fm_validation_status": "native_to_model_frame_and_sampling_unaligned"}
        result = paired_diagnostics(prediction, cfd)
        self.assertTrue(result["frame_alignment_verified"])
        self.assertFalse(result["coordinate_sampling_contract_verified"])
        self.assertFalse(result["accuracy_eligible"])
        self.assertFalse(result["matched_speedup_eligible"])
        self.assertIsNone(result["speedup"])
        self.assertIsNone(result["coefficient_absolute_difference"])
        self.assertAlmostEqual(result["unverified_coefficient_difference"]["CL"], .01)
        prediction["frame_contract"]["source_bundle_case_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "verified new prediction"):
            paired_diagnostics(prediction, cfd)


if __name__ == "__main__":
    unittest.main()
