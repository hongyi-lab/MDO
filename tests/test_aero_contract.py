"""Physical sign, frame, coverage, dimensionalization and qualification tests."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mdo_demo.aero_contract import (
    MAINWING_PATCHES, _quad_geometry, assert_same_physical_request,
    from_fm_surface, from_native_export, load_loads, make_request,
    match_native_cells, native_mainwing_vertices, read_native_surface, save_loads, validate_request,
)
from mdo_demo.io import sha256_file, write_json
from mdo_demo.matched_cfd import (
    ADFLOW_SURFACE_SOURCE_SHA256, NATIVE_INPUT_FRAME, native_to_model_sample,
)


def request(geometry_sha256="a"*64, **condition_changes):
    cond = dict(alpha_deg=0., mach=.8, reynolds=20_000_000.,
                reynolds_length_m=1., temperature_k=300., dynamic_pressure_pa=100.)
    cond.update(condition_changes)
    return make_request("b"*64, geometry_sha256, condition=cond,
                        reference=dict(area_m2=2.7, chord_m=1., moment_center_m=[.25,0.,0.]),
                        coverage=dict(families=["mainwing"], closed_TE=False, span_bounds_m=[-3.,-.3]))


def model_sample(req):
    # Open blunt-TE loop, increasing mirrored span, lower TE -> LE -> upper TE.
    angle = np.linspace(.04, 2*np.pi-.04, 257)
    v = np.empty((3,129,257))
    v[0] = .5+.5*np.cos(angle)
    v[1] = -.1*np.sin(angle)
    v[2] = np.linspace(.3,3.,129)[:,None]
    centers = .25*(v[:,1:,1:]+v[:,1:,:-1]+v[:,:-1,1:]+v[:,:-1,:-1])
    identity = req["identity"]
    result = native_to_model_sample(
        dict(original_geometry=v, geometry=centers.astype(np.float32),
             condition=np.array([identity["condition"]["alpha_deg"], .8], dtype=np.float32),
             ref_area=2.7), input_frame=NATIVE_INPUT_FRAME, native_reference=identity["reference"])
    result["frame_contract"]["source_bundle_case_sha256"] = identity["case_id"]
    result["sampling_contract"] = dict(version="native_open_mainwing_arc_sampling_v1", verified=False,
                                       physical_surface_family="mainwing", native_patches=MAINWING_PATCHES.copy())
    return result


def native_fixture(root):
    # Rectangular open-mainwing prism. Analytic constant-pressure/friction loads
    # include the mainwing's lower/front/upper panels; the TE is explicitly out.
    sections = [(1.,-.1),(0.,-.1),(0.,.1),(1.,.1),(1.,-.1)]
    blocks = []
    for z0,z1 in ((-.3,-1.5),(-1.5,-3.)):
        for a,b in zip(sections[:-1], sections[1:]):
            vertices = np.zeros((3,2,2))
            vertices[0] = [a[0],b[0]]
            vertices[1] = [a[1],b[1]]
            vertices[2] = [[z0,z0],[z1,z1]]
            blocks.append(vertices)
    for i in range(5):
        vertices = blocks[0].copy()
        vertices[2] -= 4+i
        blocks.append(vertices)
    wing = root/"wing.xyz"
    wing.write_text("13\n"+"2 2 1\n"*13+"\n".join(
        " ".join(format(x,".17g") for x in block.reshape(-1)) for block in blocks),encoding="ascii")
    req = request(sha256_file(wing))
    points = np.concatenate([_quad_geometry(b)[0].reshape(-1,3) for b in blocks])
    index = np.arange(13)[::-1]
    patch = index+1
    path = root/"native.npz"
    np.savez_compressed(path, centers=points[index], cp=np.ones(13),
                        cf_xyz=np.tile([0.,0.,.01],(13,1)), native_cell_index=index,
                        native_patch=patch, mainwing_mask=np.isin(patch,MAINWING_PATCHES))
    meta = dict(case_sha256=req["identity"]["case_id"], output_sha256=sha256_file(path),
                one_to_one_native_cell_match=True, coordinate_convention="native CFD negative-z span",
                pressure_convention_audit=dict(verified=True,source_sha256=ADFLOW_SURFACE_SOURCE_SHA256,
                                               reviewed_source_sha256=ADFLOW_SURFACE_SOURCE_SHA256),
                postprocess_seconds=.01)
    cond = copy.deepcopy(req["identity"]["condition"])
    cond.pop("dynamic_pressure_pa")
    # Analytic result: front pressure Fx=q*0.2*2.7; Cf_z area=(2+0.2)*2.7.
    truth = dict(CL=0.,CD=.2,CMx=0.,CMy=-.3345,CMz=0.,CM=0.)
    result = dict(status="ok", case_sha256=req["identity"]["case_id"],
                  identity=dict(condition=cond, reference=copy.deepcopy(req["identity"]["reference"]),
                                geometry_sha256=sha256_file(wing)), solved_alpha_deg=0.,
                  convergence=dict(converged=True, solve_failed=False, fatal_failed=False,
                                   residual_relative_to_freestream=1e-12, required_l2_convergence=1e-10),
                  function_groups=dict(mainwing="mainwing"),group_coefficients=dict(mainwing=truth),
                  timing=dict(scope="synthetic integration fixture; not a measured CFD solve",solve_seconds=0.))
    return req, wing, path, meta, result


class AeroContractTests(unittest.TestCase):
    def test_identity_hash_rejects_physical_mutation(self):
        req = request()
        self.assertEqual(validate_request(req)["coverage"]["excluded_native_patches"],[4,8,9,10,11,12,13])
        req["identity"]["condition"]["dynamic_pressure_pa"] *= 2
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            validate_request(req)

    def test_fm_matches_pinned_integral_and_recomputes_axial_moment(self):
        from mdo_demo.aerotransformer import integrate_coefficients
        req = request(alpha_deg=2.)
        sample = model_sample(req)
        fields = np.random.default_rng(41).normal(0,.1,(3,128,256))
        original = copy.deepcopy(sample)
        out = from_fm_surface(req,sample,fields,timing=dict(scope="unit integration check",predict_seconds=0.))
        expected = integrate_coefficients(sample,fields)
        np.testing.assert_allclose([out["coefficients"]["CL"],out["coefficients"]["CD"]],
                                   [expected["CL"],expected["CD"]],rtol=1e-6,atol=1e-9)
        point = out["panel_points_m"]
        force = out["panel_forces_N"]
        np.testing.assert_allclose(out["total_moment_Nm"],np.cross(point-[.25,0.,0.],force).sum(axis=0))
        self.assertLess(point[:,2].max(),0.)
        np.testing.assert_array_equal(sample["original_geometry"],original["original_geometry"])
        self.assertFalse(out["accuracy_validated"])
        self.assertFalse(out["sampling_contract"]["verified"])
        doubled = request(alpha_deg=2.,dynamic_pressure_pa=200.)
        out2 = from_fm_surface(doubled,sample,fields,timing=dict(scope="unit integration check"))
        np.testing.assert_allclose(out2["panel_forces_N"],force*2,rtol=1e-13)
        np.testing.assert_allclose(list(out2["coefficients"].values()),list(out["coefficients"].values()))
        with self.assertRaisesRegex(ValueError,"same physical request"):
            assert_same_physical_request(out,out2)

    def test_fm_cannot_silently_change_unconditioned_reynolds_or_temperature(self):
        sample = model_sample(request())
        for key,value in (("reynolds",5e6),("temperature_k",280.),("reynolds_length_m",2.)):
            with self.assertRaisesRegex(ValueError,"fixed training condition"):
                from_fm_surface(request(**{key:value}),sample,np.zeros((3,128,256)),timing=dict(scope="test"))

    def test_fm_rejects_changed_tensor_coordinates_or_coverage(self):
        req = request()
        sample = model_sample(req)
        fields = np.zeros((3,128,256))
        sample["geometry"][0,0,0] += 1
        with self.assertRaisesRegex(ValueError,"center inputs"):
            from_fm_surface(req,sample,fields,timing=dict(scope="test"))
        sample = model_sample(req)
        sample["frame_contract"]["model_input_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"model-input hash"):
            from_fm_surface(req,sample,fields,timing=dict(scope="test"))
        sample = model_sample(req)
        sample["sampling_contract"]["native_patches"].append(4)
        with self.assertRaisesRegex(ValueError,"physical surface coverage"):
            from_fm_surface(req,sample,fields,timing=dict(scope="test"))

    def test_native_pressure_sign_friction_units_permutation_and_reference_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = native_fixture(Path(tmp))
            out = from_native_export(*args)
            np.testing.assert_allclose(out["total_force_N"],[54.,0.,5.94],atol=1e-12)
            np.testing.assert_allclose(out["total_moment_Nm"],[0.,-90.315,0.],atol=1e-12)
            self.assertTrue(out["integration_audit"]["passed"])
            self.assertTrue(out["reference_eligible"])
            self.assertEqual(len(out["panel_points_m"]),6)
            self.assertEqual(native_mainwing_vertices(args[1]).shape,(3,3,4))
            self.assertTrue(assert_same_physical_request(out,out))

    def test_unconverged_and_spoofed_success_cannot_enter_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = native_fixture(Path(tmp))
            args[-1]["convergence"]["residual_relative_to_freestream"] = 1e-4
            with self.assertRaisesRegex(ValueError,"Unconverged"):
                from_native_export(*args)
            diagnostic = from_native_export(*args,allow_unconverged_interface_audit=True)
            self.assertEqual(diagnostic["status"],"diagnostic_only")
            self.assertTrue(diagnostic["integration_audit"]["passed"])
            self.assertFalse(diagnostic["interface_compatible"])
            self.assertFalse(diagnostic["reference_eligible"])
            with self.assertRaisesRegex(ValueError,"interface checks"):
                assert_same_physical_request(diagnostic,diagnostic)

    def test_wrong_native_load_sign_is_flagged_not_qualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = native_fixture(Path(tmp))
            args[-1]["group_coefficients"]["mainwing"]["CD"] = -.2
            out = from_native_export(*args)
            self.assertEqual(out["status"],"load_audit_failed")
            self.assertFalse(out["reference_eligible"])

    def test_unknown_pressure_code_or_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = native_fixture(Path(tmp))
            args[3]["pressure_convention_audit"]["source_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError,"pressure convention"):
                from_native_export(*args)
            args[3]["pressure_convention_audit"]["source_sha256"] = ADFLOW_SURFACE_SOURCE_SHA256
            args[-1]["identity"]["geometry_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError,"another native surface"):
                from_native_export(*args)

    def test_archive_roundtrip_and_integral_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = from_native_export(*native_fixture(root))
            target = save_loads(out,root/"loads.json")
            restored = load_loads(target)
            np.testing.assert_array_equal(out["panel_forces_N"],restored["panel_forces_N"])
            self.assertTrue(assert_same_physical_request(out,restored))
            with self.assertRaisesRegex(ValueError,"fresh archive"):
                save_loads(out,target)
            import json
            meta = json.loads(target.read_text(encoding="utf-8"))
            meta["total_moment_Nm"][1] += 1
            write_json(target,meta)
            with self.assertRaisesRegex(ValueError,"total_moment_Nm"):
                load_loads(target)

    def test_native_storage_rounding_is_reconstructed_not_a_free_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _,wing,*_ = native_fixture(root)
            blocks = read_native_surface(wing)
            # Large offset makes float32 storage loss exceed geometric tolerance.
            for block in blocks:
                block[0] += 1000.123456789
            wing.write_text("13\n"+"2 2 1\n"*13+"\n".join(
                " ".join(format(x,".17g") for x in b.reshape(-1)) for b in blocks))
            centers = np.concatenate([_quad_geometry(b.astype(np.float32).astype(float))[0].reshape(-1,3) for b in blocks])
            zone_ids = np.arange(1,14)
            info = [dict(zone=i,selected_for_body_loads=True,
                         coordinate_storage_types=dict.fromkeys(("CoordinateX","CoordinateY","CoordinateZ"),"RealSingle"),
                         coordinate_storage_roundoff_bound=dict(euclidean_m=999.)) for i in zone_ids]
            index,audit = match_native_cells(wing,centers,zone_ids,info)
            np.testing.assert_array_equal(index,np.arange(13))
            self.assertGreater(audit["cell_coordinate_match_max_m"],audit["base_geometry_tolerance_m"])
            self.assertLess(audit["cell_coordinate_match_tolerance_m"],.001)
            shifted = centers.copy()
            shifted[0,0] += 5*audit["base_geometry_tolerance_m"]
            # Shift still fits broad float32 uncertainty; exact rounding evidence
            # rejects it rather than trusting claimed precision or tolerance.
            with self.assertRaisesRegex(ValueError,"beyond recorded storage rounding"):
                match_native_cells(wing,shifted,zone_ids,info)
            duplicate = centers.copy()
            duplicate[0] = duplicate[1]
            with self.assertRaisesRegex(ValueError,"one-to-one"):
                match_native_cells(wing,duplicate,zone_ids,info)


if __name__ == "__main__":
    unittest.main()
