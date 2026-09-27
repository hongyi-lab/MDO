"""Checkpoint-contract tests with labeled placeholder files; no CFD is simulated."""
import copy
from pathlib import Path
import tempfile
import unittest

from mdo_demo.cfd import convergence_report, solver_options, validate_request
from mdo_demo.cfd_restart import audit_restart, validate_restart_source
from mdo_demo.io import sha256_file, write_json
from mdo_demo.matched_cfd import canonical_hash


class CheckpointRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.output = self.root / "cfd" / "solver_surface"
        self.output.mkdir(parents=True)
        self.mesh = self.output.parent / "wing_vol.cgns"
        self.mesh.write_bytes(b"placeholder original mesh for contract test only")
        self.checkpoint = self.output / "wing_000_vol.cgns"
        self.checkpoint.write_bytes(b"placeholder volume checkpoint; no real CGNS assertion")
        for name in ("wing.xyz", "fm_input.npz", "native_input.json"):
            (self.bundle / name).write_bytes(name.encode())
        condition = {"mach": .8, "alpha_deg": 0., "reynolds": 20e6,
                     "reynolds_length_m": 1., "temperature_k": 300.}
        reference = {"area_m2": 1.5, "chord_m": 1., "moment_center_m": [.25, 0., 0.]}
        bundle_identity = {"case_name": "test", "surface_sha256": sha256_file(self.bundle / "wing.xyz"),
                           "fm_input_sha256": sha256_file(self.bundle / "fm_input.npz"),
                           "native_input_sha256": sha256_file(self.bundle / "native_input.json"),
                           "condition": condition, "reference": reference}
        manifest = {"schema_version": 1, "case_sha256": canonical_hash(bundle_identity),
                    "identity": bundle_identity, "surface_path": "wing.xyz",
                    "fm_input_path": "fm_input.npz", "native_input_path": "native_input.json"}
        write_json(self.bundle / "request.json", manifest)
        request = validate_request({"schema_version": 1, "problem_id": "checkpoint-test",
            "geometry_id": "test", "geometry_sha256": bundle_identity["surface_sha256"],
            "mesh_path": str(self.mesh), "condition": condition, "reference": reference,
            "numerics": {"preset": "robust_rans_late_nk", "max_cycles": 9000}})
        conv = convergence_report((1e8, 1e8, .012369663), tolerance=1e-10,
            solve_failed=False, fatal_failed=False, coefficients={"CL": .7, "CD": .03}, solved_alpha_deg=0.)
        self.source = {"schema_version": 1, "backend": "adflow", "status": "not_converged",
            "case_sha256": manifest["case_sha256"], "volume_case_sha256": request["case_sha256"],
            "identity": request["identity"], "bundle_identity": bundle_identity,
            "coefficients": {"CL": .7, "CD": .03}, "group_coefficients": {"mainwing": {"CL": .7, "CD": .03}},
            "function_groups": {"mainwing": "mainwing"}, "convergence": conv,
            "configured_max_internal_iterations": 9000,
            "solver": {"options": solver_options(request, self.output)},
            "surface_output_directory": str(self.output), "provenance": {},
            "timing": {"total_seconds": 9500., "live_mesh_and_cfd_seconds": 9571., "mpi_ranks": 8}}
        self.source_path = self.output.parent / "result.json"
        write_json(self.source_path, self.source)

    def validate(self):
        return validate_restart_source(self.source_path, self.bundle, self.checkpoint, sha256_file(self.checkpoint))

    def restarted(self, final=.005):
        request, _ = self.validate()
        result = copy.deepcopy(self.source)
        result.pop("volume_case_sha256")
        result["case_sha256"] = request["case_sha256"]
        result["provenance"] = request["provenance"]
        result["solver"]["options"]["restartFile"] = str(self.checkpoint.resolve())
        result["convergence"] = convergence_report((1e8, .012369663, final), tolerance=1e-10,
            solve_failed=False, fatal_failed=False, coefficients=result["coefficients"], solved_alpha_deg=0.)
        result["status"] = "ok" if result["convergence"]["converged"] else "not_converged"
        result["timing"] = {"total_seconds": 40., "mpi_ranks": 8}
        return result

    def test_valid_source_preserves_physics_and_attaches_hashes(self):
        request, provenance = self.validate()
        self.assertEqual(request["identity"], self.source["identity"])
        self.assertEqual(request["case_sha256"], self.source["volume_case_sha256"])
        self.assertEqual(request["provenance"]["restart"], provenance)
        self.assertEqual(provenance["checkpoint_sha256"], sha256_file(self.checkpoint))
        self.assertEqual(provenance["source_result_file_sha256"], sha256_file(self.source_path))

    def test_ank_polish_changes_algorithm_without_relaxing_case_or_tolerance(self):
        request, _ = self.validate()
        original_identity = copy.deepcopy(request['identity'])
        request['numerics'].update(preset='robust_rans_ank_polish', max_cycles=1200)
        options = solver_options(request, self.output)
        self.assertEqual(request['identity'], original_identity)
        self.assertFalse(options['useNKSolver'])
        self.assertTrue(options['useANKSolver'])
        self.assertEqual(options['ANKCoupledSwitchTol'], 1e-16)
        self.assertEqual(options['ANKSecondOrdSwitchTol'], 1e-3)
        self.assertEqual(options['L2Convergence'], 1e-10)
        self.assertEqual(options['L2ConvergenceRel'], 1e-16)
        self.assertEqual(options['nCycles'], 1200)
        self.assertTrue(options['writeVolumeSolution'])
        self.assertEqual(options['solutionPrecision'], 'double')

    def test_rejects_modified_original_mesh(self):
        self.mesh.write_bytes(b"changed mesh")
        with self.assertRaisesRegex(ValueError, "mesh hash"):
            self.validate()

    def test_rejects_different_source_bundle_and_physics(self):
        for change in ("case", "condition", "reference", "geometry"):
            changed = copy.deepcopy(self.source)
            if change == "case": changed["case_sha256"] = "a" * 64
            if change == "condition": changed["identity"]["condition"]["alpha_deg"] = 1.
            if change == "reference": changed["identity"]["reference"]["area_m2"] = 2.
            if change == "geometry": changed["identity"]["geometry_sha256"] = "b" * 64
            write_json(self.source_path, changed)
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate()

    def test_rejects_wrong_checkpoint_hash_path_name_and_precision(self):
        with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
            validate_restart_source(self.source_path, self.bundle, self.checkpoint, "0" * 64)
        for path in (self.mesh, self.output / "wing_000_surf.cgns", self.root / "wing_000_vol.cgns"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Checkpoint"):
                validate_restart_source(self.source_path, self.bundle, path, sha256_file(self.checkpoint))
        self.source["solver"]["options"]["solutionPrecision"] = "single"
        write_json(self.source_path, self.source)
        with self.assertRaisesRegex(ValueError, "double-precision"):
            self.validate()

    def test_restart_cost_includes_source_and_uses_freestream_not_restart_start(self):
        result = self.restarted()
        before = copy.deepcopy(result)
        audit = audit_restart(self.source, result)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["continuation_chain_seconds"], 9611.)
        self.assertAlmostEqual(audit["continuation_chain_allocated_rank_hours"], 9611.*8/3600)
        self.assertEqual(audit["final_relative_to_current_freestream"], 5e-11)
        self.assertEqual(result, before)

    def test_borderline_original_residual_remains_rejected(self):
        audit = audit_restart(self.source, self.restarted(final=.012369663))
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["accepted"])

    def test_refinement_stage_clock_includes_validation_and_loading(self):
        result = self.restarted()
        result['timing']['refinement_stage_seconds'] = 44.
        audit = audit_restart(self.source, result)
        self.assertEqual(audit['continuation_chain_seconds'], 9615.)

    def test_checkpoint_start_gate_is_distinct_from_final_threshold(self):
        result = self.restarted()
        result['convergence']['residual_start'] *= 1.005
        self.assertTrue(audit_restart(self.source, result)['accepted'])
        result['convergence']['residual_start'] *= 1.02
        with self.assertRaisesRegex(ValueError, 'Starting residual'):
            audit_restart(self.source, result)

    def test_denominator_change_or_nonfinite_evidence_rejects(self):
        result = self.restarted()
        result["convergence"]["residual_initial"] = 2e8
        result["convergence"]["residual_relative_to_freestream"] = .005 / 2e8
        with self.assertRaisesRegex(ValueError, "freestream residual denominator"):
            audit_restart(self.source, result)
        for field in ("residual_initial", "residual_start", "residual_final"):
            changed = self.restarted()
            changed["convergence"][field] = float("nan")
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_restart(self.source, changed)

    def test_wrong_source_provenance_restart_file_and_cost_reject(self):
        for change in ("source", "restart_file", "cost", "ranks"):
            result = self.restarted()
            if change == "source": result["provenance"]["restart"]["source_result_content_sha256"] = "a" * 64
            if change == "restart_file": result["solver"]["options"]["restartFile"] = str(self.mesh)
            if change == "cost": result["timing"]["total_seconds"] = -1.
            if change == "ranks": result["timing"]["mpi_ranks"] = 16
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_restart(self.source, result)


if __name__ == "__main__":
    unittest.main()
