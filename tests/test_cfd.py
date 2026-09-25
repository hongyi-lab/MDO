"""Contract tests only: no test pretends to run a real CFD solver."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mdo_demo.cfd import convergence_report, history_major_iterations, memory_report, solver_options, validate_request


class CFDContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory / "wing.cgns").write_bytes(b"unit test mesh identity only")
        self.request = {"schema_version": 1, "problem_id": "test", "geometry_id": "shape-0", "mesh_path": "wing.cgns",
                        "condition": {"mach": 0.85, "alpha_deg": 2.0, "reynolds": 2e7,
                                      "reynolds_length_m": 1, "temperature_k": 300},
                        "reference": {"area_m2": 1.5, "chord_m": 1, "moment_center_m": [0.25, 0, 0]}}

    def validate(self, request=None):
        return validate_request(self.request if request is None else request, self.directory)

    def test_identity_changes_with_physics_mesh_reference(self):
        first = self.validate()
        for field, value in (("mach", 0.84), ("alpha_deg", 2.1), ("reynolds", 1e7)):
            changed = copy.deepcopy(self.request)
            changed["condition"][field] = value
            self.assertNotEqual(self.validate(changed)["case_sha256"], first["case_sha256"])
        changed = copy.deepcopy(self.request)
        changed["reference"]["moment_center_m"] = [0, 0, 0]
        self.assertNotEqual(self.validate(changed)["case_sha256"], first["case_sha256"])
        (self.directory / "wing.cgns").write_bytes(b"different mesh")
        self.assertNotEqual(self.validate()["case_sha256"], first["case_sha256"])

    def test_no_implicit_target_lift_or_missing_re(self):
        self.request["condition"]["target_cl"] = 0.5
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.validate()
        del self.request["condition"]["alpha_deg"]
        with self.assertRaisesRegex(ValueError, "alpha_initial_deg"):
            self.validate()
        self.request["condition"]["alpha_initial_deg"] = 2
        self.assertEqual(self.validate()["identity"]["condition"]["target_cl"], .5)
        del self.request["condition"]["reynolds"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate()

    def test_invalid_re_is_rejected(self):
        for value in [float("nan"), float("inf"), -1, True, "20000000"]:
            with self.subTest(value=value):
                self.request["condition"]["reynolds"] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_residual_flags_and_trim_all_required(self):
        args = dict(tolerance=1e-10, solve_failed=False, fatal_failed=False, coefficients={"CL": .5, "CD": .02})
        self.assertTrue(convergence_report((1, 1, 1e-11), **args)["converged"])
        for norms in [(1, 1, 1e-6), (0, 1, 0), (1, 1, float("nan"))]:
            self.assertFalse(convergence_report(norms, **args)["converged"])
        self.assertFalse(convergence_report((1, 1, 1e-11), **{**args, "solve_failed": True})["converged"])
        self.assertFalse(convergence_report((1, 1, 1e-11), **{**args, "coefficients": {"CD": float("nan")}})["converged"])
        self.assertFalse(convergence_report((1, 1, 1e-11), target_cl=.5, trim_converged=False, **args)["converged"])
        self.assertFalse(convergence_report((1, 1, 1e-11), target_cl=.6, trim_converged=True, **args)["converged"])
        self.assertTrue(convergence_report((1, 1, 1e-11), target_cl=.5, trim_converged=True, **args)["converged"])
        failed = convergence_report((1, 1, 1e-11), target_cl=.5, trim_converged=True,
                                    **{**args, "coefficients": {"CL": float("nan"), "CD": .02}})
        self.assertFalse(failed["converged"])
        json.dumps(failed, allow_nan=False)
        self.assertFalse(convergence_report((1, 1, 1e-11), solved_alpha_deg=float("nan"), **args)["converged"])
        self.assertFalse(convergence_report((1, 1, 1e-11), **{**args, "coefficients": {"CL": .5, "CD": .02, "CM": float("inf")}})["converged"])

    def test_memory_units_and_partial_measurements(self):
        measured = memory_report([1024, 2048])
        self.assertEqual(measured["per_rank_peak_rss_mib"], [1, 2])
        self.assertEqual(measured["max_rank_peak_rss_mib"], 2)
        self.assertEqual(measured["sum_rank_peak_rss_mib"], 3)
        self.assertIn("NOT simultaneous", measured["scope"])
        partial = memory_report([1024, None])
        self.assertFalse(partial["all_ranks_measured"])
        self.assertIsNone(partial["sum_rank_peak_rss_mib"])
        self.assertIsNone(memory_report([None])["max_rank_peak_rss_mib"])

    def test_history_distinguishes_major_and_minor_counts(self):
        self.assertEqual(history_major_iterations({"total minor iters": [0, 3, 9, 11]}), 3)
        self.assertEqual(history_major_iterations({"Total Minor Iters": [0]}), 0)
        self.assertIsNone(history_major_iterations({"unrecognized": [0, 1]}))

    def test_tutorial_preset_is_explicit(self):
        self.request["numerics"] = {"preset": "tutorial"}
        opts = solver_options(self.validate(), self.directory)
        self.assertEqual(opts["L2Convergence"], 1e-6)
        self.assertEqual(opts["MGCycle"], "sg")
        self.request["numerics"]["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.validate()

    def test_validate_cli_does_not_claim_cfd(self):
        input_file = self.directory / "request.json"
        input_file.write_text(json.dumps(self.request))
        output_file = self.directory / "result.json"
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_cfd.py"
        completed = subprocess.run([sys.executable, str(script), "--input", str(input_file), "--output", str(output_file),
                                    "--validate-only"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(output_file.read_text())
        self.assertEqual(result["status"], "validated_only")
        self.assertIs(result["cfd_executed"], False)


if __name__ == "__main__":
    unittest.main()
