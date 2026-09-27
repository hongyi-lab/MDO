"""Fail-closed report gates using synthetic record fixtures, not physics results."""
import copy
import importlib.util
from pathlib import Path
import unittest

from mdo_demo.aero_contract import make_request
from mdo_demo.aerostructural import constraint_margins, validate_protocol
from mdo_demo.io import read_json
from mdo_demo.matched_cfd import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_aerostructural_script", ROOT / "scripts/compare_aerostructural.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compare = MODULE.compare


def fixture():
    """Explicit records that meet gates; none are presented as measured data."""
    cfg = validate_protocol(read_json(ROOT / "configs/aerostructural_pilot_v1.json"))
    base = {"protocol": cfg, "protocol_sha256": cfg["protocol_sha256"],
            "base_case_sha256": "a" * 64, "base_geometry_sha256": "b" * 64}
    runs, checks = [], []
    for index, backend in enumerate(("adflow", "fm")):
        design = {"alpha_deg": .25 * index, "skin_m": .005 - .001 * index, "web_m": .004}
        case_hash = canonical_hash({"angle": design["alpha_deg"]})
        request = make_request(case_hash, base["base_geometry_sha256"], condition={
            "alpha_deg": design["alpha_deg"], "mach": .8, "reynolds": 20e6,
            "reynolds_length_m": 1., "temperature_k": 300., "dynamic_pressure_pa": 20000.},
            reference={"area_m2": 1.5, "chord_m": 1., "moment_center_m": [.25, 0., 0.]},
            coverage={"families": ["mainwing"], "closed_TE": False, "span_bounds_m": [-3., -.3]})
        structural = {"backend": "TACS", "status": "ok", "mass_kg": 90. - 10. * index,
            "ks_failure": .7, "tip_displacement_m": .05,
            "convergence": {"converged": True, "relative_residual": 1e-9, "tolerance": 1e-7,
                            "root_displacement_rotation_max": 0., "solver_reported_converged": None},
            "transfer_audit": {"passed": True},
            "thickness": {key: design[key] for key in ("skin_m", "web_m")},
            "material": copy.deepcopy(cfg["material"]), "mesh_sha256": "c" * 64,
            "common_geometry_source": base["base_geometry_sha256"],
            "provenance": {"structural_code_sha256": "d" * 64}}
        coeff = {"CL": .6, "CD": .03}
        margins = constraint_margins(coeff, structural, cfg)
        checked = {"status": "ok", "backend": "adflow", "design": design,
            "aerodynamic_case_sha256": case_hash, "structural": structural, "coefficients": coeff,
            "constraint_margins": margins, "minimum_margin": min(margins.values()), "feasible": True,
            "aerodynamic_evidence": {"backend": "adflow", "reference_eligible": True,
                "request": request, "integration_audit": {"passed": True},
                "solver_convergence": {"converged": True, "solve_failed": False, "fatal_failed": False,
                    "required_l2_convergence": 1e-10, "residual_relative_to_freestream": 1e-11},
                "provenance": {"solver_result_content_sha256": "e" * 64}}}
        candidate = copy.deepcopy(checked)
        candidate["backend"] = backend
        cost = 100. if backend == "adflow" else 10.
        run = {**copy.deepcopy(base), "action": "optimize", "backend": backend,
               "status": "budget_limited", "total_wall_seconds": cost,
               "optimization_wall_seconds": cost - 1., "optimizer_converged": False,
               "aerodynamic_call_count": 2, "best_feasible": candidate, "best_candidate": candidate}
        check = {**copy.deepcopy(base), "action": "verify", "status": "completed",
                 "common_verification_complete": True, "checked": checked,
                 "verification_wall_seconds": 5., "total_wall_seconds": 5.2,
                 "original_optimization_wall_seconds": cost,
                 "candidate_result_content_sha256": canonical_hash(run),
                 "candidate_sha256": canonical_hash(candidate)}
        runs.append(run)
        checks.append(check)
    return runs + checks


def rebind(items, branch):
    run, check = items[branch], items[branch + 2]
    check["candidate_result_content_sha256"] = canonical_hash(run)
    check["candidate_sha256"] = canonical_hash(run.get("best_feasible") or run["best_candidate"])


class ComparisonTests(unittest.TestCase):
    def test_success_reports_quality_and_all_measured_cost_without_speedup_claim(self):
        result = compare(*fixture())
        self.assertEqual(result["rows"][0]["total_wall_seconds"], 105.2)
        self.assertEqual(result["rows"][1]["total_wall_seconds"], 15.2)
        self.assertEqual(result["rows"][1]["verification_wall_seconds"], 5.2)
        self.assertAlmostEqual(result["relative_verified_mass_difference"], 80 / 90 - 1)
        self.assertIsNone(result["equal_quality_speedup"])

    def test_different_base_case_or_geometry_is_rejected(self):
        for key in ("base_case_sha256", "base_geometry_sha256"):
            records = fixture()
            records[1][key] = "f" * 64
            rebind(records, 1)
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(*records)

    def test_same_design_from_another_run_cannot_borrow_verification(self):
        records = fixture()
        records[1]["total_wall_seconds"] += 1.
        with self.assertRaisesRegex(ValueError, "exact source run"):
            compare(*records)

    def test_wrong_source_candidate_cannot_borrow_verification(self):
        records = fixture()
        records[3]["candidate_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "exact source run"):
            compare(*records)

    def test_mislabeled_backend_or_nonoptimization_run_is_rejected(self):
        for key, value in (("backend", "adflow"), ("action", "single"), ("status", "blocked")):
            records = fixture()
            records[1][key] = value
            rebind(records, 1)
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(*records)

    def test_success_labels_cannot_override_bad_cfd_residual_or_missing_audit(self):
        for key, value in (("residual_relative_to_freestream", 1e-4),
                           ("required_l2_convergence", 1e-6), ("solve_failed", True)):
            records = fixture()
            records[3]["checked"]["aerodynamic_evidence"]["solver_convergence"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(*records)
        records = fixture()
        records[3]["checked"]["aerodynamic_evidence"]["integration_audit"]["passed"] = False
        with self.assertRaises(ValueError):
            compare(*records)

    def test_feigned_tacs_success_or_unconverged_structure_is_rejected(self):
        for where, key, value in ((None, "backend", "beam"),
                                  ("convergence", "relative_residual", 1e-3),
                                  ("transfer_audit", "passed", False)):
            records = fixture()
            target = records[3]["checked"]["structural"]
            if where:
                target = target[where]
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(*records)

    def test_nan_negative_and_partial_costs_are_rejected(self):
        for value in (float("nan"), float("inf"), -1., 0., True):
            records = fixture()
            records[3]["total_wall_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare(*records)
        records = fixture()
        records[1]["optimization_wall_seconds"] = 100.
        rebind(records, 1)
        with self.assertRaises(ValueError):
            compare(*records)

    def test_verified_constraint_failure_is_reported_honestly(self):
        records = fixture()
        checked = records[3]["checked"]
        checked["coefficients"]["CL"] = .4
        cfg = records[1]["protocol"]
        checked["constraint_margins"] = constraint_margins(checked["coefficients"], checked["structural"], cfg)
        checked["minimum_margin"] = min(checked["constraint_margins"].values())
        checked["feasible"] = False
        self.assertFalse(compare(*records)["rows"][1]["verified_feasible"])

    def test_feasibility_cannot_be_declared_against_actual_margins(self):
        records = fixture()
        records[3]["checked"]["feasible"] = False
        with self.assertRaisesRegex(ValueError, "feasibility"):
            compare(*records)

    def test_candidate_and_check_need_same_structural_code_and_mesh(self):
        for where, key in ((None, "mesh_sha256"), ("provenance", "structural_code_sha256")):
            records = fixture()
            target = records[1]["best_feasible"]["structural"]
            if where:
                target = target[where]
            target[key] = "f" * 64
            rebind(records, 1)
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(*records)

    def test_same_shape_but_different_flow_condition_is_rejected(self):
        records = fixture()
        evidence = records[3]["checked"]["aerodynamic_evidence"]
        identity = evidence["request"]["identity"]
        condition = dict(identity["condition"], mach=.7)
        evidence["request"] = make_request(identity["case_id"], identity["geometry_sha256"],
            condition=condition, reference=identity["reference"], coverage=identity["coverage"])
        with self.assertRaisesRegex(ValueError, "different flow"):
            compare(*records)


if __name__ == "__main__":
    unittest.main()
