"""Orchestration tests with independently defined analytical test providers.

These fixtures exercise accounting and optimization; they are not CFD/FEA
results and are never used by the experiment CLI.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mdo_demo.aerostructural import (
    EvaluationBudgetExceeded, PhysicsFailure, PilotEvaluator, constraint_margins,
    validate_protocol,
)
from mdo_demo.io import read_json


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "aerostructural_pilot_v1.json"


def protocol():
    return read_json(PROTOCOL_PATH)


def scaled_design(cfg, alpha=0., skin=.005, web=.004):
    return np.array([alpha / cfg["design"]["alpha_deg"]["scale"],
                     skin / cfg["design"]["skin_m"]["scale"],
                     web / cfg["design"]["web_m"]["scale"]])


class AnalyticalProviders:
    def __init__(self):
        self.aero_calls = []
        self.structure_calls = []

    def aero(self, alpha, directory):
        if not -2 <= alpha <= 2:
            raise AssertionError("Out-of-domain aero call")
        self.aero_calls.append(alpha)
        return {"status": "ok", "case_sha256": alpha.hex(),
                "contract_status": "test_fixture_only",
                "coefficients": {"CL": .6 + .1 * alpha, "CD": .03 + .001 * alpha**2}}

    def structure(self, aero, design, directory):
        self.structure_calls.append(copy.deepcopy(design))
        # Independent sizing fixture: mass grows with thickness, while bending
        # stress and displacement grow as thickness falls and loading rises.
        load = aero["coefficients"]["CL"] / .6
        return {"status": "ok", "mass_kg": 10000 * (design["skin_m"] + design["web_m"]),
                "ks_failure": load * .002 / design["skin_m"],
                "tip_displacement_m": load * .0002 / design["web_m"]}


class AeroStructuralTests(unittest.TestCase):
    def test_protocol_hash_is_stable_and_physics_changes_change_it(self):
        cfg = validate_protocol(protocol())
        self.assertEqual(cfg["protocol_sha256"], validate_protocol(cfg)["protocol_sha256"])
        changed = copy.deepcopy(cfg)
        changed["limits"]["drag_coefficient_max"] *= 2
        self.assertNotEqual(cfg["protocol_sha256"], validate_protocol(changed)["protocol_sha256"])
        changed = protocol()
        changed["design"]["skin_m"]["lower"] = 0.
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_constraints_use_declared_limits_and_reject_nonfinite_mass(self):
        aero = {"CL": .6, "CD": .03}
        structure = {"mass_kg": 30., "ks_failure": .75, "tip_displacement_m": .075}
        margins = constraint_margins(aero, structure, protocol())
        np.testing.assert_allclose(list(margins.values()), [.2, .5, .25, .5])
        for bad in (float("nan"), float("inf"), 0., -1.):
            structure["mass_kg"] = bad
            with self.assertRaises(ValueError):
                constraint_margins(aero, structure, protocol())

    def test_exact_repeat_and_same_angle_thickness_change_cache_equally(self):
        cfg = protocol()
        provider = AnalyticalProviders()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(cfg, "test", Path(temporary) / "run", provider.aero, provider.structure)
            first = evaluator.evaluate(scaled_design(cfg))
            again = evaluator.evaluate(scaled_design(cfg))
            self.assertIs(first, again)
            thinner = evaluator.evaluate(scaled_design(cfg, skin=.004))
            self.assertTrue(thinner["aerodynamic_cache_hit"])
            self.assertLess(thinner["mass_kg"], first["mass_kg"])
            evaluator.evaluate(scaled_design(cfg, alpha=.1))
            self.assertEqual(len(provider.aero_calls), 2)
            self.assertEqual(len(provider.structure_calls), 3)
            self.assertEqual(len(evaluator.history), 3)

    def test_outside_bounds_never_calls_providers_or_reports_physics(self):
        cfg = protocol()
        provider = AnalyticalProviders()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(cfg, "test", Path(temporary) / "run", provider.aero, provider.structure)
            outside = scaled_design(cfg, skin=.000999999999)
            result = evaluator.evaluate(outside)
            self.assertEqual(result["status"], "domain_rejected")
            self.assertNotIn("mass_kg", result)
            self.assertNotIn("coefficients", result)
            self.assertFalse(result["physical_evaluation"])
            self.assertTrue(all(x < 0 for x in result["optimizer_domain_constraints"]))
            self.assertIs(result, evaluator.evaluate(outside))
            self.assertEqual(provider.aero_calls, [])
            self.assertEqual(provider.structure_calls, [])
            self.assertEqual(len(evaluator.history), 1)

    def test_aero_budget_uses_distinct_physical_angles_but_allows_sizing(self):
        cfg = protocol()
        cfg["optimizer"]["max_aero_evaluations"] = 1
        provider = AnalyticalProviders()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(cfg, "test", Path(temporary) / "run", provider.aero, provider.structure)
            evaluator.evaluate(scaled_design(cfg))
            evaluator.evaluate(scaled_design(cfg, skin=.004))
            with self.assertRaises(EvaluationBudgetExceeded):
                evaluator.evaluate(scaled_design(cfg, alpha=.1))
            self.assertEqual(len(provider.aero_calls), 1)
            self.assertEqual(evaluator.history[-1]["status"], "budget_exhausted")

    def test_design_budget_never_runs_an_extra_physical_point(self):
        cfg = protocol()
        cfg["optimizer"]["max_evaluations"] = 1
        provider = AnalyticalProviders()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(cfg, "test", Path(temporary) / "run", provider.aero, provider.structure)
            evaluator.evaluate(scaled_design(cfg))
            with self.assertRaises(EvaluationBudgetExceeded):
                evaluator.evaluate(scaled_design(cfg, skin=.004))
            self.assertEqual(len(provider.structure_calls), 1)
            self.assertEqual(len(evaluator.history), 1)

    def test_failed_aero_call_is_counted_and_never_turns_into_penalty(self):
        def failed_aero(alpha, directory):
            return {"status": "not_converged"}
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(protocol(), "test", Path(temporary) / "run", failed_aero,
                                       lambda *args: self.fail("Structure called after failed CFD"))
            result = evaluator.run(optimize=False)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["aerodynamic_call_count"], 1)
            self.assertEqual(result["aerodynamic_successful_call_count"], 0)
            self.assertIsNone(result["best_feasible"])
            self.assertNotIn("mass_kg", evaluator.history[0])
            self.assertFalse(result["mdo_speedup_verified"])
            self.assertEqual(read_json(evaluator.output / "result.json"), result)

    def test_invalid_structural_result_is_saved_as_failure(self):
        provider = AnalyticalProviders()
        def invalid_structure(*args):
            return {"status": "ok", "mass_kg": float("nan"),
                    "ks_failure": .5, "tip_displacement_m": .1}
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(protocol(), "test", Path(temporary) / "run", provider.aero, invalid_structure)
            result = evaluator.run(optimize=False)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["structural_call_count"], 1)
            self.assertEqual(result["successful_physical_evaluation_count"], 0)
            lines = (evaluator.output / "history.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["status"], "failed")

    def test_infeasible_point_is_not_a_failed_solve_or_feasible_design(self):
        cfg = protocol()
        cfg["limits"]["lift_coefficient_min"] = 1.
        provider = AnalyticalProviders()
        with tempfile.TemporaryDirectory() as temporary:
            evaluator = PilotEvaluator(cfg, "test", Path(temporary) / "run", provider.aero, provider.structure)
            result = evaluator.run(optimize=False)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["feasible_candidate_found"])
            self.assertIsNone(result["best_feasible"])
            self.assertEqual(result["best_candidate"]["status"], "ok")
            self.assertFalse(result["common_verification_complete"])

    def test_same_optimizer_with_independent_identical_backends_is_symmetric(self):
        try:
            import scipy.optimize  # noqa: F401
        except ImportError:
            self.skipTest("SciPy not installed in validation environment")
        cfg = protocol()
        cfg["optimizer"].update(max_evaluations=60, max_aero_evaluations=60)
        results = []
        histories = []
        with tempfile.TemporaryDirectory() as temporary:
            for backend in ("cfd_fixture", "fm_fixture"):
                provider = AnalyticalProviders()
                evaluator = PilotEvaluator(cfg, backend, Path(temporary) / backend,
                                           provider.aero, provider.structure)
                result = evaluator.run()
                self.assertNotEqual(result["status"], "blocked", result)
                self.assertLessEqual(result["evaluation_count"], 60)
                self.assertIsNotNone(result["best_feasible"])
                self.assertLess(result["best_feasible"]["mass_kg"], 90.)
                self.assertFalse(result["common_verification_complete"])
                results.append(result)
                histories.append([(r["status"], r["design"]) for r in evaluator.history])
            self.assertEqual(histories[0], histories[1])
            self.assertEqual(results[0]["best_feasible"]["mass_kg"], results[1]["best_feasible"]["mass_kg"])
            self.assertEqual(results[0]["aerodynamic_call_count"], results[1]["aerodynamic_call_count"])


if __name__ == "__main__":
    unittest.main()
