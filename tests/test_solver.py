import unittest
from unittest.mock import patch
import numpy as np

from mdo_demo.solver import HybridAerodynamicSolver, FIXED_PHYSICS, COEFFICIENT_DEFINITION, FIELD_SCALES


class Predictor:
    provenance = {"weights_sha256": "weights", "config_sha256": "config", "adapter_sha256": "adapter",
                  "flogen": {}, "cfdpost": {}}
    def predict(self, query):
        return {"coefficients": {"CL": .5, "CD": .02}, "fields": np.zeros((3, 128, 256))}


class SolverTests(unittest.TestCase):
    def query(self):
        x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 2, 128))
        return {"geometry": np.array([x, np.sin(x)*.1, y]),
                "original_geometry": np.zeros((3, 129, 257)), "condition": [2, .8],
                "ref_area": 1., "physics": dict(FIXED_PHYSICS)}

    @patch("mdo_demo.solver.assess_prediction", return_value={"accepted": False, "fallback_required": True})
    def test_missing_backend_never_uses_label_as_cfd(self, _):
        query = self.query()
        query["fields"] = np.ones((3, 128, 256))
        result = HybridAerodynamicSolver(Predictor(), {}, "data", "split").solve(query)
        self.assertEqual(result["status"], "cfd_required")
        self.assertIsNone(result["fields"])
        self.assertFalse(result["cfd_executed"])

    @patch("mdo_demo.solver.assess_prediction", return_value={"accepted": False, "fallback_required": True})
    def test_rejects_unmatched_fallback(self, _):
        backend = lambda query, fingerprint: {"status": "ok", "convergence": {"converged": True},
                                              "query_sha256": "different"}
        with self.assertRaises(RuntimeError):
            HybridAerodynamicSolver(Predictor(), {}, "data", "split", backend).solve(self.query())

    @patch("mdo_demo.solver.assess_prediction", return_value={"accepted": False, "fallback_required": True})
    def test_checked_fallback(self, _):
        def backend(query, fingerprint):
            return {"status": "ok", "convergence": {"converged": True}, "query_sha256": fingerprint,
                    "coefficient_definition": COEFFICIENT_DEFINITION,
                    "field_scales": FIELD_SCALES,
                    "coefficients": {"CL": .5, "CD": .02}, "fields": np.zeros((3,128,256))}
        result = HybridAerodynamicSolver(Predictor(), {}, "data", "split", backend).solve(self.query())
        self.assertTrue(result["cfd_executed"])
        self.assertEqual(result["status"], "cfd_ok")

    def test_unknown_reynolds_abstains_before_prediction(self):
        query = self.query()
        query["physics"]["reynolds"] = 1e6
        result = HybridAerodynamicSolver(Predictor(), {}, "data", "split").solve(query)
        self.assertEqual(result["status"], "cfd_required")

    def test_failed_model_does_not_get_accepted(self):
        predictor = Predictor()
        def failed(query):
            raise RuntimeError("Model failed")
        predictor.predict = failed
        result = HybridAerodynamicSolver(predictor, {}, "data", "split").solve(self.query())
        self.assertEqual(result["status"], "cfd_required")
        self.assertEqual(result["gate"]["reasons"], ["fm_prediction_failed"])


if __name__ == "__main__":
    unittest.main()
