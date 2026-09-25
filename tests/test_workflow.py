import tempfile
import unittest
from pathlib import Path

from mdo_demo.io import read_json, write_json
from mdo_demo.screening import screen_candidates
from mdo_demo.smoke import run_smoke


class WorkflowTests(unittest.TestCase):
    def test_end_to_end_artifacts_and_honest_label(self):
        with tempfile.TemporaryDirectory() as directory:
            html = run_smoke(directory)
            result = read_json(Path(directory)/"evaluation.json")
            self.assertEqual(result["mode"], "synthetic_interface_smoke")
            self.assertIsNone(result["speedup"])
            self.assertEqual(len(result["cases"]), 3)
            self.assertIn("SYNTHETIC INTERFACE TEST", html.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                run_smoke(directory)
            self.assertEqual(read_json(Path(directory)/"evaluation.json"), result)
            with self.assertRaises(ValueError):
                screen_candidates(result, .78, .4)

    def test_optimization_exploitation_is_reported(self):
        result = {"mode":"real_checkpoint_dataset_evaluation", "cases":[
            {"sample_id":"bad", "condition":{"mach":.78},
             "coefficients":{"CL":.6,"CD":.01}, "reference_coefficients":{"CL":.3,"CD":.03}},
            {"sample_id":"good", "condition":{"mach":.78},
             "coefficients":{"CL":.6,"CD":.02}, "reference_coefficients":{"CL":.6,"CD":.02}}]}
        screened = screen_candidates(result, .78, .5)
        self.assertEqual(screened["selected_sample_id"], "bad")
        self.assertFalse(screened["reference_feasible"])
        self.assertIsNone(screened["CD_regret"])
        self.assertAlmostEqual(screened["CL_violation"], .2)
        with self.assertRaises(ValueError):
            screen_candidates(result, .7, .5)

    def test_json_rejects_nan_without_overwriting_previous_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.json"
            write_json(path, {"valid":True})
            with self.assertRaises(ValueError):
                write_json(path, {"bad":float("nan")})
            self.assertEqual(read_json(path), {"valid":True})


if __name__ == "__main__":
    unittest.main()
