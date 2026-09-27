import unittest
import numpy as np
from mdo_demo.mdo_evaluation import DevOnlyDataset, macro_metrics


class GuardedDataset:
    sample_ids = [10,20,21,30,40]
    manifest = {}
    def sample(self, row):
        if self.sample_ids[row] not in (20,21):
            raise AssertionError("Read a non-development label")
        return {"sample_id":self.sample_ids[row]}


class DevEvaluationTests(unittest.TestCase):
    def test_iteration_never_reads_training_or_sealed_labels(self):
        subset = DevOnlyDataset(GuardedDataset(), {"dev_rows":[1,2],"dev_sample_ids":[20,21],
                              "calibration_sample_ids":[30],"test_sample_ids":[40]})
        self.assertEqual([subset.sample(i)["sample_id"] for i in range(len(subset))],[20,21])

    def test_wrong_row_or_duplicate_cannot_enter_dev(self):
        for rows in ([1,3],[1,1]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                DevOnlyDataset(GuardedDataset(), {"dev_rows":rows,"dev_sample_ids":[20,21],
                              "calibration_sample_ids":[30],"test_sample_ids":[40]})

    def test_macro_weighting_does_not_overweight_geometry_with_more_conditions(self):
        def case(g,error):
            return {"shape_id":g,"coefficients":{"CL":error,"CD":error},
                    "reference_coefficients":{"CL":0.,"CD":0.},
                    "field_errors":{n:{"rmse":error} for n in ("Cp","Cf_stream","Cf_span")},
                    "end_to_end_prediction_s":.1}
        result=macro_metrics([case(1,1.),case(1,3.),case(2,10.)])
        self.assertEqual(result["CL"]["mae"],6.)
        self.assertAlmostEqual(result["CL"]["rmse"],np.sqrt(52.5))
        self.assertEqual(result["CL"]["p95_geometry_max_abs"],10.)


if __name__ == "__main__":
    unittest.main()
