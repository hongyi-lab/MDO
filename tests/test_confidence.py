"""Tests target leakage, false confidence and fail-closed behavior."""

import copy
import unittest

import numpy as np

from mdo_demo.confidence import (assess_prediction, describe_geometry, evaluate_heldout,
                                 fit_calibration, split_geometry_groups)


class ConfidenceTests(unittest.TestCase):
    def fixture(self, calibration_count=9, score=0.8, alpha=0.1):
        split = split_geometry_groups(range(calibration_count + 4), seed=5,
                                      counts={"train": 2, "calibration": calibration_count, "test": 2})
        identity = {"model": "weights-sha", "dataset": "manifest-sha", "split": split["fingerprint"]}

        def case(shape, sample, error=0):
            return {"shape_id": shape, "sample_id": sample,
                    "prediction_identity": dict(identity),
                    "condition": {"alpha_deg": 2., "mach": .8},
                    "geometry_descriptors": {"span": 6.},
                    "coefficients": {"CL": .5, "CD": .02, "CM": 999.},
                    "reference_coefficients": {"CL": .5 + error * .01, "CD": .02, "CM": -999.}}

        support = [case(shape, f"train-{shape}") for shape in split["train"]]
        support[0]["condition"] = {"alpha_deg": 0., "mach": .7}
        support[1]["condition"] = {"alpha_deg": 4., "mach": .9}
        support[0]["geometry_descriptors"] = {"span": 5.}
        support[1]["geometry_descriptors"] = {"span": 7.}
        cal = [case(shape, f"cal-{shape}", score) for shape in split["calibration"]]
        test = [case(shape, f"test-{shape}", .3) for shape in split["test"]]
        artifact = fit_calibration(cal, identity, {"CL": .01, "CD": .0005}, alpha,
                                   split=split, support_cases=support)
        return split, identity, cal, support, test, artifact

    def test_disjoint_geometry_split_is_order_and_condition_count_invariant(self):
        first = split_geometry_groups([1, 1, 2, 3, 4, 5, 6], seed=42)
        second = split_geometry_groups(iter([6, 5, 4, 3, 2, 1]), seed=42)
        self.assertEqual(first, second)
        all_ids = sum((first[part] for part in ("train", "calibration", "test")), [])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {str(i) for i in range(1, 7)})

    def test_calibration_cannot_take_test_geometry(self):
        split, identity, cal, support, test, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "outside its partition"):
            fit_calibration(cal + test[:1], identity, split=split, support_cases=support)

    def test_support_cannot_take_calibration_or_test_geometry(self):
        split, identity, cal, support, test, _ = self.fixture()
        for leaked in (cal[:1], test[:1]):
            with self.assertRaisesRegex(ValueError, "training geometries only"):
                fit_calibration(cal, identity, split=split, support_cases=support + leaked)

    def test_missing_planned_calibration_geometry_rejected(self):
        split, identity, cal, support, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "every planned"):
            fit_calibration(cal[:-1], identity, split=split, support_cases=support)

    def test_group_max_not_condition_count_controls_finite_sample_rank(self):
        split, identity, cal, support, test, _ = self.fixture(calibration_count=8)
        many_conditions = []
        for case in cal:
            for condition in range(30):
                duplicate = copy.deepcopy(case)
                duplicate["sample_id"] += f"-{condition}"
                many_conditions.append(duplicate)
        artifact = fit_calibration(many_conditions, identity, split=split, support_cases=support)
        self.assertEqual(artifact["calibration_case_count"], 240)
        self.assertEqual(artifact["calibration_group_count"], 8)
        self.assertEqual(artifact["finite_sample_rank"], 9)
        self.assertIsNone(artifact["quantile_normalized_error"])
        self.assertEqual(assess_prediction(test[0], artifact)["decision"], "fallback")

    def test_quantile_uses_worst_condition_and_both_coefficients(self):
        split, identity, cal, support, _, _ = self.fixture(score=.1)
        extra = copy.deepcopy(cal[0])
        extra["sample_id"] += "-extra"
        extra["reference_coefficients"]["CD"] += .00045
        artifact = fit_calibration(cal + [extra], identity, split=split, support_cases=support)
        self.assertAlmostEqual(artifact["quantile_normalized_error"], .9)
        self.assertEqual(artifact["finite_sample_rank"], 9)
        self.assertAlmostEqual(artifact["interval_half_widths"]["CL"], .009)
        self.assertAlmostEqual(artifact["interval_half_widths"]["CD"], .00045)

    def test_truth_and_field_error_never_influence_acceptance(self):
        _, _, _, _, test, artifact = self.fixture()
        original = assess_prediction(test[0], artifact)
        self.assertTrue(original["accepted"])
        test[0]["reference_coefficients"] = {"CL": 1000., "CD": 1000.}
        test[0]["field_errors"] = {"Cp": {"rmse": 1000.}}
        self.assertEqual(assess_prediction(test[0], artifact), original)
        del test[0]["reference_coefficients"]
        self.assertEqual(assess_prediction(test[0], artifact), original)

    def test_excess_calibrated_error_abstains_even_when_test_truth_is_perfect(self):
        _, _, _, _, test, artifact = self.fixture(score=2.)
        test[0]["reference_coefficients"] = dict(test[0]["coefficients"])
        decision = assess_prediction(test[0], artifact)
        self.assertFalse(decision["accepted"])
        self.assertIn("calibrated_interval_exceeds_tolerance", decision["reasons"])

    def test_identity_mismatch_and_tampered_artifact_fail_closed(self):
        _, _, _, _, test, artifact = self.fixture()
        for key in ("model", "dataset", "split"):
            modified = copy.deepcopy(test[0])
            modified["prediction_identity"][key] = "changed"
            self.assertIn("prediction_identity_mismatch", assess_prediction(modified, artifact)["reasons"])
        damaged = copy.deepcopy(artifact)
        damaged["interval_half_widths"]["CD"] = 0
        self.assertIn("invalid_prediction_or_calibration", assess_prediction(test[0], damaged)["reasons"])

    def test_invalid_predictions_and_unknown_scope_fail_closed(self):
        _, _, _, _, test, artifact = self.fixture()
        mutations = [lambda c: c["coefficients"].update(CD=float("nan")),
                     lambda c: c["condition"].update(mach=.95),
                     lambda c: c["condition"].update(reynolds=1e9),
                     lambda c: c["geometry_descriptors"].update(span=8.),
                     lambda c: c["geometry_descriptors"].update(new_descriptor=1.),
                     lambda c: c.pop("prediction_identity")]
        for mutate in mutations:
            case = copy.deepcopy(test[0])
            mutate(case)
            self.assertFalse(assess_prediction(case, artifact)["accepted"])

    def test_no_support_or_calibration_shape_reuse_abstains(self):
        split, identity, cal, _, test, artifact = self.fixture()
        no_support = fit_calibration(cal, identity, split=split)
        self.assertEqual(no_support["status"], "support_unavailable")
        self.assertFalse(assess_prediction(test[0], no_support)["accepted"])
        self.assertIn("calibration_geometry_reused_for_inference",
                      assess_prediction(cal[0], artifact)["reasons"])

    def test_heldout_audit_exposes_unsafe_acceptances_without_refitting(self):
        _, _, _, _, test, artifact = self.fixture()
        before = copy.deepcopy(artifact)
        test[1]["reference_coefficients"]["CL"] = .6
        audit = evaluate_heldout(test, artifact)
        self.assertEqual(audit["accepted_case_count"], 2)
        self.assertEqual(audit["observed_unsafe_accepted_cases"], 1)
        self.assertEqual(audit["observed_simultaneous_geometry_group_coverage"], .5)
        self.assertFalse(audit["live_cfd_fallback_executed"])
        self.assertFalse(audit["truth_used_for_gate_decision"])
        self.assertTrue(audit["test_partition_complete"])
        self.assertEqual(before, artifact)

    def test_heldout_evaluation_rejects_training_and_calibration_shapes(self):
        _, _, cal, support, _, artifact = self.fixture()
        for leaked in (cal, support):
            with self.assertRaisesRegex(ValueError, "outside the test partition"):
                evaluate_heldout(leaked, artifact)

    def test_zero_accepted_is_not_reported_as_zero_failure_rate(self):
        _, _, _, _, test, artifact = self.fixture(score=2.)
        audit = evaluate_heldout(test, artifact)
        self.assertEqual(audit["accepted_case_count"], 0)
        self.assertIsNone(audit["observed_unsafe_fraction_among_accepted"])

    def test_nonfinite_and_degenerate_geometry_rejected(self):
        for geometry in (np.zeros((3, 4, 4)), np.full((3, 4, 4), np.nan)):
            with self.assertRaises(ValueError):
                describe_geometry(geometry)
        xyz = np.array([[0., 1., 0., 1.], [0., 0., .1, .1], [0., 0., 2., 2.]])
        descriptors = describe_geometry(xyz)
        self.assertEqual(descriptors["x_extent"], 1.)
        self.assertEqual(descriptors["z_extent"], 2.)
        self.assertTrue(all(np.isfinite(value) for value in descriptors.values()))


if __name__ == "__main__":
    unittest.main()
