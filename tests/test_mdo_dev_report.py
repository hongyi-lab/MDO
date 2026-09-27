import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mdo_demo.mdo_dev_report import _pack, write_mdo_dev_report


class MDODevReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.before = self.root / "before"
        self.after = self.root / "after"
        self.before.mkdir()
        self.after.mkdir()
        z, x = np.meshgrid(np.linspace(0, 2, 32), np.linspace(0, 1, 64), indexing="ij")
        geometry = np.stack((x, 0.03 * np.sin(x * np.pi), z))
        truth = np.stack((x - 0.5, x * 0.01, x * 0.02))
        for directory, delta in ((self.before, .03), (self.after, .01)):
            np.savez_compressed(directory / "case.npz", geometry=geometry, truth=truth,
                                prediction=truth + delta,
                                spanwise_lift_truth=np.ones(32) * .5,
                                spanwise_lift_prediction=np.ones(32) * (.5 + delta),
                                spanwise_width=np.ones(32) / 32)
        self.report = {
            "partition": "dev", "mode": "real_checkpoint_dataset_evaluation",
            "split_sha256": "unit-test-split", "dataset_manifest_sha256": "unit-test-data",
            "macro_metrics": {"samples": 2, "geometries": 2, "CL": {"mae": .01, "max_abs": .02},
                              "CD": {"mae": .001, "max_abs": .002}, "CD_drag_counts_mae": 10,
                              "Cp_mean_rmse": .02, "Cf_stream_mean_rmse": .0001,
                              "Cf_span_mean_rmse": .0002, "median_prediction_seconds": .1},
            "provenance": {"model": "unit-test fixture, not experiment evidence"},
            "cases": [{"sample_id": str(i), "shape_id": str(i),
                       "condition": {"alpha_deg": 3, "mach": .8}, "array_file": "case.npz",
                       "reference_coefficients": {"CL": .5, "CD": .02, "CM": .1},
                       "coefficients": {"CL": .51, "CD": .02 + .001 * (i + 1), "CM": .11}}
                      for i in range(2)]}

    def write(self, before=None, after=None):
        a, b = self.before / "evaluation.json", self.after / "evaluation.json"
        a.write_text(json.dumps(before or self.report), encoding="utf-8")
        b.write_text(json.dumps(after or self.report), encoding="utf-8")
        return write_mdo_dev_report(a, b, self.root / "report.html")

    @staticmethod
    def payload(path):
        text = path.read_text(encoding="utf-8")
        raw = text.split("const DATA=", 1)[1].split(";\nconst $=", 1)[0]
        return json.loads(raw), text

    def test_pack_error_bounded_and_little_endian(self):
        source = np.linspace(-1.57, .831, 151).reshape(1, -1)
        packed = _pack(source)
        decoded = np.frombuffer(base64.b64decode(packed["values"]), dtype="<u2")
        reconstructed = (packed["offset"] + packed["scale"] * decoded).reshape(source.shape)
        self.assertLessEqual(float(np.max(abs(reconstructed - source))), packed["scale"] / 2 + 1e-12)

    def test_all_cases_metrics_and_explicit_worst_case_selection_retained(self):
        path = self.write()
        data, text = self.payload(path)
        self.assertEqual(len(data["cases"]), 2)
        self.assertEqual(data["cases"][data["initial_index"]]["sample_id"], "1")
        self.assertEqual(data["metrics"]["adapted"], self.report["macro_metrics"])
        self.assertIsNotNone(data["cases"][0]["loads"])
        self.assertIn("DEV EXPLORATORY", text)
        self.assertIn("不是独立测试结论", text)
        self.assertNotIn("<script src=", text)
        self.assertNotIn("fetch(", text)

    def test_wrong_partition_or_mismatched_ids_rejected(self):
        changed = copy.deepcopy(self.report)
        changed["partition"] = "test"
        with self.assertRaisesRegex(ValueError, "DEV"):
            self.write(after=changed)
        changed = copy.deepcopy(self.report)
        changed["cases"][0]["sample_id"] = "9999"
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            self.write(after=changed)

    def test_mismatched_truth_or_path_escape_rejected(self):
        changed = copy.deepcopy(self.report)
        changed["cases"][0]["array_file"] = "../before/case.npz"
        with self.assertRaisesRegex(ValueError, "inside"):
            self.write(after=changed)
        with np.load(self.after / "case.npz", allow_pickle=False) as a:
            arrays = dict(a)
        arrays["truth"] = arrays["truth"] + .1
        np.savez_compressed(self.after / "case.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "truth"):
            self.write()

    def test_350_case_portable_budget_without_dropping_cases(self):
        report = copy.deepcopy(self.report)
        report["cases"] = [{**report["cases"][0], "sample_id": str(i), "shape_id": str(i)} for i in range(350)]
        report["macro_metrics"]["samples"] = 350
        report["macro_metrics"]["geometries"] = 350
        path = self.write(report, report)
        data, _ = self.payload(path)
        self.assertEqual(len(data["cases"]), 350)
        self.assertLess(path.stat().st_size, 20 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
