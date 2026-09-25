import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np

from mdo_demo.dataset import load_dataset, physical_fields

_spec = importlib.util.spec_from_file_location(
    "fetch_assets", Path(__file__).resolve().parents[1] / "scripts" / "fetch_assets.py")
fetch_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_assets)


class DatasetSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        index = np.zeros((2, 12))
        index[:, 0] = [17, 3]
        index[:, 2:4] = [[2.0, .8], [3.5, .85]]
        index[:, 4] = 10
        index[:, 6:9] = [.4, .02, -.1]
        index[:, 9:12] = [.41, .021, -.11]
        np.save(self.root / "index.npy", index)
        np.save(self.root / "data.npy", np.ones((2, 3, 2, 4)))
        geom = np.empty((2, 3, 2, 4))
        geom[0] = 3
        geom[1] = 17
        np.save(self.root / "geom0.npy", geom)
        np.save(self.root / "origingeom.npy", np.zeros((2, 3, 3, 5)))
        (self.root / "manifest.json").write_text(json.dumps({
            "sample_ids": [200, 23], "geometry_shape_ids": [3, 17], "split": "test_fixture",
        }))

    def test_source_ids_are_not_confused_with_compact_rows(self):
        data = load_dataset(self.root)
        self.assertEqual(len(data), 2)
        sample = data.sample(0)
        self.assertEqual(sample["sample_id"], 200)
        self.assertEqual(sample["shape_id"], 17)
        np.testing.assert_array_equal(sample["geometry"], 17)
        np.testing.assert_allclose(sample["condition"], [2., .8])
        self.assertAlmostEqual(sample["reference_coefficients"]["CD"], .021)
        self.assertAlmostEqual(sample["solver_coefficients"]["CD"], .02)

    def test_reject_unknown_geometry(self):
        (self.root / "manifest.json").write_text(json.dumps({"geometry_shape_ids": [0, 1]}))
        with self.assertRaises(ValueError):
            load_dataset(self.root)

    def test_reject_cell_center_geometry_for_force_integration(self):
        np.save(self.root / "origingeom.npy", np.zeros((2, 3, 2, 4)))
        with self.assertRaisesRegex(ValueError, "vertex"):
            load_dataset(self.root)

    def test_friction_unscaling(self):
        fields = np.ones((3, 2, 4)) * np.array([2, 150, 300])[:, None, None]
        result = physical_fields(fields)
        np.testing.assert_array_equal(result[0], 2)
        np.testing.assert_array_equal(result[1:], 1)

    def test_bad_area_rejected(self):
        index = np.load(self.root / "index.npy")
        index[0, 4] = 0
        np.save(self.root / "index.npy", index)
        with self.assertRaisesRegex(ValueError, "area"):
            load_dataset(self.root)

    def test_sample_bounds(self):
        data = load_dataset(self.root)
        for bad in [-1, 2]:
            with self.assertRaises(IndexError):
                data.sample(bad)


class AssetDownloadTests(unittest.TestCase):
    def test_subset_spans_different_shapes_not_repeated_conditions(self):
        index = np.array([[0], [0], [4], [4], [9], [9], [13], [13]])
        self.assertEqual(fetch_assets.select_sample_ids(index, 3), [0, 2, 6])

    def test_range_ignoring_server_is_not_read(self):
        class IgnoringServer:
            status = 200
            headers = {"Content-Length": "1686896768"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                raise AssertionError("An ignored Range response must never be read")

        with patch.object(fetch_assets, "urlopen", return_value=IgnoringServer()):
            with self.assertRaisesRegex(RuntimeError, "Range"):
                fetch_assets.request_bytes("https://example.invalid/data.npy", start=0, size=1024)

    def test_wrong_range_is_rejected(self):
        class WrongRange:
            status = 206
            headers = {"Content-Range": "bytes 0-1023/2000"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with patch.object(fetch_assets, "urlopen", return_value=WrongRange()):
            with self.assertRaisesRegex(RuntimeError, "Range"):
                fetch_assets.request_bytes("https://example.invalid/data.npy", start=1024, size=128)


if __name__ == "__main__":
    unittest.main()
