import unittest

import numpy as np

from mdo_demo.metrics import break_even_queries, coefficient_errors, field_errors


class MetricTests(unittest.TestCase):
    def test_relative_norm_and_counts(self):
        self.assertAlmostEqual(field_errors([2., 4.], [1., 2.])["relative_l2"], 1.)
        self.assertAlmostEqual(coefficient_errors({"CD": .0101}, {"CD": .01})["CD"]["error_drag_counts"], 1.)

    def test_missing_reference_not_fabricated(self):
        self.assertIsNone(field_errors([1], [0])["relative_l2"])
        self.assertEqual(coefficient_errors({"CD": .1}, {}), {})

    def test_invalid_arrays(self):
        for p, t in [([1, 2], [1]), ([np.nan], [1]), ([], [])]:
            with self.assertRaises(ValueError):
                field_errors(p, t)

    def test_break_even_is_strictly_faster(self):
        self.assertEqual(break_even_queries(100, 11, 1), 11)
        self.assertIsNone(break_even_queries(100, 1, 2))
        with self.assertRaises(ValueError):
            break_even_queries(-1, 11, 1)


if __name__ == "__main__":
    unittest.main()
