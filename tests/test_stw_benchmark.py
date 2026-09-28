import tempfile
import unittest
from pathlib import Path
import numpy as np
from mdo_demo.stw_benchmark import chord_stencil, read_point_zones, polar_scores


class STWTests(unittest.TestCase):
    def test_point_order_and_bad_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'surface.dat'
            p.write_text('VARIABLES="x","y","z"\nZone I=2 J=2\nDATAPACKING=POINT\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n')
            a = read_point_zones(p)[0]
            np.testing.assert_array_equal(a[1, 0], [0, 1, 0])
            p.write_text(p.read_text().replace('1 1 0', 'nan 1 0'))
            with self.assertRaises(ValueError): read_point_zones(p)

    def test_stencil_endpoints_and_order(self):
        a = chord_stencil()
        self.assertEqual(a[0], 0); self.assertEqual(a[-1], 1)
        self.assertTrue(np.all(np.diff(a)>0))
        self.assertLess(a[1]-a[0], a[64]-a[63])

    def test_count_units_and_signed_bias(self):
        s = polar_scores([[.49, .0101], [.52, .0099]], [[.5, .01], [.5, .01]])
        self.assertAlmostEqual(s['CD_MAE_drag_counts'], 1.)
        self.assertAlmostEqual(s['CL_MAE'], .015)
        self.assertAlmostEqual(s['CL_signed_bias'], .005)
        self.assertAlmostEqual(s['CD_signed_bias_counts'], 0.)
        with self.assertRaises(ValueError): polar_scores([[np.nan, 1]], [[1, 1]])


if __name__ == '__main__': unittest.main()
