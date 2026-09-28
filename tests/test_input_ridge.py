import unittest
import numpy as np
from mdo_demo.input_contract import encode_condition, encode_stw_surface, decode_stw_vertices
from mdo_demo.ridge_probe import fit_ridge, predict_ridge, select_ridge


class InputContractTests(unittest.TestCase):
    def test_frame_length_area_and_roundtrip(self):
        span, ring = np.meshgrid(np.linspace(0,14,129), np.linspace(0,2*np.pi,257), indexing='ij')
        native = np.stack([2.5*(1+np.cos(ring)), span, .3*np.sin(ring)], -1)
        data = encode_stw_surface(native, root_chord_m=5., reference_half_area_m2=45.5)
        np.testing.assert_allclose(decode_stw_vertices(data['original_geometry'], root_chord_m=5.), native, atol=1e-14)
        np.testing.assert_allclose(data['original_geometry'][2,:,0], np.linspace(0,2.8,129))
        self.assertAlmostEqual(data['ref_area'], 1.82)
        self.assertEqual(data['geometry'].shape, (3,128,256))
        self.assertEqual(data['geometry'].dtype, np.float32)
        native[:, -1, 2] += .01
        with self.assertRaises(ValueError): encode_stw_surface(native, root_chord_m=5., reference_half_area_m2=45.5)

    def test_condition_not_swapped_scaled_clipped_or_offset(self):
        for a in [-1., 0., .2, 8.7166]:
            np.testing.assert_array_equal(encode_condition(alpha_deg=a, mach=.77), np.array([a,.77],np.float32))
        with self.assertRaises(ValueError): encode_condition(alpha_deg=np.nan, mach=.77)
        with self.assertRaises(ValueError): encode_condition(alpha_deg=1, mach=0)


class RidgeTests(unittest.TestCase):
    def test_linear_recovery_and_constant_column(self):
        x = np.column_stack([np.arange(12.), np.ones(12)])
        y = np.column_stack([2*x[:,0]+3, -.1*x[:,0]+1])
        fit = fit_ridge(x[:10], y[:10], 1e-8)
        np.testing.assert_allclose(predict_ridge(fit,x[10:]), y[10:], atol=1e-7)
        self.assertTrue(np.isfinite(fit['weight']).all())

    def test_scaling_invariance(self):
        rng=np.random.default_rng(8); x=rng.normal(size=(12,5)); y=rng.normal(size=(12,2))
        f1=fit_ridge(x,y,3.); f2=fit_ridge(7*x+100,y,3.)
        np.testing.assert_allclose(predict_ridge(f1,x), predict_ridge(f2,7*x+100), atol=1e-10)

    def test_eval_arrays_cannot_affect_cv_or_fit(self):
        rng=np.random.default_rng(1); x=rng.normal(size=(14,5)); y=rng.normal(size=(14,2)); train=np.arange(0,14,2)
        f1,c1=select_ridge(x[train],y[train],[.1,1,10])
        x[1::2] += 1e6; y[1::2] += 1e9
        f2,c2=select_ridge(x[train],y[train],[.1,1,10])
        np.testing.assert_array_equal(f1['weight'], f2['weight']); self.assertEqual(c1,c2)


if __name__=='__main__': unittest.main()
