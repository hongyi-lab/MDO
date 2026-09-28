"""Geometry/physical-task tests; no synthetic outputs used in experiments."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mdo_demo.shape_design import (validate_config,default_design,morph_blocks,write_surface,
                                  geometry_measures,shape_margins)
from mdo_demo.aero_contract import read_native_surface
from mdo_demo.io import read_json


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.cfg=validate_config(read_json(Path(__file__).resolve().parents[1]/'configs/fm_shape_design_v1.json'))
        self.d=default_design(self.cfg)
        ring=np.array([[1,-.01],[.5,-.06],[0,0],[.5,.06],[1,.01]])
        self.v=np.empty((3,3,5))
        for i,z in enumerate([-.3,-1.65,-3.]):
            self.v[0,i]=ring[:,0]+.2*i;self.v[1,i]=ring[:,1];self.v[2,i]=z

    def test_identity_and_plot3d_roundtrip(self):
        out=morph_blocks([self.v]*13,self.v,self.d)
        np.testing.assert_allclose(out[0],self.v,rtol=0,atol=1e-15)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wing.xyz';write_surface(p,out)
            for a,b in zip(read_native_surface(p),out):np.testing.assert_array_equal(a,b)

    def test_twist_sign_and_root_fixed(self):
        self.d['twist_tip_deg']=2
        v=morph_blocks([self.v],self.v,self.d)[0]
        np.testing.assert_allclose(v[:,0],self.v[:,0],atol=1e-14)
        self.assertGreater(v[1,-1,2],self.v[1,-1,2]) # nose-up LE
        np.testing.assert_array_equal(v[2],self.v[2])

    def test_chord_span_volume_scaling(self):
        for k in ('chord_root_scale','chord_mid_scale','chord_tip_scale'):self.d[k]=1.1
        self.d['span_scale']=1.05
        v=morph_blocks([self.v],self.v,self.d)[0]
        a,b=geometry_measures(self.v),geometry_measures(v)
        self.assertAlmostEqual(b['projected_area_m2']/a['projected_area_m2'],1.1*1.05)
        self.assertAlmostEqual(b['enclosed_geometric_volume_m3']/a['enclosed_geometric_volume_m3'],1.1**2*1.05)
        np.testing.assert_allclose(v[2,0],self.v[2,0])

    def test_identical_shared_points_stay_identical(self):
        self.d.update(twist_mid_deg=1,twist_tip_deg=-2,chord_mid_scale=.95,sweep_delta_deg=2)
        a=self.v[:,:2];b=self.v[:,1:]
        out=morph_blocks([a,b],self.v,self.d)
        np.testing.assert_array_equal(out[0][:,-1],out[1][:,0])

    def test_lift_requirement_does_not_shrink_with_area(self):
        base=geometry_measures(self.v);small={k:v*.9 for k,v in base.items()}
        s={'ks_failure':.8,'tip_displacement_m':.1}
        m=shape_margins({'CL':.5,'CD':.03},s,small,base,self.cfg)
        self.assertEqual(m['lift'],0.)
        self.assertAlmostEqual(m['volume'],0.)

    def test_invalid_scales_rejected(self):
        self.d['span_scale']=0
        with self.assertRaises(ValueError):morph_blocks([self.v],self.v,self.d)


if __name__=='__main__':unittest.main()
