"""Regression: ADflow surface files include non-body boundary condition zones."""
import ctypes as ct
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_surface_export",ROOT/"scripts/export_native_cfd_surface.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class FakeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self,*args):
        self.function(*args)
        return 0


class NativeSurfaceExportTests(unittest.TestCase):
    def test_only_explicit_adiabatic_wall_zone_names_are_selected(self):
        self.assertTrue(export.is_native_wall_zone("NSWallAdiabaticBCZone0014"))
        for name in ("FarFieldBCZone1","SymmetryBCZone27","NSWallAdiabaticBCZoneX",
                     "PrefixNSWallAdiabaticBCZone1","NSWallIsothermalBCZone1"):
            self.assertFalse(export.is_native_wall_zone(name))

    def test_storage_roundoff_is_derived_from_each_coordinate_type(self):
        vertices = np.array([[[1.,2.],[1.,2.]],[[.1,.2],[.1,.2]],[[3.,3.],[4.,4.]]])
        keys = ("CoordinateX","CoordinateY","CoordinateZ")
        single = export.coordinate_storage_roundoff_bound(vertices,dict.fromkeys(keys,"RealSingle"))
        double = export.coordinate_storage_roundoff_bound(vertices,dict.fromkeys(keys,"RealDouble"))
        actual = np.max(np.abs(vertices.astype(np.float32).astype(float)-vertices),axis=(1,2))
        self.assertTrue(np.all(actual <= single["per_axis_m"]))
        self.assertGreater(single["euclidean_m"],double["euclidean_m"]*1e8)
        with self.assertRaisesRegex(ValueError,"storage precision"):
            export.coordinate_storage_roundoff_bound(vertices,dict.fromkeys(keys,"Integer"))

    def test_reader_does_not_read_farfield_or_symmetry_fields(self):
        class Library:
            pass
        lib = Library()
        names = {1:b"FarFieldBCZone1",2:b"NSWallAdiabaticBCZone14",3:b"SymmetryBCZone27"}
        coords = {b"CoordinateX":[0.,1.,0.,1.],b"CoordinateY":[0.,0.,0.,0.],b"CoordinateZ":[0.,0.,1.,1.]}
        field_zones = []
        def set_value(pointer,value):
            pointer._obj.value = value
        def zone(fn,base,z,name,size):
            name.value = names[z]
            size[:] = [2,2,1,1,0,0]
        def base(fn,b,name,cell,phys):
            set_value(cell,2)
            set_value(phys,3)
        def coord_info(fn,b,z,index,kind,name):
            set_value(kind,4)
            name.value = list(coords)[index-1]
        def coord_read(fn,b,z,field,kind,begin,end,pointer):
            for index,value in enumerate(coords[field]):
                ct.cast(pointer,ct.POINTER(ct.c_double))[index] = value
        def field_read(fn,b,z,sol,field,kind,begin,end,pointer):
            field_zones.append(z)
            ct.cast(pointer,ct.POINTER(ct.c_double))[0] = 1.
        callbacks = dict(
            cg_get_error=lambda:b"fake error",cg_open=lambda path,mode,fn:set_value(fn,1),cg_close=lambda fn:None,
            cg_base_read=base,cg_nzones=lambda fn,b,p:set_value(p,3),cg_zone_read=zone,
            cg_zone_type=lambda fn,b,z,p:set_value(p,2),
            cg_ncoords=lambda fn,b,z,p:set_value(p,3),cg_coord_info=coord_info,
            cg_sol_info=lambda fn,b,z,sol,name,p:set_value(p,3),cg_coord_read=coord_read,cg_field_read=field_read)
        for key, callback in callbacks.items():
            setattr(lib,key,FakeFunction(callback))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"include").mkdir()
            (root/"include/cgnstypes.h").write_text("#define CG_BUILD_64BIT 0\n")
            with patch.object(export.ct,"CDLL",return_value=lib):
                centers,fields,zones,info = export.read_surface(root/"surface.cgns",root/"lib/libcgns.so")
        np.testing.assert_array_equal(centers,[[.5,0.,.5]])
        np.testing.assert_array_equal(fields,np.ones((1,5)))
        np.testing.assert_array_equal(zones,[2])
        self.assertEqual(field_zones,[2]*5)
        self.assertEqual([item["selected_for_body_loads"] for item in info],[False,True,False])
        self.assertEqual(info[1]["coordinate_storage_types"]["CoordinateX"],"RealDouble")


if __name__ == "__main__":
    unittest.main()
