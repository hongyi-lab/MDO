#!/usr/bin/env python3
"""Export converged ADflow's native Cp/Cf cells, with exact surface identity checks.

This preserves native CFD cells. It does not interpolate to the FM tensor or
claim conservative transfer. Run in the audited official image (CGNS 4.3, int32).
"""
from __future__ import annotations
import argparse
import ctypes as ct
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdo_demo.io import read_json, sha256_file, write_json
from mdo_demo.matched_cfd import load_bundle, pressure_convention_audit


def native_centers(path: Path):
    tokens = path.read_text().split()
    blocks = int(tokens[0])
    dims = np.asarray(tokens[1:1+3*blocks], dtype=int).reshape(blocks, 3)
    values = np.asarray(tokens[1+3*blocks:], dtype=float)
    centers, patches = [], []
    offset = 0
    for patch, (ni, nj, nk) in enumerate(dims, 1):
        if nk != 1 or min(ni, nj) < 2:
            raise ValueError("Expected the native 13-block two-dimensional PLOT3D surface")
        count = ni*nj*nk
        xyz = values[offset:offset+3*count].reshape(3, nj, ni).transpose(1, 2, 0)
        offset += 3*count
        centers.append(((xyz[:-1,:-1]+xyz[1:,:-1]+xyz[:-1,1:]+xyz[1:,1:])/4).reshape(-1,3))
        patches.extend([patch]*((ni-1)*(nj-1)))
    if offset != values.size or blocks != 13:
        raise ValueError("Unexpected native surface layout")
    return np.concatenate(centers), np.asarray(patches, dtype=np.int32)


def read_surface(path: Path, library: Path):
    # The audited image is int32. Refuse another ABI instead of risking a buffer
    # overflow from assuming cgsize_t's width in an arbitrary CGNS build.
    header = library.parent.parent / "include" / "cgnstypes.h"
    if not header.is_file() or not re.search(r"^#define\s+CG_BUILD_64BIT\s+0\s*$", header.read_text(), re.M):
        raise RuntimeError("This reader requires the audited CGNS int32 build; header check failed")
    lib = ct.CDLL(str(library))
    lib.cg_get_error.restype = ct.c_char_p
    i, p = ct.c_int, ct.POINTER(ct.c_int)
    lib.cg_open.argtypes = [ct.c_char_p, i, p]
    lib.cg_close.argtypes = [i]
    lib.cg_base_read.argtypes = [i, i, ct.c_char_p, p, p]
    lib.cg_nzones.argtypes = [i, i, p]
    lib.cg_zone_read.argtypes = [i, i, i, ct.c_char_p, p]
    lib.cg_zone_type.argtypes = [i, i, i, p]
    lib.cg_sol_info.argtypes = [i, i, i, i, ct.c_char_p, p]
    lib.cg_coord_read.argtypes = [i, i, i, ct.c_char_p, i, p, p, ct.c_void_p]
    lib.cg_field_read.argtypes = [i, i, i, i, ct.c_char_p, i, p, p, ct.c_void_p]
    def check(code):
        if code:
            raise RuntimeError(lib.cg_get_error().decode())
    fn = i()
    check(lib.cg_open(str(path).encode(), 0, ct.byref(fn)))
    centers, fields, zone_ids, zones = [], [], [], []
    try:
        name, cell_dim, phys_dim = ct.create_string_buffer(33), i(), i()
        check(lib.cg_base_read(fn, 1, name, ct.byref(cell_dim), ct.byref(phys_dim)))
        if cell_dim.value != 2 or phys_dim.value != 3:
            raise ValueError("Expected a structured 2D wall surface embedded in 3D")
        count = i()
        check(lib.cg_nzones(fn, 1, ct.byref(count)))
        for z in range(1, count.value+1):
            size, kind = (i*6)(), i()
            check(lib.cg_zone_type(fn, 1, z, ct.byref(kind)))
            check(lib.cg_zone_read(fn, 1, z, name, size))
            if kind.value != 2:
                raise ValueError("Unstructured wall zones are not supported by this reader")
            ni, nj, ci, cj = list(size)[:4]
            if ci != ni-1 or cj != nj-1:
                raise ValueError("Unexpected vertex/cell dimensions")
            zone_name = name.value.decode()
            location = i()
            check(lib.cg_sol_info(fn, 1, z, 1, name, ct.byref(location)))
            if location.value != 3:
                raise ValueError("Expected CellCenter Cp/Cf fields")
            xyz = np.empty((3, nj, ni), dtype=np.float64)
            begin, end = (i*2)(1, 1), (i*2)(ni, nj)
            for k, field in enumerate((b"CoordinateX", b"CoordinateY", b"CoordinateZ")):
                check(lib.cg_coord_read(fn, 1, z, field, 4, begin, end, xyz[k].ctypes.data))
            values = np.empty((5, cj, ci), dtype=np.float64)
            end = (i*2)(ci, cj)
            for k, field in enumerate((b"CoefPressure", b"SkinFrictionX", b"SkinFrictionY", b"SkinFrictionZ", b"YPlus")):
                check(lib.cg_field_read(fn, 1, z, 1, field, 4, begin, end, values[k].ctypes.data))
            cen = (xyz[:,:-1,:-1]+xyz[:,1:,:-1]+xyz[:,:-1,1:]+xyz[:,1:,1:])/4
            centers.append(cen.reshape(3,-1).T)
            fields.append(values.reshape(5,-1).T)
            zone_ids.extend([z]*(ci*cj))
            zones.append({"zone": z, "name": zone_name, "vertex_shape": [nj,ni], "cell_count": ci*cj})
    finally:
        check(lib.cg_close(fn))
    return np.concatenate(centers), np.concatenate(fields), np.asarray(zone_ids), zones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=Path("/home/mdolabuser/packages/CGNS-4.3.0/opt-gcc/lib/libcgns.so"))
    args = parser.parse_args()
    start = time.perf_counter()
    manifest, result = load_bundle(args.request), read_json(args.result)
    if result.get("case_sha256") != manifest["case_sha256"] or not result.get("convergence", {}).get("converged"):
        raise ValueError("Require the same verified case and a converged CFD result")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Choose fresh output names")
    audit = pressure_convention_audit()
    if not audit["verified"]:
        raise ValueError("Installed pressure integration source is not the audited version")
    native, native_patch = native_centers(args.request.parent/manifest["surface_path"])
    centers, fields, zones, zone_info = read_surface(args.surface, args.library)
    distance, index = cKDTree(native).query(centers)
    tolerance = 1e-8 * max(1., float(np.ptp(native, axis=0).max()))
    if not np.isfinite(fields).all() or distance.max() > tolerance:
        raise ValueError(f"Nonfinite fields or geometry mismatch: max distance {distance.max()} > {tolerance}")
    if len(index) != len(native) or len(np.unique(index)) != len(native):
        raise ValueError("CFD wall cells do not map one-to-one onto every original surface cell")
    patch = native_patch[index]
    # Fields remain in CFD coordinates, with span in negative z. Mirror z and
    # Cf_z together when plotting them with the FM's positive-z convention.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, centers=centers, cp=fields[:,0], cf_xyz=fields[:,1:4], yplus=fields[:,4],
                        native_patch=patch, cgns_zone=zones, native_cell_index=index,
                        mainwing_mask=np.isin(patch, [1,2,3,5,6,7]))
    write_json(args.output.with_suffix(".json"), {"case_sha256": manifest["case_sha256"],
        "surface_sha256": sha256_file(args.surface), "output_sha256": sha256_file(args.output),
        "native_cell_count": len(native), "cell_coordinate_match_max_m": float(distance.max()),
        "cell_coordinate_match_tolerance_m": tolerance, "one_to_one_native_cell_match": True,
        "fields": ["Cp", "Cf_x", "Cf_y", "Cf_z", "yplus"], "coordinate_convention": "native CFD negative-z span",
        "pressure_convention_audit": audit, "zones": zone_info, "postprocess_seconds": time.perf_counter()-start,
        "field_ranges": {key: [float(fields[:,k].min()), float(fields[:,k].max())]
                         for k,key in enumerate(["Cp", "Cf_x", "Cf_y", "Cf_z", "yplus"])},
        "fm_grid_transfer": False, "equal_accuracy_verified": False,
        "note": "Raw native CFD cells with exact original-cell identity; no interpolation or conservative FM transfer claimed."})
    print(f"Exported {len(native)} verified native CFD cells to {args.output}")


if __name__ == "__main__":
    main()
