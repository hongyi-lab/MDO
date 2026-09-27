#!/usr/bin/env python3
"""Read-only CGNS audit of one ADflow restart against its original mesh.

Pinned runtime: CGNS 4.3, 32-bit cgsize_t. Uses the CGNS C API, not HDF5
layout assumptions. ADflow writes original CGNS zones after gathering MPI
partitions, so comparisons are by zone name rather than file order.
Only an optional, newly named JSON audit is written; input files are read-only.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

LIBRARY = "/home/mdolabuser/packages/CGNS-4.3.0/opt-gcc/lib/libcgns.so"
ATOL_M = 1e-10
REAL_DOUBLE = 4
STRUCTURED = 2
CELL_CENTER = 3
REQUIRED_FIELDS = (
    "Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure", "TurbulentSANuTilde"
)
Int = C.c_int
Size = C.c_int32


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class CGNS:
    def __init__(self, library: str):
        self.lib = C.CDLL(library)
        self.lib.cg_get_error.restype = C.c_char_p
        signatures = {
            "cg_open": [C.c_char_p, Int, C.POINTER(Int)],
            "cg_close": [Int],
            "cg_nbases": [Int, C.POINTER(Int)],
            "cg_base_read": [Int, Int, C.c_char_p, C.POINTER(Int), C.POINTER(Int)],
            "cg_nzones": [Int, Int, C.POINTER(Int)],
            "cg_zone_read": [Int, Int, Int, C.c_char_p, C.POINTER(Size)],
            "cg_zone_type": [Int, Int, Int, C.POINTER(Int)],
            "cg_ncoords": [Int, Int, Int, C.POINTER(Int)],
            "cg_coord_info": [Int, Int, Int, Int, C.POINTER(Int), C.c_char_p],
            "cg_coord_read": [Int, Int, Int, C.c_char_p, Int,
                              C.POINTER(Size), C.POINTER(Size), C.c_void_p],
            "cg_nsols": [Int, Int, Int, C.POINTER(Int)],
            "cg_sol_info": [Int, Int, Int, Int, C.c_char_p, C.POINTER(Int)],
            "cg_nfields": [Int, Int, Int, Int, C.POINTER(Int)],
            "cg_field_info": [Int, Int, Int, Int, Int, C.POINTER(Int), C.c_char_p],
            "cg_field_read": [Int, Int, Int, Int, C.c_char_p, Int,
                              C.POINTER(Size), C.POINTER(Size), C.c_void_p],
            "cg_rind_read": [C.POINTER(Int)],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = signature
            function.restype = Int
        # cg_goto is variadic: explicitly typed arguments are supplied below.
        self.lib.cg_goto.restype = Int

    def call(self, name, *args):
        code = int(getattr(self.lib, name)(*args))
        if code:
            message = self.lib.cg_get_error()
            raise RuntimeError(f"{name} returned {code}: {message.decode(errors='replace') if message else 'unknown CGNS error'}")

    def open(self, path: Path) -> int:
        handle = Int()
        self.call("cg_open", str(path).encode(), 0, C.byref(handle))  # CG_MODE_READ
        return handle.value

    def zones(self, handle: int) -> dict:
        count = Int()
        self.call("cg_nbases", handle, C.byref(count))
        if count.value != 1:
            raise ValueError("Require one CGNS base, as used by the pinned ADflow restart reader")
        name = C.create_string_buffer(33)
        cell_dim, phys_dim = Int(), Int()
        self.call("cg_base_read", handle, 1, name, C.byref(cell_dim), C.byref(phys_dim))
        if (cell_dim.value, phys_dim.value) != (3, 3):
            raise ValueError("Require a 3D volume mesh/restart, not a surface CGNS file")
        self.call("cg_nzones", handle, 1, C.byref(count))
        if count.value <= 0:
            raise ValueError("No volume zones found")
        output = {}
        for zone in range(1, count.value + 1):
            sizes = (Size * 9)()
            zone_type = Int()
            self.call("cg_zone_read", handle, 1, zone, name, sizes)
            self.call("cg_zone_type", handle, 1, zone, C.byref(zone_type))
            zone_name = name.value.decode()
            if zone_type.value != STRUCTURED or zone_name in output:
                raise ValueError("Require unique, structured original CGNS zones")
            nodes = tuple(int(v) for v in sizes[:3])
            cells = tuple(int(v) for v in sizes[3:6])
            if min(nodes) < 2 or cells != tuple(v - 1 for v in nodes):
                raise ValueError(f"Invalid structured volume dimensions for {zone_name}: {list(sizes)}")
            output[zone_name] = {"index": zone, "nodes": nodes, "cells": cells}
        return output

    def coordinates(self, handle, zone):
        count, datatype = Int(), Int()
        name = C.create_string_buffer(33)
        self.call("cg_ncoords", handle, 1, zone, C.byref(count))
        result = {}
        for index in range(1, count.value + 1):
            self.call("cg_coord_info", handle, 1, zone, index, C.byref(datatype), name)
            key = name.value.decode()
            if key in result:
                raise ValueError(f"Duplicate coordinate: {key}")
            result[key] = datatype.value
        if set(result) != {"CoordinateX", "CoordinateY", "CoordinateZ"}:
            raise ValueError(f"Unexpected coordinate names: {sorted(result)}")
        return result

    def read_coordinate(self, handle, zone, name, nodes):
        low, high = (Size * 3)(1, 1, 1), (Size * 3)(*nodes)
        values = np.empty(int(np.prod(nodes, dtype=np.int64)), dtype=np.float64)
        self.call("cg_coord_read", handle, 1, zone, name.encode(), REAL_DOUBLE,
                  low, high, C.c_void_p(values.ctypes.data))
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {name} in zone {zone}")
        return values

    def solution(self, handle, zone, cells):
        count, location, datatype = Int(), Int(), Int()
        name = C.create_string_buffer(33)
        self.call("cg_nsols", handle, 1, zone, C.byref(count))
        if count.value != 1:
            raise ValueError(f"Require exactly one flow solution per volume zone, found {count.value}")
        self.call("cg_sol_info", handle, 1, zone, 1, name, C.byref(location))
        solution_name = name.value.decode()
        if location.value != CELL_CENTER:
            raise ValueError("Require the original cell-centred checkpoint without vertex interpolation")
        self.call("cg_goto", Int(handle), Int(1), C.c_char_p(b"Zone_t"), Int(zone),
                  C.c_char_p(b"FlowSolution_t"), Int(1), C.c_char_p(b"end"))
        rind = (Int * 6)()
        self.call("cg_rind_read", rind)
        if any(value not in (0, 1) for value in rind):
            raise ValueError(f"Unexpected checkpoint rind: {list(rind)}")
        self.call("cg_nfields", handle, 1, zone, 1, C.byref(count))
        fields = {}
        for index in range(1, count.value + 1):
            self.call("cg_field_info", handle, 1, zone, 1, index, C.byref(datatype), name)
            key = name.value.decode()
            if key in fields:
                raise ValueError(f"Duplicate flow field: {key}")
            fields[key] = datatype.value
        missing = set(REQUIRED_FIELDS) - fields.keys()
        if missing:
            raise ValueError(f"Missing exact restart state fields: {sorted(missing)}")
        low, high = (Size * 3)(1, 1, 1), (Size * 3)(*cells)
        values = np.empty(int(np.prod(cells, dtype=np.int64)), dtype=np.float64)
        statistics = {}
        for field in REQUIRED_FIELDS:
            if fields[field] != REAL_DOUBLE:
                raise ValueError(f"Restart state field {field} is not RealDouble")
            self.call("cg_field_read", handle, 1, zone, 1, field.encode(), REAL_DOUBLE,
                      low, high, C.c_void_p(values.ctypes.data))
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite restart field {field}")
            minimum, maximum = float(values.min()), float(values.max())
            if field in ("Density", "Pressure") and minimum <= 0:
                raise ValueError(f"Nonpositive interior {field}")
            statistics[field] = {"datatype": fields[field], "finite": True,
                                 "minimum": minimum, "maximum": maximum}
        return {"name": solution_name, "location": "CellCenter", "rind": list(rind),
                "all_fields": fields, "required_interior_fields": statistics,
                "interior_cell_count": int(values.size),
                "field_validation_scope": "All required interior state values; halo values are not included in finite/positivity checks"}


def inspect(grid: Path, checkpoint: Path, library: str) -> dict:
    start = time.perf_counter()
    grid, checkpoint = grid.resolve(strict=True), checkpoint.resolve(strict=True)
    if grid == checkpoint or not grid.is_file() or not checkpoint.is_file():
        raise ValueError("Provide distinct original grid and completed volume checkpoint files")
    stat_before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (grid, checkpoint)}
    api = CGNS(library)
    handles = []
    try:
        for path in (grid, checkpoint):
            handles.append(api.open(path))
        original, restart = (api.zones(h) for h in handles)
        if original.keys() != restart.keys():
            raise ValueError("Checkpoint and original volume grid have different zone names")
        report = []
        max_difference = 0.0
        total_cells = 0
        for name in sorted(original):
            left, right = original[name], restart[name]
            if left["nodes"] != right["nodes"] or left["cells"] != right["cells"]:
                raise ValueError(f"Checkpoint/grid dimensions differ for {name}")
            original_types = api.coordinates(handles[0], left["index"])
            restart_types = api.coordinates(handles[1], right["index"])
            if any(value != REAL_DOUBLE for value in restart_types.values()):
                raise ValueError("Require double precision checkpoint grid coordinates")
            differences = {}
            for coordinate in sorted(original_types):
                a = api.read_coordinate(handles[0], left["index"], coordinate, left["nodes"])
                b = api.read_coordinate(handles[1], right["index"], coordinate, right["nodes"])
                difference = float(np.abs(a - b).max())
                if difference > ATOL_M:
                    raise ValueError(f"Wrong restart geometry: {name}/{coordinate} differs by {difference:.17g} m > {ATOL_M} m")
                differences[coordinate] = difference
                max_difference = max(max_difference, difference)
            solution = api.solution(handles[1], right["index"], right["cells"])
            total_cells += solution["interior_cell_count"]
            report.append({"zone": name, "nodes": list(left["nodes"]), "cells": list(left["cells"]),
                           "grid_coordinate_types": original_types, "restart_coordinate_types": restart_types,
                           "coordinate_max_abs_difference_m": differences, "solution": solution})
    finally:
        for handle in reversed(handles):
            api.call("cg_close", handle)
    hashes = {str(p): digest(p) for p in (grid, checkpoint)}
    stat_after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (grid, checkpoint)}
    if stat_before != stat_after:
        raise ValueError("Input file changed during read-only audit")
    return {"schema_version": 1, "passed": True, "status": "grid_and_checkpoint_fields_verified",
            "library": library, "cgsize_t_bits": 32, "coordinate_absolute_tolerance_m": ATOL_M,
            "coordinate_relative_tolerance": 0.0, "max_coordinate_difference_m": max_difference,
            "original_grid": str(grid), "checkpoint": str(checkpoint), "sha256": hashes,
            "mesh_sha256": hashes[str(grid)], "checkpoint_sha256": hashes[str(checkpoint)],
            "zone_count": len(report), "total_interior_cells": total_cells, "zones": report,
            "wall_seconds": time.perf_counter() - start,
            "scope": "Grid identity and restart state integrity only; not convergence, physical case metadata, or qualified CFD/MDO accuracy"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", default=LIBRARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Audit output already exists; use a fresh JSON path")
    try:
        result = inspect(args.grid, args.checkpoint, args.library)
        code = 0
    except Exception as exc:
        result = {"schema_version": 1, "passed": False, "status": "restart_audit_failed",
                  "error_type": type(exc).__name__, "error": str(exc)}
        code = 2
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
