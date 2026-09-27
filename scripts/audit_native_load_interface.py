#!/usr/bin/env python3
"""Audit native field conversion, including archived unconverged solver fields.

Always diagnostic-only: no CFD solve, model inference, optimization, reference
accuracy or speedup is produced. Source convergence records are never modified.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from export_native_cfd_surface import native_centers, read_surface, match_native_cells
from mdo_demo.aero_contract import from_native_export, make_request, native_mainwing_vertices, save_loads
from mdo_demo.io import read_json, sha256_file, write_json
from mdo_demo.matched_cfd import load_bundle, pressure_convention_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("request", "result", "surface", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--library", type=Path, default=Path("/home/mdolabuser/packages/CGNS-4.3.0/opt-gcc/lib/libcgns.so"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = load_bundle(args.request)
    solver = read_json(args.result)
    if manifest["case_sha256"] != solver["case_sha256"]:
        raise ValueError("Source solver case does not match the native geometry/condition bundle")
    wing = args.request.parent / manifest["surface_path"]
    original, patch_ids = native_centers(wing)
    points, fields, zones, zone_info = read_surface(args.surface, args.library)
    index, coordinate_audit = match_native_cells(wing, points, zones, zone_info)
    distance = np.linalg.norm(points - original[index], axis=1)
    if not np.isfinite(fields).all():
        raise ValueError("Nonfinite native CFD fields")
    path = args.output / "native_surface.npz"
    patch = patch_ids[index]
    np.savez_compressed(path, centers=points, cp=fields[:, 0], cf_xyz=fields[:, 1:4], yplus=fields[:, 4],
                        native_patch=patch, cgns_zone=zones, native_cell_index=index,
                        mainwing_mask=np.isin(patch, [1, 2, 3, 5, 6, 7]))
    meta = {"case_sha256": manifest["case_sha256"], "surface_sha256": sha256_file(args.surface),
            "output_sha256": sha256_file(path), "one_to_one_native_cell_match": True,
            "coordinate_convention": "native CFD negative-z span", "pressure_convention_audit": pressure_convention_audit(),
            "postprocess_seconds": time.perf_counter() - start, "zones": zone_info,
            "cell_coordinate_match_max_m": float(distance.max()), "diagnostic_only": True,
            "coordinate_precision_audit": coordinate_audit,
            "qualified_flow_reference": False, "source_result_sha256": sha256_file(args.result)}
    write_json(path.with_suffix(".json"), meta)
    from baseclasses import AeroProblem
    identity = manifest["identity"]
    c = identity["condition"]
    ap = AeroProblem(name="interface_audit", mach=c["mach"], alpha=c["alpha_deg"], reynolds=c["reynolds"],
                     reynoldsLength=c["reynolds_length_m"], T=c["temperature_k"])
    vertices = native_mainwing_vertices(wing)
    span = vertices[2].mean(axis=1)
    request = make_request(manifest["case_sha256"], identity["surface_sha256"],
                          condition=dict(c, dynamic_pressure_pa=float(.5*ap.rho*ap.V**2)),
                          reference=identity["reference"], coverage={"families": ["mainwing"], "closed_TE": False,
                          "span_bounds_m": [float(span[-1]), float(span[0])]})
    loads = from_native_export(request, wing, path, path.with_suffix(".json"), args.result,
                               allow_unconverged_interface_audit=True)
    save_loads(loads, args.output / "diagnostic_loads.json")
    summary = {"status": loads["status"], "integration_audit": loads["integration_audit"],
               "coefficients": loads["coefficients"], "solver_family_coefficients": loads["solver_family_coefficients"],
               "dynamic_pressure_pa": request["identity"]["condition"]["dynamic_pressure_pa"],
               "source_converged": solver["convergence"]["converged"], "qualified_flow_reference": False,
               "source_result_sha256": sha256_file(args.result), "elapsed_seconds": time.perf_counter()-start}
    write_json(args.output / "audit.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
