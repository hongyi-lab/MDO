"""A shared-geometry FM / ADflow bridge, with explicit reconstruction boundaries.

The native CFD mesh includes a blunt trailing edge and a rounded tip. Released
CRMpert ML arrays do not contain those patches, so this module never pretends
that meshing ``origingeom.npy`` reproduces the released CFD experiment.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .cfd import AEROTRANSFORMER_REVISION, run_adflow, validate_request
from .io import read_json, sha256_file, write_json

MESH_SCRIPT_SHA256 = "bc29faa2587d77aa1b45992c598172dc718807b1e5fb596c1e8bb3d0c571882c"
TIP_SHA256 = "49c08ffc533168e75880c461f75ff5b106c907e32028cc28a1826203829e6c95"
CST_REVISION = "9500b19a732463c26b32f6860ac6c517c7d212a5"
CONTROL_POINTS = [0.09999976658796647, 0.23391927215290928, 0.36783877771785206,
                  0.5258790832883891, 0.683919388858926, 0.841959694429463, 1.0]


def canonical_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_native_input(raw: dict) -> dict:
    """Require every native geometry parameter; do not infer missing constants."""
    if raw.get("schema_version") != 1 or not isinstance(raw.get("geometry"), dict):
        raise ValueError("Expected schema_version=1 with a complete geometry object")
    g = copy.deepcopy(raw["geometry"])
    required = {"SA", "half_span", "chords", "twists", "DAz", "cst_u", "cst_l"}
    if required - g.keys():
        raise ValueError(f"Missing native geometry parameters: {sorted(required - g.keys())}")
    for key, shape in (("chords", (7,)), ("twists", (7,)), ("DAz", (8,)),
                       ("cst_u", (7, 10)), ("cst_l", (7, 10))):
        array = np.asarray(g[key], dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"geometry.{key} must be finite with shape {shape}")
        g[key] = array.tolist()
    for key in ("SA", "half_span"):
        if isinstance(g[key], bool) or not math.isfinite(float(g[key])):
            raise ValueError(f"geometry.{key} must be finite")
        g[key] = float(g[key])
    if g["half_span"] <= 0 or min(g["chords"]) <= 0 or not -70 < g["SA"] < 70:
        raise ValueError("Require positive span/chords and sweep strictly between -70 and 70 degrees")
    cond = raw.get("condition", {})
    if set(cond) != {"mach", "alpha_deg"}:
        raise ValueError("condition must explicitly contain mach and alpha_deg only")
    if any(isinstance(v, bool) for v in cond.values()):
        raise ValueError("Conditions must be real numbers, not booleans")
    cond = {k: float(v) for k, v in cond.items()}
    if not all(math.isfinite(v) for v in cond.values()) or cond["mach"] <= 0:
        raise ValueError("Conditions must be finite and Mach positive")
    name = raw.get("case_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("case_name is required")
    return {"schema_version": 1, "case_name": name, "geometry": g, "condition": cond,
            "provenance": raw.get("provenance", {})}


def _native_surface_functions(upstream: Path) -> dict:
    """Load only geometry functions from the SHA-verified author's script.

    AST selection avoids importing pyHyp/MPI on a Windows preparation host. It
    preserves the geometry function bodies; no meshing or CFD runs implicitly.
    """
    script = upstream / "simulation" / "gen-mesh.crmpert.py"
    tip = upstream / "simulation" / "original_tip.xyz"
    if sha256_file(script) != MESH_SCRIPT_SHA256 or sha256_file(tip) != TIP_SHA256:
        raise ValueError("Native mesher/tip do not match the audited AeroTransformer revision")
    try:
        from cst_modeling.basic import rotation_3d
        from cst_modeling.io import read_plot3d, output_plot3d_concat
        from cst_modeling.foil import cst_foil
        from scipy.interpolate import CubicSpline
        from functools import reduce
    except ImportError as exc:
        raise RuntimeError(f"Install scipy and cst-modeling3d at {CST_REVISION}; see docs/MATCHED_CFD.md") from exc
    names = {"linear_smooth", "scaling_tip", "move_block", "power_growth0", "power_growth", "surface_meshing"}
    constants = {"twists0", "mesh_point_numbers", "xx0"}
    tree = ast.parse(script.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    namespace = {"np": np, "rotation_3d": rotation_3d, "read_plot3d": read_plot3d,
                 "output_plot3d_concat": output_plot3d_concat, "cst_foil": cst_foil,
                 "CubicSpline": CubicSpline, "reduce": reduce, "tip_path": str(tip.resolve())}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(script), "exec"), namespace)
    return namespace


def reference_from_native_blocks(blocks: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Sample the actual native main-wing patches for the FM, without fitting CST.

    Coordinates are linearly interpolated on the author's surface mesh. The
    negative-span CFD half-wing is mirrored to the positive-span FM convention.
    Tip and trailing-edge patches are retained in CFD, excluded from FM output.
    """
    if len(blocks) != 13:
        raise ValueError("The pinned native CRM mesher must produce exactly 13 surface patches")
    sides = []
    for start in (0, 4):
        lower, front, upper = [np.asarray(b)[:, :, 0, :] for b in blocks[start:start + 3]]
        ring = np.concatenate((lower[:-1], front[:-1], upper), axis=0).transpose(1, 0, 2)
        sides.append(ring)
    if not np.allclose(sides[0][-1], sides[1][0], rtol=0, atol=1e-9):
        raise ValueError("Native inner/outer main-wing patches do not meet")
    main = np.concatenate((sides[0][:-1], sides[1]), axis=0)
    main[:, :, 2] *= -1
    span = main[:, 0, 2]
    if not np.all(np.diff(span) > 0) or not np.all(np.isfinite(main)):
        raise ValueError("Native surface requires finite, increasing span coordinates")
    # The author's 117-point airfoil makes a 233-point lower-LE-upper ring.
    if main.shape[1] != 233:
        raise ValueError("Unexpected native circumferential topology")
    clustered = (1 - np.cos(np.linspace(0, np.pi, 129))) / 2
    sampled = np.empty((len(span), 257, 3))
    for j, section in enumerate(main):
        halves = []
        for part in (section[:117], section[116:]):
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(part, axis=0), axis=1))]
            if np.any(np.diff(arc) <= 0):
                raise ValueError("Repeated/degenerate native main-wing edge")
            halves.append(np.stack([np.interp(clustered * arc[-1], arc, part[:, k]) for k in range(3)], axis=1))
        sampled[j] = np.concatenate((halves[0], halves[1][1:]))
    target_span = np.linspace(span[0], span[-1], 129)
    target = np.empty((129, 257, 3))
    for i in range(257):
        for k in range(3):
            target[:, i, k] = np.interp(target_span, span, sampled[:, i, k])
    return target.transpose(2, 0, 1), {
        "method": "linear native-surface interpolation; cosine-clustered arc coordinate on each foil side",
        "source": "first 8 native patches, excluding trailing-edge patches 4/8 and rounded-tip patches 9-13",
        "span_mirroring": "CFD z<0 becomes FM z>0; pressure unchanged, spanwise vector component reverses",
        "native_main_vertices": list(main.shape), "fm_vertex_shape": [3, 129, 257],
        "released_CRMpert_sampling_reproduced": False,
        "calibration_transfer_allowed": False,
        "note": "New explicit surface sampling. Recalibrate on new CFD labels; existing CRMpert bounds do not apply."}


def prepare_bundle(input_path: Path, output: Path, upstream: Path, *, allow_reconstructed_case: bool = False) -> dict:
    """Create a portable bundle containing the shared native surface and FM input."""
    if not allow_reconstructed_case:
        raise ValueError("This creates a NEW case/sampling contract. Pass --allow-reconstructed-case explicitly; it is not exact released CRMpert reproduction.")
    definition = validate_native_input(read_json(input_path))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty bundle directory; do not mix geometry identities")
    functions = _native_surface_functions(upstream.resolve())
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        os.chdir(output)
        with (output / "surface_generation.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            geometry = functions["surface_meshing"](copy.deepcopy(definition["geometry"]), control_points=CONTROL_POINTS)
        with (output / "surface_generation.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            blocks = functions["read_plot3d"](str(output / "wing.xyz"))
    finally:
        os.chdir(old_cwd)
    original, sampling = reference_from_native_blocks(blocks)
    centers = .25 * (original[:, 1:, 1:] + original[:, 1:, :-1] + original[:, :-1, 1:] + original[:, :-1, :-1])
    condition = np.array([definition["condition"]["alpha_deg"], definition["condition"]["mach"]], dtype=np.float32)
    np.savez(output / "fm_input.npz", original_geometry=original, geometry=centers.astype(np.float32), condition=condition,
             ref_area=float(geometry["surface_area"]))
    write_json(output / "native_input.json", definition)
    identity = {"case_name": definition["case_name"], "native_input_sha256": sha256_file(output / "native_input.json"),
                "surface_sha256": sha256_file(output / "wing.xyz"), "fm_input_sha256": sha256_file(output / "fm_input.npz"),
                "condition": {"mach": float(condition[1]), "alpha_deg": float(condition[0]), "reynolds": 20_000_000.,
                              "reynolds_length_m": 1., "temperature_k": 300.},
                "reference": {"area_m2": float(geometry["surface_area"]), "chord_m": 1., "moment_center_m": [.25, 0., 0.]},
                "surface_sampling": sampling, "recipe_revision": AEROTRANSFORMER_REVISION,
                "geometry_definition": "new fully specified case; NOT original CRMpert CFD reproduction"}
    manifest = {"schema_version": 1, "case_sha256": canonical_hash(identity), "identity": identity,
                "surface_path": "wing.xyz", "fm_input_path": "fm_input.npz", "native_input_path": "native_input.json",
                "surface_generation_seconds": time.perf_counter() - start,
                "native_surface_quads": sum((b.shape[0]-1)*(b.shape[1]-1) for b in blocks),
                "estimated_volume_cells_at_81_layers": 80 * sum((b.shape[0]-1)*(b.shape[1]-1) for b in blocks),
                "calibration_status": "uncalibrated_new_geometry_and_sampling_contract",
                "matched_speedup_eligible": False, "cfd_executed": False,
                "provenance": definition["provenance"]}
    write_json(output / "request.json", manifest)
    return manifest


def load_bundle(request_path: Path) -> dict:
    request_path = request_path.resolve()
    manifest = read_json(request_path)
    if manifest.get("schema_version") != 1 or manifest.get("case_sha256") != canonical_hash(manifest["identity"]):
        raise ValueError("Bundle identity hash mismatch")
    root = request_path.parent
    for key, digest in (("surface_path", "surface_sha256"), ("fm_input_path", "fm_input_sha256"),
                        ("native_input_path", "native_input_sha256")):
        path = (root / manifest[key]).resolve()
        if path.parent != root or sha256_file(path) != manifest["identity"][digest]:
            raise ValueError(f"Bundle {key} missing, outside bundle, or modified")
    return manifest


def prediction_sample(request_path: Path) -> tuple[dict, dict]:
    manifest = load_bundle(request_path)
    with np.load(request_path.resolve().parent / manifest["fm_input_path"], allow_pickle=False) as arrays:
        sample = {k: np.array(arrays[k], copy=True) for k in ("original_geometry", "geometry", "condition")}
        sample["ref_area"] = float(arrays["ref_area"])
    return sample, manifest


def mesh_options(surface: Path, mach: float, *, wall_normal_layers: int = 81) -> dict:
    """Author's physical and pyHyp recipe, with explicit standard 81 layers."""
    if type(wall_normal_layers) is not int or wall_normal_layers < 9 or (wall_normal_layers - 1) % 8:
        raise ValueError("wall_normal_layers must be 8*k+1, >=9")
    if not math.isfinite(mach) or mach <= 0:
        raise ValueError("Mach must be positive and finite")
    speed = mach * math.sqrt(1.4 * 287 * 300.)
    first_spacing = (1.8375e-5 / 1.225) / math.sqrt(.5 * .0576 * 20_000_000.**(-.2) * speed**2)
    return {"inputFile": str(surface), "fileType": "PLOT3D", "unattachedEdgesAreSymmetry": True,
            "outerFaceBC": "farfield", "autoConnect": True,
            "BC": {i: {"jLow": "zSymm"} for i in range(1, 5)},
            "families": {i: ("mainwing" if i in (1, 2, 3, 5, 6, 7) else "trailingedge" if i in (4, 8) else "tip")
                         for i in range(1, 14)},
            "N": wall_normal_layers, "s0": first_spacing, "marchDist": 50., "ps0": -1., "pGridRatio": -1.,
            "cMax": 1., "epsE": 1., "epsI": 2., "theta": 3., "volCoef": .2, "volBlend": .0005,
            "volSmoothIter": 20, "kspRelTol": 1e-10, "kspMaxIts": 1500, "kspSubspaceSize": 50}


def run_bundle(request_path: Path, output: Path, *, comm=None, wall_normal_layers: int = 81) -> dict:
    """Collective pyHyp + ADflow execution on Linux, with no stale result reuse."""
    from mpi4py import MPI
    from pyhyp import pyHyp
    comm = MPI.COMM_WORLD if comm is None else comm
    start = time.perf_counter()
    manifest = load_bundle(request_path)
    output = output.resolve()
    if comm.rank == 0:
        if output.exists() and any(output.iterdir()):
            error = "Live output must be a new empty directory"
        else:
            output.mkdir(parents=True, exist_ok=True)
            error = None
    else:
        error = None
    error = comm.bcast(error, root=0)
    if error:
        raise ValueError(error)
    if comm.rank == 0:
        write_json(output / "result.json", {"status": "running", "case_sha256": manifest["case_sha256"],
                                           "convergence": {"converged": False}})
    comm.Barrier()
    opts = mesh_options(request_path.resolve().parent / manifest["surface_path"],
                        manifest["identity"]["condition"]["mach"], wall_normal_layers=wall_normal_layers)
    mesh_start = time.perf_counter()
    hyp = pyHyp(comm=comm, options=opts)
    hyp.run()
    min_volume = comm.allreduce(float(hyp.hyp.hypdata.minvolumeoverall), op=MPI.MIN)
    min_quality = comm.allreduce(float(hyp.hyp.hypdata.minqualityoverall), op=MPI.MIN)
    if not math.isfinite(min_volume) or not math.isfinite(min_quality) or min_volume <= 0 or min_quality <= 0:
        raise RuntimeError(f"Invalid generated mesh: minimum volume={min_volume}, minimum quality={min_quality}")
    volume = output / "wing_vol.cgns"
    hyp.writeCGNS(str(volume))
    comm.Barrier()
    mesh_seconds = comm.allreduce(time.perf_counter() - mesh_start, op=MPI.MAX)
    raw = {"schema_version": 1, "problem_id": "new-matched-wing-cfd-v1", "geometry_id": manifest["identity"]["case_name"],
           "geometry_sha256": manifest["identity"]["surface_sha256"], "mesh_path": str(volume),
           "condition": manifest["identity"]["condition"], "reference": manifest["identity"]["reference"],
           "numerics": {"preset": "aerotransformer"},
           "provenance": {"bundle_case_sha256": manifest["case_sha256"], "bundle_identity": manifest["identity"]}}
    normalized = validate_request(raw)
    result = run_adflow(normalized, output / "solver_surface", comm=comm,
                        function_groups={"mainwing": "mainwing", "trailingedge": "trailingedge", "tip": "tip"})
    result["volume_case_sha256"] = result["case_sha256"]
    result["case_sha256"] = manifest["case_sha256"]
    result["bundle_identity"] = manifest["identity"]
    result["timing"]["mesh_seconds"] = mesh_seconds
    result["timing"]["live_mesh_and_cfd_seconds"] = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
    result["mesh_options"] = opts
    result["mesh_quality"] = {"minimum_volume": min_volume, "minimum_quality": min_quality,
                              "positive_volume_and_quality": True, "mesh_independence_verified": False}
    result["coefficient_contract"] = "ADflow entire native wall, including tip and blunt trailing edge"
    result["mainwing_coefficient_contract"] = "ADflow integration on native patches 1,2,3,5,6,7; same physical subset as FM sampling; different discrete integration"
    result["structured_field_comparison"] = {"available": False,
        "reason": "Native CFD surface files are saved, but audited conservative transfer to the FM reference surface is not implemented."}
    result["matched_speedup_eligible"] = False
    result["speedup"] = None
    result["comparison_note"] = "Use group_coefficients.mainwing for FM comparison. Physical subset matches; discrete surface interpolation/integration error remains. Equal-accuracy speedup requires measured tolerances and new-case calibration."
    if comm.rank == 0:
        write_json(output / "result.json", result)
    return result


def paired_diagnostics(prediction: dict, cfd: dict) -> dict:
    """Reject wrong/nonconverged cases; keep unequal output definitions visible."""
    if not prediction.get("case_sha256") or prediction["case_sha256"] != cfd.get("case_sha256"):
        raise ValueError("FM and CFD case hashes differ")
    if cfd.get("status") != "ok" or not cfd.get("convergence", {}).get("converged"):
        raise ValueError("Nonconverged CFD cannot be used as reference")
    try:
        truth = cfd["group_coefficients"]["mainwing"]
    except KeyError as exc:
        raise ValueError("CFD mainwing family coefficients required; total-wall integrals are not interchangeable") from exc
    errors = {}
    for key in ("CL", "CD", "CM"):
        a, b = float(prediction["coefficients"][key]), float(truth[key])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Non-finite coefficients")
        errors[key] = abs(a-b)
    return {"case_sha256": prediction["case_sha256"], "coefficient_absolute_difference": errors,
            "interpretation": "Same physical main-wing subset; includes surface-resampling/discrete-integration differences, not model-only error.",
            "cfd_mainwing_coefficients": truth, "cfd_total_wall_coefficients": cfd["coefficients"],
            "fm_timing": prediction["timing"], "cfd_timing": cfd["timing"], "speedup": None,
            "equal_accuracy_verified": False, "matched_speedup_eligible": False}
