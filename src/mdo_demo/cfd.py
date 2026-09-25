"""ADflow forward/target-lift runs with explicit physical identity and provenance.

ADflow is imported only inside run_adflow; validation works on Windows without MPI.
This is an independently written adapter to ADflow's public Python interface.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any

AEROTRANSFORMER_REVISION = "3dc350ff69e354d1451eae468c686a368bd6151f"
ADFLOW_API_AUDIT_REVISION = "8155e98119ec138f805a916400837cd27c41b961"
MACH_AERO_REVISION = "47545f536d41bd8075ee7e7dd750fb078edffe63"
TUTORIAL_MESH_SHA256 = "72d6d58cf9cbb086a6920dfc67713e31b35991a28c3cd6b96b7392e120d7915b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _keys(data: dict, required: set, optional: set, label: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    if required - data.keys():
        raise ValueError(f"{label} missing: {sorted(required - data.keys())}")
    if data.keys() - required - optional:
        raise ValueError(f"{label} unknown fields: {sorted(data.keys() - required - optional)}")


def validate_request(raw: dict, base_dir: Path = Path(".")) -> dict:
    """Normalize and hash an explicit request; never infer missing flow conditions."""
    _keys(raw, {"schema_version", "problem_id", "geometry_id", "mesh_path", "condition", "reference"},
          {"geometry_sha256", "numerics", "provenance"}, "request")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for name in ("problem_id", "geometry_id"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    geometry_sha = raw.get("geometry_sha256")
    if geometry_sha is not None and not re.fullmatch("[a-f0-9]{64}", geometry_sha):
        raise ValueError("geometry_sha256 must be lowercase SHA256")
    mesh = Path(raw["mesh_path"])
    mesh = (base_dir / mesh).resolve() if not mesh.is_absolute() else mesh.resolve()
    if not mesh.is_file() or mesh.stat().st_size == 0:
        raise ValueError(f"mesh file missing/empty: {mesh}")
    if mesh.suffix.lower() != ".cgns":
        raise ValueError("mesh_path must point to a CGNS volume mesh")
    cond = dict(raw["condition"])
    _keys(cond, {"mach", "reynolds", "reynolds_length_m", "temperature_k"},
          {"alpha_deg", "target_cl", "alpha_initial_deg"}, "condition")
    if ("alpha_deg" in cond) == ("target_cl" in cond):
        raise ValueError("condition requires exactly one of alpha_deg or target_cl")
    if "target_cl" in cond and "alpha_initial_deg" not in cond:
        raise ValueError("target_cl requires explicit alpha_initial_deg")
    if "alpha_deg" in cond and "alpha_initial_deg" in cond:
        raise ValueError("alpha_initial_deg is only valid with target_cl")
    for key in cond:
        cond[key] = _number(cond[key], f"condition.{key}", positive=key in {
            "mach", "reynolds", "reynolds_length_m", "temperature_k"})
    ref = dict(raw["reference"])
    _keys(ref, {"area_m2", "chord_m", "moment_center_m"}, set(), "reference")
    for key in ("area_m2", "chord_m"):
        ref[key] = _number(ref[key], f"reference.{key}", positive=True)
    if not isinstance(ref["moment_center_m"], list) or len(ref["moment_center_m"]) != 3:
        raise ValueError("reference.moment_center_m must be [x,y,z]")
    ref["moment_center_m"] = [_number(x, "moment_center_m") for x in ref["moment_center_m"]]
    supplied = raw.get("numerics", {})
    _keys(supplied, set(), {"preset", "l2_convergence", "max_cycles", "coarse_cycles",
                           "trim_tolerance", "trim_max_iterations"}, "numerics")
    preset = supplied.get("preset", "aerotransformer")
    if preset not in {"aerotransformer", "tutorial"}:
        raise ValueError("numerics.preset must be aerotransformer or tutorial")
    numerics = {"preset": preset, "l2_convergence": 1e-10 if preset == "aerotransformer" else 1e-6,
                "max_cycles": 3000 if preset == "aerotransformer" else 1000,
                "coarse_cycles": 500, "trim_tolerance": 1e-4, "trim_max_iterations": 20}
    numerics.update(supplied)
    for key in ("l2_convergence", "trim_tolerance"):
        numerics[key] = _number(numerics[key], f"numerics.{key}", positive=True)
    if numerics["l2_convergence"] >= 1:
        raise ValueError("l2_convergence must be below 1")
    for key in ("max_cycles", "coarse_cycles", "trim_max_iterations"):
        if type(numerics[key]) is not int or numerics[key] <= 0:
            raise ValueError(f"numerics.{key} must be a positive integer")
    identity = {"problem_id": raw["problem_id"], "geometry_id": raw["geometry_id"],
                "geometry_sha256": geometry_sha, "volume_mesh_sha256": sha256_file(mesh),
                "condition": cond, "reference": ref, "equations": "steady RANS", "turbulence_model": "SA",
                "coefficient_definition": "ADflow wall integration; CM is CMz about explicit reference"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema_version": 1, "mesh_path": str(mesh), "identity": identity,
            "case_sha256": hashlib.sha256(encoded).hexdigest(), "numerics": numerics,
            "provenance": raw.get("provenance", {})}


def solver_options(request: dict, output_dir: Path) -> dict:
    n = request["numerics"]
    options = {"gridFile": request["mesh_path"], "outputDirectory": str(output_dir),
               "monitorvariables": ["resrho", "cl", "cd", "cmz", "resturb"],
               "writeTecplotSurfaceSolution": True, "writeVolumeSolution": False,
               "surfaceVariables": ["cp", "cfx", "cfy", "cfz", "yplus"],
               "equationType": "RANS", "turbulenceModel": "SA", "useANKSolver": True,
               "L2Convergence": n["l2_convergence"], "L2ConvergenceRel": 1e-16,
               "nCycles": n["max_cycles"], "nCyclesCoarse": n["coarse_cycles"]}
    if n["preset"] == "aerotransformer":
        options.update(MGCycle="3w", useNKSolver=False, NKSwitchTol=1e-8)
    else:
        options.update(MGCycle="sg", nSubiterTurb=10, useNKSolver=True, NKSwitchTol=1e-4)
    return options


def convergence_report(norms: tuple, *, tolerance: float, solve_failed: bool,
                       fatal_failed: bool, coefficients: dict, target_cl: float | None = None,
                       trim_tolerance: float = 1e-4, trim_converged: bool | None = None,
                       solved_alpha_deg: float | None = None) -> dict:
    initial, start, final = (float(x) for x in norms)
    finite_norms = all(math.isfinite(x) for x in (initial, start, final))
    ratio = final / initial if finite_norms and initial > 0 and final >= 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    finite_coefficients = bool(coefficients) and all(math.isfinite(float(x)) for x in coefficients.values())
    finite_alpha = None if solved_alpha_deg is None else math.isfinite(float(solved_alpha_deg))
    lift_error = None if target_cl is None else abs(coefficients.get("CL", math.inf) - target_cl)
    if lift_error is not None and not math.isfinite(lift_error):
        lift_error = None
    residual_pass = ratio is not None and ratio <= tolerance
    trim_pass = target_cl is None or (trim_converged is True and lift_error is not None and lift_error < trim_tolerance)
    accepted = residual_pass and trim_pass and finite_coefficients and finite_alpha is not False and not solve_failed and not fatal_failed
    return {"converged": bool(accepted), "solve_failed": bool(solve_failed), "fatal_failed": bool(fatal_failed),
            "residual_initial": initial if math.isfinite(initial) else None,
            "residual_start": start if math.isfinite(start) else None,
            "residual_final": final if math.isfinite(final) else None,
            "residual_relative_to_freestream": ratio, "required_l2_convergence": tolerance,
            "finite_coefficients": finite_coefficients, "target_cl_absolute_error": lift_error,
            "finite_solved_alpha": finite_alpha,
            "trim_converged": trim_converged, "residual_pass": residual_pass, "trim_pass": trim_pass}


def memory_report(peak_rss_kib_by_rank: list) -> dict:
    """Summarize Linux per-process RSS high-water marks, not simultaneous job RSS."""
    measured = [float(value) for value in peak_rss_kib_by_rank if value is not None]
    complete = len(measured) == len(peak_rss_kib_by_rank) and bool(measured)
    return {"source": "Linux resource.getrusage(RUSAGE_SELF).ru_maxrss (KiB)",
            "all_ranks_measured": complete,
            "per_rank_peak_rss_mib": [None if value is None else float(value) / 1024 for value in peak_rss_kib_by_rank],
            "max_rank_peak_rss_mib": max(measured) / 1024 if measured else None,
            "sum_rank_peak_rss_mib": sum(measured) / 1024 if complete else None,
            "scope": "Per-rank process-lifetime high-water RSS. Sum is NOT simultaneous job peak; shared pages may be counted in multiple ranks. Excludes external MPI launcher and other processes."}


def _linux_peak_rss_kib() -> float | None:
    if platform.system() != "Linux":
        return None
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError):
        return None


def history_major_iterations(history: dict) -> int | None:
    """ADflow stores an initial row then one row per major iteration."""
    for key, values in history.items():
        if key.lower() == "total minor iters":
            return max(0, len(values) - 1)
    return None


def _module_provenance(module: Any) -> dict:
    location = Path(module.__file__).resolve()
    try:
        version = importlib.metadata.version(module.__name__)
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "unknown"))
    try:
        revision = subprocess.run(["git", "-C", str(location.parent), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=3, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"version": version, "git_revision": revision, "module_file": str(location)}


def run_adflow(request: dict, output_dir: Path, *, comm=None) -> dict:
    """Collective across MPI ranks. Return nonconverged results for diagnostics only."""
    total_start = time.perf_counter()
    import adflow
    from adflow import ADFLOW
    from baseclasses import AeroProblem
    from mpi4py import MPI

    comm = MPI.COMM_WORLD if comm is None else comm
    output_dir = output_dir.resolve()
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    opts = solver_options(request, output_dir)
    identity = request["identity"]
    cond, ref = identity["condition"], identity["reference"]
    solver = ADFLOW(comm=comm, options=opts)
    ap = AeroProblem(name="wing", mach=cond["mach"],
                     alpha=cond.get("alpha_deg", cond.get("alpha_initial_deg")),
                     reynolds=cond["reynolds"], reynoldsLength=cond["reynolds_length_m"],
                     T=cond["temperature_k"], areaRef=ref["area_m2"], chordRef=ref["chord_m"],
                     xRef=ref["moment_center_m"][0], yRef=ref["moment_center_m"][1],
                     zRef=ref["moment_center_m"][2], evalFuncs=["cl", "cd", "cmx", "cmy", "cmz"])
    comm.Barrier()
    setup_seconds = time.perf_counter() - total_start
    solve_start = time.perf_counter()
    trim = None
    if "target_cl" in cond:
        trim = solver.solveCL(ap, CLStar=cond["target_cl"], alpha0=cond["alpha_initial_deg"],
                              tol=request["numerics"]["trim_tolerance"],
                              maxIter=request["numerics"]["trim_max_iterations"],
                              stopOnStall=True, writeSolution=True)
    else:
        solver(ap)
    comm.Barrier()
    solve_seconds = time.perf_counter() - solve_start
    functions = {}
    solver.evalFunctions(ap, functions)
    coefficients = {key: float(functions[f"wing_{name}"]) for key, name in
                    (("CL", "cl"), ("CD", "cd"), ("CMx", "cmx"), ("CMy", "cmy"), ("CMz", "cmz"))}
    coefficients["CM"] = coefficients["CMz"]
    solved_alpha = float(ap.alpha)
    convergence = convergence_report(solver.getResNorms(), tolerance=request["numerics"]["l2_convergence"],
                                     solve_failed=ap.solveFailed, fatal_failed=ap.fatalFail,
                                     coefficients=coefficients, target_cl=cond.get("target_cl"),
                                     trim_tolerance=request["numerics"]["trim_tolerance"],
                                     trim_converged=None if trim is None else bool(trim["converged"]),
                                     solved_alpha_deg=solved_alpha)
    history = solver.getConvergenceHistory()
    iterations_last_solve = history_major_iterations(history)
    trim_major_counts = None if trim is None else [history_major_iterations(item) for item in trim["history"]]
    trim_total_major = (sum(trim_major_counts) if trim_major_counts and all(x is not None for x in trim_major_counts)
                        else None)
    memory = memory_report(comm.allgather(_linux_peak_rss_kib()))
    total_seconds = time.perf_counter() - total_start
    version = _module_provenance(adflow)
    return {"schema_version": 1, "backend": "adflow", "status": "ok" if convergence["converged"] else "not_converged",
            "case_sha256": request["case_sha256"], "identity": identity,
            "coefficients": {k: v if math.isfinite(v) else None for k, v in coefficients.items()},
            "solved_alpha_deg": solved_alpha if math.isfinite(solved_alpha) else None, "convergence": convergence,
            "iterations_last_solve": iterations_last_solve,
            "iterations_kind": "major; history row count excluding initial row",
            "trim_iterations": None if trim is None else int(trim["iterations"]),
            "trim_total_major_iterations": trim_total_major,
            "memory": memory,
            "timing": {"setup_seconds": setup_seconds, "solve_seconds": solve_seconds,
                       "total_seconds": total_seconds, "mpi_ranks": comm.size,
                       "allocated_rank_hours": total_seconds * comm.size / 3600,
                       "scope": "total: imports+solver setup+solve+surface output+coefficients; excludes mesh generation, input hashing and process launch"},
            "solver": {"name": "ADflow", **version, "options": opts,
                       "container_digest": os.environ.get("MDO_CFD_IMAGE_DIGEST"),
                       "api_audit_revision": ADFLOW_API_AUDIT_REVISION,
                       "aerotransformer_recipe_revision": AEROTRANSFORMER_REVISION},
            "host": {"hostname": platform.node(), "platform": platform.platform(),
                     "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
            "provenance": request["provenance"], "surface_output_directory": str(output_dir),
            "comparison_note": "A finite solver coefficient is not proof of matching ML surface integration or a converged result."}


def tutorial_request(mesh_path: Path) -> dict:
    """Known public wing mesh, explicit illustrative flow case (not CRMpert)."""
    if sha256_file(mesh_path) != TUTORIAL_MESH_SHA256:
        raise ValueError("Tutorial mesh SHA256 differs from pinned MACH-Aero L3 asset")
    return {"schema_version": 1, "problem_id": "mach-aero-wing-L3-smoke-v1", "geometry_id": "mach-aero-tutorial-wing",
            "mesh_path": str(mesh_path.resolve()),
            "condition": {"mach": 0.8, "alpha_deg": 1.5, "reynolds": 20_000_000,
                          "reynolds_length_m": 1.0, "temperature_k": 300.0},
            "reference": {"area_m2": 45.5, "chord_m": 3.25, "moment_center_m": [0.0, 0.0, 0.0]},
            "numerics": {"preset": "tutorial"},
            "provenance": {"source": f"https://github.com/mdolab/MACH-Aero/tree/{MACH_AERO_REVISION}",
                           "purpose": "CFD installation/CPU timing smoke case only; not CRMpert and not exact official altitude-based flow case"}}
