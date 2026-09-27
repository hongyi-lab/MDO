"""Validate one same-case ADflow volume checkpoint and audit its measured cost.

This module never changes residual norms, starts a solver, or treats successful
restart loading as convergence. Paths must resolve in the source run's recorded
namespace (normally inside the existing rootless CFD image).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

from .cfd import validate_request
from .io import read_json, sha256_file
from .matched_cfd import canonical_hash, load_bundle

TOLERANCE = 1e-10
DENOMINATOR_RTOL = 1e-8
CHECKPOINT_START_RTOL = .01
RESTART_SCHEMA = "adflow_same_case_volume_restart_v1"


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _norms(result):
    conv = result["convergence"]
    initial = _finite(conv["residual_initial"], "freestream residual", positive=True)
    start = _finite(conv["residual_start"], "starting residual")
    final = _finite(conv["residual_final"], "final residual")
    reported = _finite(conv["residual_relative_to_freestream"], "reported residual ratio")
    if not math.isclose(reported, final / initial, rel_tol=1e-12, abs_tol=0.):
        raise ValueError("Recorded ratio is not final / freestream residual")
    if conv.get("required_l2_convergence") != TOLERANCE:
        raise ValueError("Checkpoint refinement must preserve the 1e-10 criterion")
    return initial, start, final


def _check_coefficients(result):
    groups = [result.get("coefficients", {})]
    groups.extend(result.get("group_coefficients", {}).values())
    if not groups[0]:
        raise ValueError("A source/result must contain finite aerodynamic coefficients")
    for group in groups:
        for value in group.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Nonfinite aerodynamic coefficient")


def _volume_hash(result):
    expected = canonical_hash(result["identity"])
    actual = result.get("volume_case_sha256", result.get("case_sha256"))
    if _sha(actual, "volume case hash") != expected:
        raise ValueError("Volume case identity hash mismatch")
    return expected


def _stage_cost(result):
    timing = result["timing"]
    total = _finite(timing["total_seconds"], "solver total seconds", positive=True)
    ranks = timing["mpi_ranks"]
    if type(ranks) is not int or ranks != 8:
        raise ValueError("Checkpoint refinement must retain eight MPI ranks")
    # Prefer the broader existing mesh-preparation + CFD clock, when present.
    if "refinement_stage_seconds" in timing:
        elapsed = _finite(timing["refinement_stage_seconds"], "refinement stage seconds", positive=True)
        if elapsed + 1e-8 < total:
            raise ValueError("Refinement stage clock is smaller than solver total")
        scope = "refinement_stage_seconds; checkpoint validation, MPI synchronization, solver setup/solve/output; excludes launcher and final JSON serialization"
    elif "live_mesh_and_cfd_seconds" in timing:
        elapsed = _finite(timing["live_mesh_and_cfd_seconds"], "mesh + CFD stage seconds", positive=True)
        if elapsed + 1e-8 < total:
            raise ValueError("Mesh + CFD clock is smaller than solver total")
        scope = "live_mesh_and_cfd_seconds; includes recorded mesh preparation and CFD"
    else:
        elapsed = total
        scope = "total_seconds; solver imports/setup/solve/output; no unmeasured launch or mesh cost inferred"
    if "continuation_chain_seconds" in timing:
        raise ValueError("Do not nest an existing continuation clock as a fresh source stage")
    return {"seconds": elapsed, "mpi_ranks": ranks,
            "allocated_rank_hours": elapsed * ranks / 3600., "scope": scope}


def validate_restart_source(source_result_path, bundle_dir, checkpoint_path,
                            expected_checkpoint_sha256):
    """Return ``(normalized_request, provenance)`` after read-only validation.

    The request initially retains the source numerical preset/budget. The caller
    must explicitly set its reviewed refinement preset and additional budget.
    ``request['provenance']['restart']`` is already populated for audit_restart.
    This validates association to an ADflow-written volume output, not the CGNS
    field contents; inspect the actual volume state separately before execution.
    Only one checkpoint-refinement stage is supported by this bounded adapter.
    """
    source_path = Path(source_result_path).resolve()
    bundle_path = Path(bundle_dir).resolve() / "request.json"
    checkpoint = Path(checkpoint_path).resolve()
    expected_sha = _sha(expected_checkpoint_sha256, "expected checkpoint hash")
    source = read_json(source_path)
    if source.get("backend") != "adflow" or source.get("status") not in {"ok", "not_converged"}:
        raise ValueError("Restart source must be a completed ADflow result")
    if source.get("provenance", {}).get("restart") is not None:
        raise ValueError("This bounded adapter allows one checkpoint refinement, not a retry chain")
    if source["convergence"].get("fatal_failed") is not False:
        raise ValueError("A fatal solver failure is not a valid checkpoint source")
    r0, rstart, rfinal = _norms(source)
    _check_coefficients(source)
    source_cost = _stage_cost(source)
    manifest = load_bundle(bundle_path)
    if (source.get("case_sha256") != manifest["case_sha256"] or
            source.get("bundle_identity") != manifest["identity"]):
        raise ValueError("Source case/bundle identity mismatch")
    identity = source["identity"]
    bundle = manifest["identity"]
    if (identity.get("geometry_sha256") != bundle["surface_sha256"] or
            identity.get("geometry_id") != bundle["case_name"] or
            identity.get("condition") != bundle["condition"] or
            identity.get("reference") != bundle["reference"]):
        raise ValueError("Source physical condition/reference/geometry differs from bundle")
    options = source["solver"]["options"]
    if (options.get("writeVolumeSolution") is not True or
            options.get("solutionPrecision") != "double"):
        raise ValueError("Source must have saved a double-precision volume solution")
    output = Path(source["surface_output_directory"]).resolve()
    if (output != (source_path.parent / "solver_surface").resolve() or
            Path(options["outputDirectory"]).resolve() != output):
        raise ValueError("Source recorded output directory does not belong to this result")
    if (checkpoint.parent != output or
            re.fullmatch(r"wing(?:_\d+)?_vol\.cgns", checkpoint.name) is None):
        raise ValueError("Checkpoint must be this run's wing volume output, not a mesh or surface")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("Checkpoint missing or empty")
    if sha256_file(checkpoint) != expected_sha:
        raise ValueError("Checkpoint hash mismatch")
    mesh = Path(options["gridFile"]).resolve()
    if checkpoint == mesh:
        raise ValueError("Volume mesh alone is not a checkpoint")
    if sha256_file(mesh) != identity["volume_mesh_sha256"]:
        raise ValueError("Original volume mesh hash mismatch")
    preset = {1e-5: "robust_rans", 1e-7: "robust_rans_late_nk"}.get(options.get("NKSwitchTol"))
    if preset is None or options.get("equationType") != "RANS" or options.get("turbulenceModel") != "SA":
        raise ValueError("Unsupported source solver physics/preset")
    raw = {"schema_version": 1, "problem_id": identity["problem_id"],
           "geometry_id": identity["geometry_id"], "geometry_sha256": identity["geometry_sha256"],
           "mesh_path": str(mesh), "condition": identity["condition"], "reference": identity["reference"],
           "numerics": {"preset": preset, "l2_convergence": TOLERANCE,
                        "max_cycles": source["configured_max_internal_iterations"],
                        "coarse_cycles": options["nCyclesCoarse"]},
           "provenance": copy.deepcopy(source.get("provenance", {}))}
    normalized = validate_request(raw)
    if normalized["identity"] != identity or normalized["case_sha256"] != _volume_hash(source):
        raise ValueError("Reconstructed request does not preserve source physical identity")
    provenance = {"schema": RESTART_SCHEMA, "source_result_path": str(source_path),
                  "source_result_file_sha256": sha256_file(source_path),
                  "source_result_content_sha256": canonical_hash(source),
                  "source_bundle_case_sha256": manifest["case_sha256"],
                  "source_volume_case_sha256": normalized["case_sha256"],
                  "original_volume_mesh_sha256": identity["volume_mesh_sha256"],
                  "checkpoint_path": str(checkpoint), "checkpoint_sha256": expected_sha,
                  "checkpoint_bytes": checkpoint.stat().st_size,
                  "source_solver_options_sha256": canonical_hash(options),
                  "source_residual_initial": r0, "source_residual_start": rstart,
                  "source_residual_final": rfinal, "source_stage_cost": source_cost,
                  "restart_kind": "same-case volume checkpoint; new solver initialization",
                  "checkpoint_field_content_audit": "required separately; filename/hash association alone does not inspect CGNS fields",
                  "normalization": "getResNorms final/initial; never final/restart-start or injected residual norms"}
    normalized["provenance"]["restart"] = copy.deepcopy(provenance)
    return normalized, provenance


def audit_restart(source_result, result):
    """Audit identity, unmodified residual normalization and source+stage cost.

    Returns a report with separate ``passed`` (audit) and ``accepted`` (solver
    convergence) flags. Raises on invalid evidence; never mutates either result.
    ``result`` can be raw run_adflow output or a matched wrapper preserving its
    volume_case_sha256. It must carry the validated restart provenance.
    """
    source = source_result
    if source.get("identity") != result.get("identity") or _volume_hash(source) != _volume_hash(result):
        raise ValueError("Checkpoint refinement changed the physical case or mesh")
    if source.get("function_groups") != result.get("function_groups"):
        raise ValueError("Checkpoint refinement changed aerodynamic integration families")
    prov = result.get("provenance", {}).get("restart", {})
    if prov.get("schema") != RESTART_SCHEMA or prov.get("source_result_content_sha256") != canonical_hash(source):
        raise ValueError("Restart provenance does not identify the exact source result")
    if prov.get("source_volume_case_sha256") != _volume_hash(source):
        raise ValueError("Restart provenance has a different volume case")
    if prov.get("original_volume_mesh_sha256") != source["identity"]["volume_mesh_sha256"]:
        raise ValueError("Restart provenance has a different mesh")
    _sha(prov.get("checkpoint_sha256"), "checkpoint provenance hash")
    options = result["solver"]["options"]
    if str(Path(options.get("restartFile", "")).resolve()) != prov.get("checkpoint_path"):
        raise ValueError("Actual solver restartFile differs from the validated checkpoint")
    if (options.get("L2Convergence") != TOLERANCE or
            options.get("L2ConvergenceRel") != 1e-16):
        raise ValueError("Actual solver options changed a convergence criterion")
    source_r0, _, source_final = _norms(source)
    r0, start, final = _norms(result)
    if prov.get("source_residual_initial") != source_r0:
        raise ValueError("Provenance freestream denominator differs from source")
    if not math.isclose(r0, source_r0, rel_tol=DENOMINATOR_RTOL, abs_tol=0.):
        raise ValueError("Restart changed the freestream residual denominator")
    # A near-converged checkpoint is reconstructed from primitive state fields;
    # derivative and boundary recomputation can magnify roundoff in its tiny
    # residual. This predeclared 1% check is ONLY a loading-consistency check.
    # Neither final residual threshold is relaxed by it.
    if not math.isclose(start, source_final, rel_tol=CHECKPOINT_START_RTOL, abs_tol=0.):
        raise ValueError("Starting residual is inconsistent with the source checkpoint")
    _check_coefficients(result)
    source_cost, stage_cost = _stage_cost(source), _stage_cost(result)
    if prov.get("source_stage_cost") != source_cost:
        raise ValueError("Provenance source cost differs from recorded source stage")
    conv = result["convergence"]
    accepted = (result.get("status") == "ok" and conv.get("converged") is True
                and conv.get("solve_failed") is False and conv.get("fatal_failed") is False
                and conv.get("finite_coefficients") is True and conv.get("finite_solved_alpha") is True
                and conv.get("trim_pass") is True and final / r0 <= TOLERANCE
                and final / source_r0 <= TOLERANCE)
    return {"schema": RESTART_SCHEMA, "passed": True, "accepted": accepted,
            "source_result_content_sha256": canonical_hash(source),
            "checkpoint_sha256": prov["checkpoint_sha256"],
            "freestream_denominator_relative_difference": abs(r0 / source_r0 - 1.),
            "freestream_denominator_rtol": DENOMINATOR_RTOL,
            "source_freestream_residual": source_r0, "current_freestream_residual": r0,
            "current_starting_residual": start, "final_residual": final,
            "source_final_residual": source_final,
            "checkpoint_start_relative_difference": abs(start / source_final - 1.) if source_final else 0.,
            "checkpoint_start_rtol": CHECKPOINT_START_RTOL,
            "final_relative_to_source_freestream": final / source_r0,
            "final_relative_to_current_freestream": final / r0,
            "required_l2_convergence": TOLERANCE,
            "source_stage_cost": source_cost, "refinement_stage_cost": stage_cost,
            "continuation_chain_seconds": source_cost["seconds"] + stage_cost["seconds"],
            "continuation_chain_allocated_rank_hours": source_cost["allocated_rank_hours"] + stage_cost["allocated_rank_hours"],
            "cost_scope": "Measured source stage plus this refinement; neither checkpoint generation nor source solve is free. Earlier independent solver-development attempts and unmeasured process launch costs remain separately reported.",
            "mdo_result": False}
