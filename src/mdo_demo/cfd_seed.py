"""Explicit fixed-zero-alpha CFD initialization, distinct from same-case restart.

No solver runs here. Each target retains its own freestream residual denominator.
The qualified seed's generation cost is a shared ledger, never a hidden free
solve or a cost charged again on every aerodynamic call.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

from .cfd import validate_request
from .cfd_restart import (_check_coefficients, _finite, _norms, _sha,
                          _volume_hash, audit_restart)
from .io import read_json, sha256_file
from .matched_cfd import canonical_hash, load_bundle

SEED_SCHEMA = "adflow_fixed_zero_alpha_seed_v1"
HOST_COST_SCHEMA = "adflow_shared_seed_host_cost_v1"
TOLERANCE = 1e-10
ADDITIONAL_CYCLES = 1200
REQUIRED_FIELDS = {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure", "TurbulentSANuTilde"}


def _fixed_bundle(identity):
    value = copy.deepcopy(identity)
    for key in ("native_input_sha256", "fm_input_sha256"):
        value.pop(key, None)
    value["condition"].pop("alpha_deg", None)
    return value


def _fixed_volume(identity):
    value = copy.deepcopy(identity)
    value["condition"].pop("alpha_deg", None)
    return value


def _accepted(result):
    initial, _, final = _norms(result)
    _check_coefficients(result)
    conv = result["convergence"]
    alpha = result["identity"]["condition"]["alpha_deg"]
    solved = result.get("solved_alpha_deg")
    if (isinstance(solved, bool) or not isinstance(solved, (int, float))
            or not math.isfinite(solved) or solved != alpha):
        raise ValueError("Actual solved alpha does not match the target physical condition")
    return (result.get("backend") == "adflow" and result.get("status") == "ok"
            and conv.get("converged") is True and conv.get("solve_failed") is False
            and conv.get("fatal_failed") is False and conv.get("finite_coefficients") is True
            and conv.get("finite_solved_alpha") is True and conv.get("trim_pass") is True
            and final / initial <= TOLERANCE)


def _qualified_seed(source):
    if source["identity"]["condition"].get("alpha_deg") != 0. or not _accepted(source):
        raise ValueError("Fixed seed must be the qualified zero-degree ADflow solution")
    recorded = source.get("restart_audit", {})
    if recorded.get("passed") is not True or recorded.get("accepted") is not True:
        raise ValueError("Seed requires its accepted same-case restart audit")
    previous = source.get("provenance", {}).get("restart", {})
    parent_path = Path(previous["source_result_path"])
    if sha256_file(parent_path) != previous.get("source_result_file_sha256"):
        raise ValueError("Seed parent result file changed after the accepted restart")
    parent = read_json(parent_path)
    audited = audit_restart(parent, source)
    if audited != recorded or not audited["accepted"]:
        raise ValueError("Seed's stored restart audit cannot be reproduced")
    return audited


def _grid_audit(path, mesh_sha, checkpoint_sha):
    audit = read_json(path)
    if (audit.get("passed") is not True or audit.get("status") != "grid_and_checkpoint_fields_verified"
            or audit.get("mesh_sha256") != mesh_sha or audit.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("Seed grid/field audit does not identify this mesh and checkpoint")
    tolerance = _finite(audit["coordinate_absolute_tolerance_m"], "coordinate tolerance", positive=True)
    maximum = _finite(audit["max_coordinate_difference_m"], "maximum coordinate difference")
    if tolerance > 1e-10 or maximum > tolerance or audit.get("coordinate_relative_tolerance") != 0.:
        raise ValueError("Seed grid audit used a different geometry criterion")
    zones = audit.get("zones", [])
    if not zones or len(zones) != audit.get("zone_count"):
        raise ValueError("Seed grid audit lacks its complete zone records")
    count = 0
    for zone in zones:
        solution = zone["solution"]
        fields = solution.get("required_interior_fields", {})
        if solution.get("location") != "CellCenter" or set(fields) != REQUIRED_FIELDS:
            raise ValueError("Seed checkpoint audit lacks required cell-centred restart fields")
        for name, field in fields.items():
            if field.get("datatype") != 4 or field.get("finite") is not True:
                raise ValueError("Seed checkpoint requires finite RealDouble state fields")
            low, high = field.get("minimum"), field.get("maximum")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (low, high)):
                raise ValueError("Invalid restart state-field extrema")
            if high < low or (name in {"Density", "Pressure"} and low <= 0):
                raise ValueError("Invalid restart state-field range")
        cells = solution.get("interior_cell_count")
        if type(cells) is not int or cells <= 0:
            raise ValueError("Invalid checkpoint interior cell count")
        count += cells
    if count != audit.get("total_interior_cells"):
        raise ValueError("Seed checkpoint field audit has inconsistent coverage")
    return audit


def validate_seed_source(source_result_path, source_request_path, target_request_path,
                         checkpoint_path, expected_sha, grid_audit_path, *,
                         seed_cost_manifest_path=None):
    """Return a target request plus fixed-seed provenance, without solving.

    A host cost manifest is optional. When supplied it must contain schema
    ``adflow_shared_seed_host_cost_v1``, ``seed_result_content_sha256``,
    ``checkpoint_sha256`` and ``total_host_seed_preparation_seconds``. Its clock
    is reported separately from the seed's actual CFD continuation clock.
    """
    source_path, source_request_path, target_request_path = map(Path, (
        source_result_path, source_request_path, target_request_path))
    source = read_json(source_path)
    accepted_restart = _qualified_seed(source)
    source_bundle, target_bundle = load_bundle(source_request_path), load_bundle(target_request_path)
    if (source["case_sha256"] != source_bundle["case_sha256"]
            or source.get("bundle_identity") != source_bundle["identity"]):
        raise ValueError("Seed result and source bundle disagree")
    original, target = source_bundle["identity"], target_bundle["identity"]
    if _fixed_bundle(original) != _fixed_bundle(target):
        raise ValueError("Only alpha may change between fixed-seed source and target bundles")
    source_native = read_json(source_request_path.resolve().parent / source_bundle["native_input_path"])
    target_native = read_json(target_request_path.resolve().parent / target_bundle["native_input_path"])
    if source_native.get("geometry") != target_native.get("geometry"):
        raise ValueError("Native geometry parameters changed between seed and target")
    alpha = target["condition"].get("alpha_deg")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not -2. <= alpha <= 2.:
        raise ValueError("Target alpha must lie within the frozen [-2, 2] degree domain")
    identity, options = source["identity"], source["solver"]["options"]
    if (identity["condition"] != original["condition"] or identity["reference"] != original["reference"]
            or identity["geometry_sha256"] != original["surface_sha256"]
            or identity["geometry_id"] != original["case_name"]):
        raise ValueError("Seed physical identity disagrees with its source bundle")
    checkpoint, source_path = Path(checkpoint_path).resolve(), source_path.resolve()
    checkpoint_sha = _sha(expected_sha, "expected seed checkpoint hash")
    output = Path(source["surface_output_directory"]).resolve()
    if (output != (source_path.parent / "solver_surface").resolve()
            or output != Path(options["outputDirectory"]).resolve()
            or checkpoint.parent != output
            or re.fullmatch(r"wing(?:_\d+)?_vol\.cgns", checkpoint.name) is None):
        raise ValueError("Seed checkpoint must be the qualified run's own volume output")
    if (options.get("writeVolumeSolution") is not True or options.get("solutionPrecision") != "double"
            or not checkpoint.is_file() or checkpoint.stat().st_size == 0):
        raise ValueError("Qualified seed requires its saved double-precision volume solution")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise ValueError("Seed checkpoint hash mismatch")
    mesh = Path(options["gridFile"]).resolve()
    if checkpoint == mesh or sha256_file(mesh) != identity["volume_mesh_sha256"]:
        raise ValueError("Original seed volume mesh hash mismatch")
    grid = _grid_audit(grid_audit_path, identity["volume_mesh_sha256"], checkpoint_sha)
    normalized = validate_request({"schema_version": 1, "problem_id": identity["problem_id"],
        "geometry_id": identity["geometry_id"], "geometry_sha256": identity["geometry_sha256"],
        "mesh_path": str(mesh), "condition": target["condition"], "reference": target["reference"],
        "numerics": {"preset": "robust_rans_ank_polish", "max_cycles": ADDITIONAL_CYCLES,
                     "l2_convergence": TOLERANCE, "coarse_cycles": options["nCyclesCoarse"]},
        "provenance": {"bundle_case_sha256": target_bundle["case_sha256"],
                       "bundle_identity": target}})
    _volume_hash(source)
    if _fixed_volume(normalized["identity"]) != _fixed_volume(identity):
        raise ValueError("Target CFD request changed physics beyond alpha")
    source_hash = canonical_hash(source)
    ledger = {"seed_result_content_sha256": source_hash, "checkpoint_sha256": checkpoint_sha,
              "cfd_continuation_seconds": _finite(accepted_restart["continuation_chain_seconds"], "seed preparation CFD cost", positive=True),
              "cfd_continuation_allocated_rank_hours": _finite(accepted_restart["continuation_chain_allocated_rank_hours"], "seed preparation rank hours", positive=True),
              "host_preparation_seconds": None, "host_manifest_sha256": None,
              "counting": "Shared setup ledger; aggregate once per frozen seed, never add to every online call"}
    if seed_cost_manifest_path is not None:
        host = read_json(seed_cost_manifest_path)
        if (host.get("schema") != HOST_COST_SCHEMA or host.get("seed_result_content_sha256") != source_hash
                or host.get("checkpoint_sha256") != checkpoint_sha):
            raise ValueError("Host seed cost manifest belongs to different evidence")
        seconds = _finite(host["total_host_seed_preparation_seconds"], "host seed preparation time", positive=True)
        if seconds < ledger["cfd_continuation_seconds"]:
            raise ValueError("Host preparation clock cannot omit the source CFD continuation cost")
        ledger.update(host_preparation_seconds=seconds, host_manifest_sha256=sha256_file(Path(seed_cost_manifest_path)))
    ledger["id"] = canonical_hash({k: ledger[k] for k in ("seed_result_content_sha256", "checkpoint_sha256")})
    provenance = {"schema": SEED_SCHEMA, "source_result_path": str(source_path),
        "source_result_file_sha256": sha256_file(source_path), "source_result_content_sha256": source_hash,
        "source_bundle_case_sha256": source_bundle["case_sha256"], "target_bundle_case_sha256": target_bundle["case_sha256"],
        "source_volume_case_sha256": _volume_hash(source), "target_volume_case_sha256": normalized["case_sha256"],
        "target_identity": normalized["identity"], "seed_alpha_deg": 0., "target_alpha_deg": alpha,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "original_volume_mesh_sha256": identity["volume_mesh_sha256"],
        "grid_audit_sha256": sha256_file(Path(grid_audit_path)), "grid_audit_interior_cells": grid["total_interior_cells"],
        "shared_seed_preparation": ledger,
        "shared_seed_ledger_sha256": canonical_hash(ledger),
        "mode": "immutable zero-alpha seed; target alpha only changes; not same-case residual continuation",
        "normalization": "new target's getResNorms final/initial; seed freestream and starting norms are not target acceptance denominators"}
    normalized["provenance"]["seed_initialization"] = copy.deepcopy(provenance)
    return normalized, provenance


def audit_seed(source_result, new_result):
    """Audit a target using its own residual denominator and separate seed cost."""
    source, result = source_result, new_result
    seed_audit = _qualified_seed(source)
    prov = result.get("provenance", {}).get("seed_initialization", {})
    if "restart" in result.get("provenance", {}):
        raise ValueError("Cross-alpha initialization cannot inherit same-case restart provenance")
    if (prov.get("schema") != SEED_SCHEMA
            or prov.get("source_result_content_sha256") != canonical_hash(source)
            or prov.get("source_volume_case_sha256") != _volume_hash(source)):
        raise ValueError("Target initialization does not identify the immutable qualified seed")
    if (prov.get("target_identity") != result.get("identity")
            or prov.get("target_volume_case_sha256") != _volume_hash(result)
            or _fixed_volume(source["identity"]) != _fixed_volume(result["identity"])):
        raise ValueError("Initialized target physical identity differs from its validated request")
    target_case = prov.get("target_bundle_case_sha256")
    if result.get("case_sha256") not in {_volume_hash(result), target_case}:
        raise ValueError("Target result carries another bundle case")
    if result.get("function_groups") != source.get("function_groups"):
        raise ValueError("Target aerodynamic integration families changed")
    options = result["solver"]["options"]
    if str(Path(options.get("restartFile", "")).resolve()) != prov.get("checkpoint_path"):
        raise ValueError("Solver did not use the declared immutable seed checkpoint")
    _sha(prov.get("checkpoint_sha256"), "seed checkpoint provenance hash")
    expected = {"useANKSolver": True, "useNKSolver": False, "ANKCoupledSwitchTol": 1e-16,
                "ANKSecondOrdSwitchTol": 1e-3, "nSubiterTurb": 10, "MGCycle": "sg",
                "equationType": "RANS", "turbulenceModel": "SA", "nCycles": ADDITIONAL_CYCLES,
                "L2Convergence": TOLERANCE, "L2ConvergenceRel": 1e-16,
                "infChangeCorrection": False}
    if any(options.get(key) != value for key, value in expected.items()):
        raise ValueError("Target used a different fixed-seed numerical policy")
    r0, start, final = _norms(result)
    accepted = _accepted(result)
    timing = result["timing"]
    if type(timing.get("mpi_ranks")) is not int or timing["mpi_ranks"] != 8:
        raise ValueError("Fixed-seed target must retain eight MPI ranks")
    total = _finite(timing["total_seconds"], "solver total seconds", positive=True)
    seconds = _finite(timing["seed_initialization_stage_seconds"], "online target stage seconds", positive=True)
    if seconds + 1e-8 < total:
        raise ValueError("Online stage clock is smaller than the solver clock")
    ledger = prov.get("shared_seed_preparation", {})
    if prov.get("shared_seed_ledger_sha256") != canonical_hash(ledger):
        raise ValueError("Shared seed preparation ledger changed after validation")
    if (ledger.get("seed_result_content_sha256") != canonical_hash(source)
            or ledger.get("checkpoint_sha256") != prov["checkpoint_sha256"]
            or ledger.get("cfd_continuation_seconds") != seed_audit["continuation_chain_seconds"]
            or ledger.get("cfd_continuation_allocated_rank_hours") != seed_audit["continuation_chain_allocated_rank_hours"]):
        raise ValueError("Shared seed preparation ledger differs from actual source cost")
    expected_id = canonical_hash({k: ledger[k] for k in ("seed_result_content_sha256", "checkpoint_sha256")})
    if ledger.get("id") != expected_id:
        raise ValueError("Shared seed ledger identifier mismatch")
    return {"schema": SEED_SCHEMA, "passed": True, "accepted": accepted,
        "source_result_content_sha256": canonical_hash(source), "target_volume_case_sha256": _volume_hash(result),
        "target_freestream_residual": r0, "target_starting_residual": start, "target_final_residual": final,
        "target_final_relative_to_target_freestream": final / r0, "required_l2_convergence": TOLERANCE,
        "seed_denominator_used_for_acceptance": False, "seed_start_residual_comparison_required": False,
        "online_call_seconds": seconds, "online_call_allocated_rank_hours": seconds * 8 / 3600.,
        "online_timing_scope": "Validation, initialization, solve and output measured by seed_initialization_stage_seconds; excludes separately recorded host launch",
        "shared_seed_preparation": copy.deepcopy(ledger), "shared_seed_cost_added_to_call": False,
        "mdo_result": False}
