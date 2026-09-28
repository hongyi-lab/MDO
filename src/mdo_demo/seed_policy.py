"""Frozen, bounded fixed-seed policy used by every aerodynamic route.

This module only validates files and contracts; it never launches a solver.
The manifest contains project-relative paths, not a machine or SSH address.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

from .io import read_json, sha256_file
from .matched_cfd import canonical_hash, load_bundle

SCHEMA = "aerostructural_fixed_zero_seed_v3"
POLICY = {"preset": "robust_rans_ank_polish", "max_cycles": 1200,
          "l2_convergence": 1e-10, "mpi_ranks": 8}
FILE_KEYS = {"source_result", "source_request", "checkpoint", "grid_audit",
             "seed_cost_manifest", "initial_surface_export", "initial_export_metadata"}


def require_sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Require a complete {name} SHA256")
    return value


def fixed_seed_policy(cfg):
    """Return the immutable initialization contract, or None for legacy runs."""
    init = cfg.get("cfd_initialization")
    if init is None:
        if cfg.get("cfd_solver", {}).get("preset") == POLICY["preset"]:
            raise ValueError("ANK-only optimization requires a frozen fixed-seed manifest")
        return None
    required = {"schema", "manifest_sha256", "max_continuations", "continuation_max_cycles",
                "shared_cost_accounting"}
    if set(init) != required or init.get("schema") != SCHEMA:
        raise ValueError("Unsupported fixed-seed initialization contract")
    require_sha(init["manifest_sha256"], "initialization manifest")
    for name, expected in (("max_continuations", 1), ("continuation_max_cycles", 1200)):
        if type(init[name]) is not int or init[name] != expected:
            raise ValueError("Only one bounded 1200-cycle continuation is supported")
    if init["shared_cost_accounting"] != "once_per_route_including_verification":
        raise ValueError("Seed preparation must be charged once per complete route")
    if cfg.get("cfd_solver") != POLICY:
        raise ValueError("Fixed seed requires the reviewed ANK-only/1200/1e-10/eight-rank policy")
    alpha = cfg["design"]["alpha_deg"]
    if (alpha["initial"] != 0. or alpha["lower"] < -2. or alpha["upper"] > 2.):
        raise ValueError("Fixed-seed pilot must start at zero within [-2,2] degrees")
    return dict(init)


def project_file(project, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Manifest paths must be project-relative POSIX paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts):
        raise ValueError("Manifest paths cannot escape the project")
    resolved = Path(project).resolve().joinpath(*path.parts).resolve()
    resolved.relative_to(Path(project).resolve())
    return resolved


def load_seed_manifest(project, path, cfg):
    init = fixed_seed_policy(cfg)
    if init is None:
        raise ValueError("No fixed-seed policy in this protocol")
    path = Path(path).resolve()
    path.relative_to(Path(project).resolve())
    if sha256_file(path) != init["manifest_sha256"]:
        raise ValueError("Initialization manifest differs from the frozen protocol")
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA or set(manifest.get("files", {})) != FILE_KEYS:
        raise ValueError("Require the complete qualified-zero manifest")
    files = {}
    for key, item in manifest["files"].items():
        if set(item) != {"path", "sha256"}:
            raise ValueError("Each seed file requires its path and hash")
        file = project_file(project, item["path"])
        if sha256_file(file) != require_sha(item["sha256"], key):
            raise ValueError(f"Immutable seed file changed: {key}")
        files[key] = file
    source = read_json(files["source_result"])
    request = load_bundle(files["source_request"])
    if (source.get("case_sha256") != request["case_sha256"]
            or source.get("bundle_identity") != request["identity"]
            or request["identity"]["condition"]["alpha_deg"] != 0.):
        raise ValueError("Initial cache does not belong to the qualified zero-degree request")
    conv = source.get("convergence", {})
    r = conv.get("residual_relative_to_freestream")
    if (source.get("status") != "ok" or conv.get("converged") is not True
            or isinstance(r, bool) or not isinstance(r, (float, int)) or not math.isfinite(r)
            or not 0 <= r <= 1e-10 or conv.get("required_l2_convergence") != 1e-10
            or source.get("restart_audit", {}).get("accepted") is not True):
        raise ValueError("Initial cache is not a qualified CFD solution")
    # Full restart/volume audits are repeated in the original image namespace by
    # validate_seed_source. The host never rewrites paths in a saved result.
    cost = read_json(files["seed_cost_manifest"])
    seconds = cost.get("total_host_seed_preparation_seconds")
    if (cost.get("schema") != "adflow_shared_seed_host_cost_v1"
            or cost.get("seed_result_content_sha256") != canonical_hash(source)
            or cost.get("checkpoint_sha256") != manifest["files"]["checkpoint"]["sha256"]
            or isinstance(seconds, bool) or not isinstance(seconds, (float, int))
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Shared preparation ledger is not tied to this seed")
    ledger = {"manifest_sha256": init["manifest_sha256"],
              "seed_result_content_sha256": canonical_hash(source),
              "checkpoint_sha256": cost["checkpoint_sha256"],
              "shared_seed_preparation_seconds": float(seconds),
              "accounting": init["shared_cost_accounting"]}
    return manifest, files, ledger


def validate_target_policy(cfg, target):
    if fixed_seed_policy(cfg) is None:
        raise ValueError("Generic target requires the frozen shared-seed policy")
    alpha = target["identity"]["condition"]["alpha_deg"]
    limits = cfg["design"]["alpha_deg"]
    if (isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha)
            or not limits["lower"] <= alpha <= limits["upper"]):
        raise ValueError("Target angle is outside the frozen optimization domain")


def continuation_eligible(result):
    """Only a completed, audited, finite exhausted first target can continue."""
    audit = result.get("seed_audit", {})
    conv = result.get("convergence", {})
    count = result.get("internal_iterations_last_solve")
    return (result.get("status") == "seed_initialization_unqualified"
            and audit.get("passed") is True and audit.get("accepted") is False
            and conv.get("converged") is False and conv.get("fatal_failed") is False
            and conv.get("finite_coefficients") is True and conv.get("finite_solved_alpha") is True
            and conv.get("trim_pass") is True and type(count) is int and count >= 1200
            and result.get("configured_max_internal_iterations") == 1200
            and "restart" not in result.get("provenance", {}))
