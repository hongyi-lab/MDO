#!/usr/bin/env python3
"""Export only independently checked candidate quality and measured wall cost."""
import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdo_demo.io import read_json, write_json
from mdo_demo.aerostructural import VARIABLES, constraint_margins, validate_protocol
from mdo_demo.aero_contract import validate_request
from mdo_demo.matched_cfd import canonical_hash
from mdo_demo.seed_policy import fixed_seed_policy


def finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite recorded number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def require_sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"A complete {name} SHA256 is required")
    return value


def check_physics(checked, cfg, base_geometry):
    """Require recorded numerical evidence, not just a success label."""
    if checked.get("backend") != "adflow" or checked.get("status") != "ok":
        raise ValueError("Verification must use successful ADflow and TACS providers")
    aero = checked.get("aerodynamic_evidence", {})
    if aero.get("backend") != "adflow" or aero.get("reference_eligible") is not True:
        raise ValueError("Verification lacks reference-eligible native ADflow evidence")
    ident = validate_request(aero["request"])
    if ident["geometry_sha256"] != base_geometry or ident["case_id"] != checked.get("aerodynamic_case_sha256"):
        raise ValueError("Verification aerodynamic geometry/case differs from its task")
    if ident["condition"]["alpha_deg"] != checked["design"]["alpha_deg"]:
        raise ValueError("Verified loads do not use the candidate's angle")
    conv = aero.get("solver_convergence", {})
    tolerance = finite(conv.get("required_l2_convergence"), "CFD convergence tolerance", positive=True)
    relative = finite(conv.get("residual_relative_to_freestream"), "CFD residual")
    if (conv.get("converged") is not True or conv.get("solve_failed") is not False
            or conv.get("fatal_failed") is not False or tolerance > 1e-10 or relative > tolerance):
        raise ValueError("Native CFD did not satisfy the frozen convergence requirement")
    if aero.get("integration_audit", {}).get("passed") is not True:
        raise ValueError("Native CFD force/moment reconstruction was not audited")
    require_sha(aero.get("provenance", {}).get("solver_result_content_sha256"), "native CFD result")
    init = fixed_seed_policy(cfg)
    if init is not None:
        receipt = aero.get("fixed_seed_receipt") or {}
        if (receipt.get("schema") != "aerostructural_fixed_seed_receipt_v3"
                or receipt.get("accepted") is not True or receipt.get("field_audit_passed") is not True
                or receipt.get("protocol_sha256") != cfg["protocol_sha256"]
                or receipt.get("initialization_manifest_sha256") != init["manifest_sha256"]
                or receipt.get("case_sha256") != ident["case_id"]
                or receipt.get("alpha_deg") != checked["design"]["alpha_deg"]
                or receipt.get("result_content_sha256") != aero["provenance"]["solver_result_content_sha256"]
                or receipt.get("actual_mpi_ranks") != 8
                or receipt.get("relative_residual") != relative
                or receipt.get("shared_cost_added_to_call") is not False):
            raise ValueError("Verification lacks matching actual shared-seed solver evidence")
        mode = receipt.get("mode")
        if mode == "qualified_zero_cache":
            if ident["condition"]["alpha_deg"] != 0. or receipt.get("continuation_count") != 0:
                raise ValueError("Only zero-degree reference may use the initial cache")
        else:
            expected_count = {"fixed_zero_seed": 0, "fixed_zero_seed_plus_one_continuation": 1}.get(mode)
            if expected_count is None or receipt.get("continuation_count") != expected_count:
                raise ValueError("Verification used an unsupported restart chain")
            options = receipt.get("actual_solver_options", {})
            required = {"useANKSolver": True, "useNKSolver": False, "infChangeCorrection": False,
                        "equationType": "RANS", "turbulenceModel": "SA", "nCycles": 1200,
                        "L2Convergence": 1e-10, "L2ConvergenceRel": 1e-16,
                        "writeVolumeSolution": True, "solutionPrecision": "double",
                        "MGCycle": "sg", "ANKSecondOrdSwitchTol": 1e-3,
                        "ANKCoupledSwitchTol": 1e-16, "nSubiterTurb": 10}
            if any(options.get(k) != v or (type(v) is bool and options.get(k) is not v)
                   for k, v in required.items()):
                raise ValueError("Actual CFD options differ from the common numerical strategy")
    structural = checked["structural"]
    if structural.get("backend") != "TACS" or structural.get("status") != "ok":
        raise ValueError("Verification must contain an actual successful TACS solve")
    sc = structural.get("convergence", {})
    sr = finite(sc.get("relative_residual"), "TACS residual")
    st = finite(sc.get("tolerance"), "TACS tolerance", positive=True)
    if (sc.get("converged") is not True or sr > st or st > 1e-7
            or finite(sc.get("root_displacement_rotation_max"), "root clamp residual") > 1e-10
            or sc.get("solver_reported_converged") is False):
        raise ValueError("TACS did not meet the common structural convergence requirement")
    if structural.get("transfer_audit", {}).get("passed") is not True:
        raise ValueError("Physical load-transfer conservation is unverified")
    for name in ("skin_m", "web_m"):
        if structural.get("thickness", {}).get(name) != checked["design"][name]:
            raise ValueError("Verified TACS thickness differs from the candidate")
    for name in VARIABLES:
        value = checked["design"][name]
        spec = cfg["design"][name]
        if isinstance(value, bool) or not math.isfinite(value) or not spec["lower"] <= value <= spec["upper"]:
            raise ValueError("Verified candidate is outside the frozen design domain")
    if structural.get("material") != cfg["material"]:
        raise ValueError("Verified structural material differs from the protocol")
    margins = constraint_margins(checked["coefficients"], structural, cfg)
    if (checked.get("constraint_margins") != margins or checked.get("minimum_margin") != min(margins.values())
            or checked.get("feasible") is not (min(margins.values()) >= -cfg["optimizer"]["catol"])):
        raise ValueError("Verified feasibility disagrees with the frozen physical constraints")
    return ident


def compare(cfd_run, fm_run, cfd_check, fm_check):
    sources = [cfd_run, fm_run, cfd_check, fm_check]
    hashes = {s.get("protocol_sha256") for s in sources}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("All runs and verifications must use the exact same frozen protocol")
    cfg = validate_protocol(cfd_run["protocol"])
    if cfg["protocol_sha256"] not in hashes:
        raise ValueError("Protocol contents do not match the recorded protocol hash")
    for source in sources:
        if validate_protocol(source["protocol"])["protocol_sha256"] != cfg["protocol_sha256"]:
            raise ValueError("Full frozen protocol contents differ")
    for key in ("base_case_sha256", "base_geometry_sha256"):
        identities = {require_sha(source.get(key), key) for source in sources}
        if len(identities) != 1:
            raise ValueError(f"The paired runs do not share the same {key}")
    for backend, run in (("adflow", cfd_run), ("fm", fm_run)):
        if run.get("backend") != backend:
            raise ValueError("The comparison needs one ADflow run and one FM run")
        if run.get("action") != "optimize" or run.get("status") not in ("completed", "budget_limited"):
            raise ValueError("Only completed or honestly budget-limited optimization runs can be compared")
    rows = []
    init = fixed_seed_policy(cfg)
    shared = None
    if init is not None:
        shared = cfd_run.get("shared_seed_preparation")
        if not isinstance(shared, dict) or any(s.get("shared_seed_preparation") != shared for s in sources):
            raise ValueError("All routes and verifications must share the same preparation ledger")
        if (shared.get("manifest_sha256") != init["manifest_sha256"]
                or shared.get("accounting") != init["shared_cost_accounting"]):
            raise ValueError("Preparation ledger differs from the common protocol")
        require_sha(shared.get("seed_result_content_sha256"), "shared initial result")
        require_sha(shared.get("checkpoint_sha256"), "shared checkpoint")
        finite(shared.get("shared_seed_preparation_seconds"), "shared preparation cost", positive=True)
    verified_requests = []
    structures = []
    for name, run, check in (("ADflow + TACS", cfd_run, cfd_check), ("FM + TACS", fm_run, fm_check)):
        if check.get("action") != "verify" or check.get("common_verification_complete") is not True or check.get("status") != "completed":
            raise ValueError("Both candidates need completed common physical verification")
        checked = check["checked"]
        candidate = run.get("best_feasible") or run.get("best_candidate")
        if candidate is None or candidate.get("status") != "ok":
            raise ValueError("Optimization did not produce a physical candidate")
        if (check.get("candidate_result_content_sha256") != canonical_hash(run)
                or check.get("candidate_sha256") != canonical_hash(candidate)):
            raise ValueError("Verification is not tied to the exact source run and candidate")
        if checked["design"] != candidate["design"]:
            raise ValueError("Verification design does not match the reported candidate")
        if checked.get("aerodynamic_case_sha256") != candidate.get("aerodynamic_case_sha256"):
            raise ValueError("Verification physical case differs from the candidate")
        verified_requests.append(check_physics(checked, cfg, run["base_geometry_sha256"]))
        if shared is not None and checked["aerodynamic_evidence"]["fixed_seed_receipt"].get("shared_preparation") != shared:
            raise ValueError("Final solver receipt identifies another shared preparation ledger")
        structural = checked["structural"]
        structures.extend([candidate["structural"], structural])
        optimization_time = finite(run["total_wall_seconds"], "complete optimization cost", positive=True)
        if finite(run["optimization_wall_seconds"], "optimizer/provider cost", positive=True) > optimization_time:
            raise ValueError("Optimizer/provider time exceeds the reported complete cost")
        if check.get("original_optimization_wall_seconds") != optimization_time:
            raise ValueError("Verification cost is associated with a different optimization run")
        verification_time = finite(check["total_wall_seconds"], "complete verification cost", positive=True)
        if finite(check["verification_wall_seconds"], "verification cost", positive=True) > verification_time:
            raise ValueError("Verification time exceeds the reported complete cost")
        if shared is not None:
            complete_opt = finite(run.get("stage_process_wall_seconds"), "optimization process cost", positive=True)
            complete_check = finite(check.get("stage_process_wall_seconds"), "verification process cost", positive=True)
            if complete_opt < optimization_time or complete_check < verification_time:
                raise ValueError("Complete process timings cannot be narrower than inner timings")
            optimization_time, verification_time = complete_opt, complete_check
        rows.append({"method": name, "verified_wingbox_mass_kg": structural["mass_kg"],
                     "verified_CL": checked["coefficients"]["CL"], "verified_CD": checked["coefficients"]["CD"],
                     "verified_KS_yield": structural["ks_failure"],
                     "verified_tip_displacement_m": structural["tip_displacement_m"],
                     "verified_minimum_constraint_margin": checked["minimum_margin"],
                     "verified_feasible": checked["feasible"],
                     "optimization_wall_seconds": optimization_time,
                     "verification_wall_seconds": verification_time,
                     "total_wall_seconds": optimization_time + verification_time,
                     "optimizer_converged": run["optimizer_converged"],
                     "aerodynamic_calls": run["aerodynamic_call_count"]})
        if shared is not None:
            preparation = shared["shared_seed_preparation_seconds"]
            rows[-1].update(shared_seed_preparation_seconds=preparation,
                           online_optimization_and_verification_seconds=optimization_time + verification_time,
                           total_wall_seconds=preparation + optimization_time + verification_time)
    a = structures[0]
    for b in structures:
        for key in ("common_geometry_source", "mesh_sha256", "material"):
            if not a.get(key) or a.get(key) != b.get(key):
                raise ValueError(f"Optimization/verification structural models differ: {key}")
        if b["common_geometry_source"] != cfd_run["base_geometry_sha256"]:
            raise ValueError("Structural geometry differs from the task's physical surface")
        code_hash = require_sha(b.get("provenance", {}).get("structural_code_sha256"), "structural implementation")
        if code_hash != a["provenance"]["structural_code_sha256"]:
            raise ValueError("Optimization/verification structural implementations differ")
    # Candidate angles can legitimately differ after optimization; all other
    # physical flow, reference and load-domain definitions must remain common.
    def fixed_request(identity):
        return {key: ({k: v for k, v in value.items() if k != "alpha_deg"}
                      if key == "condition" else value)
                for key, value in identity.items() if key != "case_id"}
    if fixed_request(verified_requests[0]) != fixed_request(verified_requests[1]):
        raise ValueError("Final candidates were verified using different flow/reference/coverage definitions")
    return {"schema_version": 1, "protocol_sha256": next(iter(hashes)), "rows": rows,
            "scope": "fixed-planform one-way aero-structural sizing; mainwing-only common loads; not full aircraft or two-way aeroelastic MDO",
            "relative_verified_mass_difference": (rows[1]["verified_wingbox_mass_kg"] / rows[0]["verified_wingbox_mass_kg"] - 1),
            "equal_quality_speedup": None,
            "shared_preparation": shared,
            "note": "Cost and quality are reported together. This bounded integration pilot does not establish matched-quality speedup or global optimality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cfd-run", "fm-run", "cfd-check", "fm-check"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(*(read_json(getattr(args, name)) for name in ("cfd_run", "fm_run", "cfd_check", "fm_check")))
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise FileExistsError("Preserve existing comparison outputs")
    write_json(args.output, result)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])


if __name__ == "__main__":
    main()
