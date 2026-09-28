#!/usr/bin/env python3
"""Finite overnight queue; stop at any failed physics/build gate.

No training. Runs are serial and each solver entry holds the shared compute
lock. Existing results are never resumed or overwritten implicitly.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdo_demo.io import read_json, sha256_file, write_json
from mdo_demo.aerostructural import validate_protocol
from mdo_demo.matched_cfd import load_bundle
from mdo_demo.seed_policy import fixed_seed_policy, load_seed_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "request", "output", "build-receipt", "pretrained-checkpoint", "adapted-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/aerostructural_pilot_v1.json")
    parser.add_argument("--reuse-mesh-result", type=Path)
    parser.add_argument("--initialization-manifest", type=Path)
    args = parser.parse_args()
    protocol = validate_protocol(read_json(args.protocol))
    receipt = read_json(args.build_receipt)
    checks = ("local_tests", "actual_tacs_mechanics", "fm_to_tacs_single_point", "native_field_reintegration")
    fixed = fixed_seed_policy(protocol)
    shared_ledger = None
    if fixed is not None:
        if args.initialization_manifest is None or args.reuse_mesh_result is not None:
            raise ValueError("Frozen initialization manifest required; no adaptive reuse is allowed")
        _, _, shared_ledger = load_seed_manifest(args.project, args.initialization_manifest, protocol)
        checks += ("fixed_seed_manifest", "actual_fixed_seed_target", "actual_zero_cache_to_tacs")
        if receipt.get("initialization_manifest_sha256") != fixed["manifest_sha256"]:
            raise ValueError("Build receipt used a different initialization manifest")
        required_code = {"src/mdo_demo/seed_policy.py", "src/mdo_demo/fixed_seed_runtime.py",
                         "src/mdo_demo/aerostructural.py", "src/mdo_demo/aerostructural_runtime.py",
                         "src/mdo_demo/cfd_seed.py", "src/mdo_demo/cfd_restart.py",
                         "src/mdo_demo/shared_resources.py",
                         "scripts/run_seed_initialization.py", "scripts/run_checkpoint_refinement.py",
                         "scripts/audit_shared_seed.py", "scripts/run_aerostructural.py",
                         "scripts/run_aerostructural_pair.py", "scripts/compare_aerostructural.py"}
        if not required_code.issubset(receipt.get("code_files", {})):
            raise ValueError("Build receipt must bind all new shared-seed runtime and audit files")
    if any(receipt.get("checks", {}).get(key) is not True for key in checks):
        raise ValueError("Complete and record all build checks before launching the experiment")
    if receipt.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Build evidence and experiment protocol differ")
    if receipt.get("base_geometry_sha256") != load_bundle(args.request)["identity"]["surface_sha256"]:
        raise ValueError("Build validation used a different geometry")
    if not receipt.get("code_files"):
        raise ValueError("Build receipt has no tested code hashes")
    for relative, digest in receipt["code_files"].items():
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        if sha256_file(path) != digest:
            raise ValueError(f"Code changed after build checks: {relative}")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "queue.json"
    state = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": protocol["protocol_sha256"], "build_receipt_sha256": sha256_file(args.build_receipt),
             "scope": "bounded one-way common-mainwing aero-structural integration pilot", "stages": [],
             "new_training": False, "max_unique_optimization_cfd_cases": protocol["optimizer"]["max_aero_evaluations"],
             "max_additional_final_verification_cfd_cases": 3}
    state["shared_seed_preparation"] = shared_ledger
    state["project_shared_preparation_count"] = 1 if fixed is not None else 0
    write_json(args.output / "protocol.json", protocol)
    started = time.perf_counter()
    reuse = args.reuse_mesh_result
    for stage, backend, action, checkpoint, candidate in (
        ("adflow_optimization", "adflow", "optimize", None, None),
        ("pretrained_optimization", "fm", "optimize", args.pretrained_checkpoint, None),
        ("adapted_optimization", "fm", "optimize", args.adapted_checkpoint, None),
        ("adflow_verification", "adflow", "verify", None, "adflow_optimization"),
        ("pretrained_verification", "adflow", "verify", None, "pretrained_optimization"),
        ("adapted_verification", "adflow", "verify", None, "adapted_optimization"),
    ):
        entry = {"name": stage, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
        state["stages"].append(entry)
        state["active_stage"] = stage
        write_json(manifest_path, state)
        command = [sys.executable, str(ROOT / "scripts/run_aerostructural.py"), "--project", str(args.project),
                   "--request", str(args.request), "--protocol", str(args.protocol), "--backend", backend,
                   "--action", action, "--output", str(args.output / stage)]
        if checkpoint is not None:
            command += ["--checkpoint", str(checkpoint)]
        if args.initialization_manifest is not None:
            command += ["--initialization-manifest", str(args.initialization_manifest)]
        if reuse is not None:
            command += ["--reuse-mesh-result", str(reuse)]
        if candidate is not None:
            command += ["--candidate-result", str(args.output / candidate / "result.json")]
        stage_start = time.perf_counter()
        with (args.output / (stage + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry.update(exit_code=process.returncode, wall_seconds=time.perf_counter()-stage_start,
                     status="completed" if process.returncode == 0 else "failed",
                     finished_utc=datetime.now(timezone.utc).isoformat())
        if process.returncode:
            state.update(status="blocked", reason=f"Stage {stage} failed; subsequent computations were not started",
                         wall_seconds=time.perf_counter()-started)
            write_json(manifest_path, state)
            return 2
        if fixed is not None:
            result_path = args.output / stage / "result.json"
            stage_result = read_json(result_path)
            stage_result["stage_process_wall_seconds"] = entry["wall_seconds"]
            stage_result["stage_process_timing_scope"] = "complete child process, including imports, lock wait, runtime setup, analysis, optimization or verification and serialization"
            # Record this BEFORE the candidate is read by its later verification.
            # The verification hash therefore binds the final, immutable record.
            write_json(result_path, stage_result)
        if stage == "adflow_optimization" and fixed is None:
            results = sorted((args.output / stage / "analysis").glob("eval_*/aero/cfd/result.json"))
            if results:
                reuse = results[-1]
    try:
        from compare_aerostructural import compare
        import csv
        cfd_run = read_json(args.output / "adflow_optimization/result.json")
        cfd_check = read_json(args.output / "adflow_verification/result.json")
        for model in ("pretrained", "adapted"):
            comparison = compare(cfd_run, read_json(args.output / f"{model}_optimization/result.json"),
                                 cfd_check, read_json(args.output / f"{model}_verification/result.json"))
            write_json(args.output / f"comparison_{model}.json", comparison)
            with (args.output / f"comparison_{model}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(comparison["rows"][0]))
                writer.writeheader()
                writer.writerows(comparison["rows"])
        state.update(status="completed", active_stage=None, wall_seconds=time.perf_counter()-started)
        write_json(manifest_path, state)
        return 0
    except Exception as exc:
        state.update(status="blocked", active_stage="comparison_audit", reason=str(exc),
                     wall_seconds=time.perf_counter()-started)
        write_json(manifest_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
