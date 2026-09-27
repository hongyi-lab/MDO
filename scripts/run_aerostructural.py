#!/usr/bin/env python3
"""Execute one actual common-interface CFD/FM -> TACS pilot under compute.lock."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdo_demo.io import read_json, write_json
from mdo_demo.aerostructural import PilotEvaluator, VARIABLES, validate_protocol
from mdo_demo.aerostructural_runtime import UnifiedRuntime
from mdo_demo.matched_cfd import canonical_hash, load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/aerostructural_pilot_v1.json")
    parser.add_argument("--backend", choices=("fm", "adflow"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=("single", "optimize", "verify"), default="single")
    parser.add_argument("--candidate-result", type=Path)
    parser.add_argument("--reuse-mesh-result", type=Path)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    args = parser.parse_args()
    import fcntl
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = validate_protocol(read_json(args.protocol))
    base = load_bundle(args.request)
    lock_path = args.project / "manifests" / "compute.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        try:
            runtime = UnifiedRuntime(args.project, args.request, cfg, args.output / "runtime",
                        backend=args.backend, checkpoint=args.checkpoint, device=args.device,
                        reuse_mesh_result=args.reuse_mesh_result, mpi_ranks=args.mpi_ranks)
            evaluation = PilotEvaluator(cfg, args.backend, args.output / "analysis", runtime.aero, runtime.structure)
            if args.action == "verify":
                if args.backend != "adflow" or args.candidate_result is None:
                    raise ValueError("Common verification requires ADflow and a recorded candidate result")
                original = read_json(args.candidate_result)
                if original.get("protocol_sha256") != cfg["protocol_sha256"]:
                    raise ValueError("Candidate and verification protocols differ")
                if (original.get("base_case_sha256") != base["case_sha256"] or
                        original.get("base_geometry_sha256") != base["identity"]["surface_sha256"]):
                    raise ValueError("Candidate and verification base physical cases differ")
                candidate = original.get("best_feasible") or original.get("best_candidate")
                if not candidate:
                    raise ValueError("No physical candidate is available for verification")
                x = [candidate["design"][v] / cfg["design"][v]["scale"] for v in VARIABLES]
                checked = evaluation.evaluate(x)
                if checked.get("status") != "ok":
                    raise ValueError("Candidate did not produce a successful physical verification")
                result = {"status": "completed", "action": "verify", "protocol_sha256": cfg["protocol_sha256"],
                          "candidate_result": str(args.candidate_result), "checked": checked,
                          "candidate_result_content_sha256": canonical_hash(original),
                          "candidate_sha256": canonical_hash(candidate),
                          "common_verification_complete": True,
                          "verification_wall_seconds": time.perf_counter() - started,
                          "original_optimization_wall_seconds": original["total_wall_seconds"]}
            else:
                result = evaluation.run(optimize=args.action == "optimize")
                result["action"] = args.action
            result["total_wall_seconds"] = time.perf_counter() - started
            result.update(protocol=cfg, base_case_sha256=base["case_sha256"],
                          base_geometry_sha256=base["identity"]["surface_sha256"])
            result["runtime_setup_wall_seconds"] = runtime.setup_wall_seconds
            result["full_timing_scope"] = "from acquired shared compute lock: runtime probe, model loading/native input, CFD/FM, transfer, TACS, optimizer and output"
            write_json(args.output / "result.json", result)
            print(f"{args.action} {args.backend}: {result['status']}; {args.output / 'result.json'}", flush=True)
            return 0 if result["status"] in ("completed", "budget_limited") else 2
        except Exception as exc:
            write_json(args.output / "result.json", {"status": "failed", "action": args.action,
                       "protocol_sha256": cfg["protocol_sha256"], "error_type": type(exc).__name__, "error": str(exc),
                       "total_wall_seconds": time.perf_counter()-started, "common_verification_complete": False,
                       "mdo_speedup_verified": False})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
