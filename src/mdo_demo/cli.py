"""Small CLI that keeps synthetic fixtures and measured results separate."""

import argparse
import json
import sys

from .io import read_json, write_json, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paper-backed aerodynamic surrogate workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Read-only environment inventory")
    doctor.add_argument("--output", default="results/doctor.json")
    smoke = sub.add_parser("smoke", help="Synthetic interface/report test; not CFD or AI evidence")
    smoke.add_argument("--output", default="results/smoke")
    evaluate = sub.add_parser("evaluate", help="Real AeroTransformer checkpoint vs published CFD data")
    evaluate.add_argument("--data", default="assets/CRMpert")
    evaluate.add_argument("--checkpoint", default="assets/AeroTransformer/ATsurf_S")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--limit", type=int, default=8)
    evaluate.add_argument("--warmup", type=int, default=2)
    evaluate.add_argument("--output", default="results/aerotransformer")
    report = sub.add_parser("report", help="Regenerate HTML from saved evaluation and arrays")
    report.add_argument("evaluation")
    screen = sub.add_parser("screen", help="Finite candidate selection with published CFD-label verification")
    screen.add_argument("--evaluation", required=True)
    screen.add_argument("--mach", type=float, required=True)
    screen.add_argument("--min-cl", type=float, required=True)
    screen.add_argument("--mach-tolerance", type=float, default=1e-6)
    screen.add_argument("--output", default="results/screening.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .doctor import inspect_environment
            result = inspect_environment()
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "smoke":
            from .smoke import run_smoke
            print(f"SYNTHETIC plumbing test only: {run_smoke(args.output)}")
        elif args.command == "evaluate":
            from .aerotransformer import AeroTransformerPredictor
            from .dataset import load_dataset
            from .evaluation import evaluate as run_evaluation
            from .report import write_report
            dataset = load_dataset(args.data)
            predictor = AeroTransformerPredictor(args.checkpoint, device=args.device)
            result = run_evaluation(dataset, predictor, args.output, args.limit, args.warmup)
            print(f"Real checkpoint evaluation: {write_report(result, args.output)}")
            print(f"Cases: {result['case_count']}; median predict: {result['prediction_timing']['median_s']:.6f} s")
            print("Reference: published CFD data. Live CFD speedup has not been measured.")
        elif args.command == "report":
            from pathlib import Path
            from .report import write_report
            print(write_report(read_json(args.evaluation), Path(args.evaluation).parent))
        elif args.command == "screen":
            from .screening import screen_candidates
            result = screen_candidates(read_json(args.evaluation), args.mach, args.min_cl, args.mach_tolerance)
            result["source_evaluation_sha256"] = sha256_file(args.evaluation)
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
