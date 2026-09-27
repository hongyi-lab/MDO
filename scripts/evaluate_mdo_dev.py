#!/usr/bin/env python3
"""Evaluate only the frozen development partition; calibration/test stay sealed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from mdo_demo.mdo_evaluation import evaluate_dev, compare_dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--data", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--split", default="configs/protocol_v1_split.json")
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--warmup", type=int, default=10)
    compare = sub.add_parser("compare")
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "evaluate":
        result = evaluate_dev(args.data, args.checkpoint, args.split, args.output, args.device, args.warmup)
        print(json.dumps(result["macro_metrics"], indent=2))
    else:
        print(json.dumps(compare_dev(args.before, args.after, args.output), indent=2))


if __name__ == "__main__":
    main()
