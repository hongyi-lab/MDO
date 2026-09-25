#!/usr/bin/env python3
"""Train a small real FM adaptation and evaluate untouched geometry groups."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdo_demo.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="assets/CRMpert72")
    parser.add_argument("--checkpoint", default="assets/AeroTransformer/ATsurf_S")
    parser.add_argument("--config", default="configs/cfd_validation.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_experiment(args.data, args.checkpoint, args.config, args.output, args.device)
    for name, value in result["results"].items():
        print(name, value["metrics"])
    print(f"Open {Path(args.output) / 'report.html'}. Live CFD speedup remains unmeasured.")


if __name__ == "__main__":
    main()
