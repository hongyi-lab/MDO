#!/usr/bin/env python3
"""Create the portable exploratory development comparison (no server required)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdo_demo.mdo_dev_report import write_mdo_dev_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path, help="Pretrained DEV evaluation.json")
    parser.add_argument("--after", required=True, type=Path, help="MDO-adapted DEV evaluation.json")
    parser.add_argument("--output", required=True, type=Path, help="Output self-contained HTML path")
    args = parser.parse_args()
    result = write_mdo_dev_report(args.before, args.after, args.output)
    print(f"DEV exploratory report: {result} ({result.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
