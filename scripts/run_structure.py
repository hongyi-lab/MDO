#!/usr/bin/env python3
"""Run the native-frame, one-way TACS shell wingbox analysis (no beam fallback)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from mdo_demo.structural import run_structural_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, help="Physical surface vertices and point forces in native SI frame")
    parser.add_argument("--config", required=True, help="Explicit material, skin/web thickness and mesh configuration JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_structural_case(args.input_npz, args.config, args.output_dir)
        print(json.dumps({key: result[key] for key in ("status", "mass_kg", "max_displacement_m",
                         "tip_displacement_m", "ks_failure", "solve_seconds")}, indent=2))
        return 0 if result["status"] == "ok" else 3
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
