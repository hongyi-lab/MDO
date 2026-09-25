#!/usr/bin/env python3
"""Train only on explicit training IDs; preserve the official model architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdo_demo.adaptation import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="Base model_config/best_model_weights directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", help="JSON with train/calibration/test source sample IDs")
    source.add_argument("--train-sample-ids", type=int, nargs="+", help="Explicit source IDs, not compact dataset row indices")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", choices=("head", "full"), default="head")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        manifest = train_model(args.data, args.checkpoint, args.split if args.split else args.train_sample_ids,
                               args.output, device=args.device, epochs=args.epochs, mode=args.mode,
                               batch_size=args.batch_size, lr=args.lr, seed=args.seed)
        print(json.dumps({"status": manifest["status"], "output": args.output, "mode": args.mode,
                          "train_samples": len(manifest["train_sample_ids"]),
                          "fit_wall_s": manifest["fit_wall_s"],
                          "checkpoint_selection": manifest["checkpoint_selection"]}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
