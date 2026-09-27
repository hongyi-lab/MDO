#!/usr/bin/env python3
"""Train the exploratory MDO-only Large pilot with a frozen train450/dev350 split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdo_demo.mdo_adaptation import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Pinned real CRMpert arrays containing train+dev IDs")
    parser.add_argument("--checkpoint", required=True, help="Official ATsurf_L asset directory")
    parser.add_argument("--split", default=str(Path(__file__).resolve().parents[1] / "configs/protocol_v1_split.json"))
    parser.add_argument("--full-index", help="Optional complete hash-pinned index.npy (metadata only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5, help="Fixed final epoch, 1 to 30; default five-epoch pilot")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-steps", type=int, default=20, help="Measured real training steps; updates discarded")
    parser.add_argument("--probe-warmup", type=int, default=5)
    parser.add_argument("--probe-only", action="store_true", help="Resource probe only; do not export trained weights")
    parser.add_argument("--resume", action="store_true", help="Continue matching probe output or last completed epoch")
    args = parser.parse_args(argv)
    try:
        result = train_model(args.data, args.checkpoint, args.split, args.output,
                             device=args.device, epochs=args.epochs, lr=args.lr, seed=args.seed,
                             probe_steps=args.probe_steps, probe_warmup=args.probe_warmup,
                             probe_only=args.probe_only, resume=args.resume, full_index=args.full_index)
        print(json.dumps({"status": result["status"], "experiment": result["experiment"],
                          "output": args.output, "completed_epochs": result.get("completed_epochs")}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
