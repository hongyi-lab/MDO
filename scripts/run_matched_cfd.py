#!/usr/bin/env python3
"""Prepare a new shared-geometry case, predict with FM, and execute live CFD."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdo_demo.io import read_json, write_json
from mdo_demo.matched_cfd import load_bundle, paired_diagnostics, prediction_sample, prepare_bundle, run_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="Portable native surface + FM input; no CFD dependencies")
    prepare.add_argument("--input-json", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--upstream", type=Path, default=ROOT / "external" / "AeroTransformer")
    prepare.add_argument("--allow-reconstructed-case", action="store_true", help="Acknowledge NEW case / new sampling, not an original CRMpert reproduction")
    validate = sub.add_parser("validate", help="Rehash shared-geometry bundle without CFD")
    validate.add_argument("--request", type=Path, required=True)
    predict = sub.add_parser("predict", help="FM inference on the generated native-case surface")
    predict.add_argument("--request", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Linux/MPI: pyHyp mesh + ADflow; CFD runs on CPU")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--wall-normal-layers", type=int, default=81)
    compare = sub.add_parser("compare", help="Diagnostic pair, not an equal-accuracy speedup claim")
    compare.add_argument("--prediction", type=Path, required=True)
    compare.add_argument("--cfd-result", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_bundle(args.input_json, args.output, args.upstream,
                                allow_reconstructed_case=args.allow_reconstructed_case)
        print(f"New case prepared: {args.output / 'request.json'}; case={result['case_sha256']}")
    elif args.action == "validate":
        result = load_bundle(args.request)
        print(f"Bundle hashes verified; CFD not executed: {result['case_sha256']}")
    elif args.action == "predict":
        import numpy as np
        from mdo_demo.aerotransformer import AeroTransformerPredictor
        sample, manifest = prediction_sample(args.request)
        if args.output.exists():
            raise ValueError("Prediction output already exists; choose a fresh filename")
        model = AeroTransformerPredictor(args.checkpoint, args.device)
        warmup = model.predict(sample)
        prediction = model.predict(sample)
        fields = args.output.with_suffix(".npz")
        if fields.exists():
            raise ValueError("Prediction field output already exists")
        fields.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fields, fields=prediction["fields"], geometry=sample["geometry"],
                            original_geometry=sample["original_geometry"])
        write_json(args.output, {"schema_version": 1, "case_sha256": manifest["case_sha256"],
                                 "coefficients": prediction["coefficients"], "timing": prediction["timings"],
                                 "model_load_seconds": model.load_seconds, "warmup_passes": 1,
                                 "warmup_seconds": warmup["timings"]["total_seconds"],
                                 "coefficient_contract": "FM main-wing surface integral; excludes tip and blunt trailing edge",
                                 "calibration_status": manifest["calibration_status"],
                                 "provenance": prediction["provenance"], "fields_file": str(fields),
                                 "equal_accuracy_verified": False, "matched_speedup_eligible": False})
        print(f"FM prediction saved: {args.output}; UNCALIBRATED on this new case")
    elif args.action == "run":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        try:
            result = run_bundle(args.request, args.output, comm=comm, wall_normal_layers=args.wall_normal_layers)
            if comm.rank == 0:
                print(f"CFD status={result['status']}; {args.output / 'result.json'}", flush=True)
            return 0 if result["status"] == "ok" else 2
        except Exception as exc:
            if comm.rank == 0:
                print(f"CFD failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if comm.size > 1:
                comm.Abort(3)
            return 3
    else:
        write_json(args.output, paired_diagnostics(read_json(args.prediction), read_json(args.cfd_result)))
        print(f"Diagnostic comparison saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
