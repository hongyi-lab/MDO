#!/usr/bin/env python3
"""Repeat batch-one FM queries on the exact surface/conditions supplied to CFD."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np
from mdo_demo.aerotransformer import AeroTransformerPredictor
from mdo_demo.matched_cfd import prediction_sample
from mdo_demo.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--repeats",type=int,default=30)
    args=parser.parse_args()
    if args.warmup<1 or args.repeats<2:
        parser.error("Use at least one warmup and two measured calls")
    array_path=args.output.with_suffix(".npz")
    if args.output.exists() or array_path.exists():
        raise ValueError("Choose new output paths to preserve prior evidence")
    sample,bundle=prediction_sample(args.request)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    model=AeroTransformerPredictor(args.checkpoint,args.device)
    warmup=[]
    for _ in range(args.warmup):
        warmup.append(model.predict(sample)["timings"]["total_seconds"])
    times=[]
    elapsed=[]
    for _ in range(args.repeats):
        started=time.perf_counter()
        prediction=model.predict(sample)
        elapsed.append(time.perf_counter()-started)
        times.append(prediction["timings"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(array_path,fields=prediction["fields"],geometry=sample["geometry"],
                        original_geometry=sample["original_geometry"])
    result={"schema_version":1,"case_sha256":bundle["case_sha256"],
            "coefficients":prediction["coefficients"],
            "timing":{key:float(np.median([row[key] for row in times])) for key in times[0]},
            "raw_timings":times,"raw_prediction_wall_s":elapsed,
            "median_prediction_wall_s":float(np.median(elapsed)),
            "p95_prediction_wall_s":float(np.percentile(elapsed,95)),
            "model_load_seconds":model.load_seconds,"warmup_passes":args.warmup,
            "warmup_seconds":warmup,"timed_repeats":args.repeats,
            "timing_scope":"Batch-one input preparation, synchronized model, CPU fields and coefficient integration; loading and disk output excluded and logged separately",
            "coefficient_contract":"FM main-wing surface integral; excludes tip and blunt trailing edge",
            "calibration_status":bundle["calibration_status"],"provenance":prediction["provenance"],
            "fields_file":str(array_path),"equal_accuracy_verified":False,"matched_speedup_eligible":False}
    write_json(args.output,result)
    print(f"Exact-case FM median {result['median_prediction_wall_s']:.6f}s; {args.output}; accuracy not yet verified")


if __name__=="__main__":
    main()
