"""Geometry-disjoint adaptation, calibration, and held-out CFD-label validation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .io import read_json, write_json, sha256_file


def verify_dataset(dataset):
    if (dataset.manifest.get("dataset") != "thuerey-group/CRMpert"
            or dataset.manifest.get("revision") != "88ece28b846fd1d9870933252db556cf97d30ae0"
            or dataset.manifest.get("field_scales") != [1.0, 150.0, 300.0]):
        raise ValueError("This real-data experiment requires the pinned CRMpert release and field conventions")
    files = dataset.manifest.get("files", {})
    if set(files) != {"index.npy", "data.npy", "geom0.npy", "origingeom.npy"}:
        raise ValueError("A verified fetch_assets dataset manifest is required")
    for name, record in files.items():
        if sha256_file(dataset.data_dir / name) != record["local_sha256"]:
            raise ValueError(f"Dataset asset hash mismatch: {name}")
    return sha256_file(dataset.data_dir / "manifest.json")


def augment_split(dataset, split):
    """Join source sample IDs to shape partitions without reading any field labels."""
    for partition in ("train", "calibration", "test"):
        shapes = set(map(str, split[partition]))
        split[f"{partition}_sample_ids"] = [int(sid) for sid, row in zip(dataset.sample_ids, dataset.index)
                                               if str(int(row[0])) in shapes]
    return split


def partition_cases(cases, split, partition):
    shapes = set(map(str, split[partition]))
    return [case for case in cases if str(case["shape_id"]) in shapes]


def metric_summary(cases):
    if not cases:
        raise ValueError("Empty evaluation partition")
    result = {"samples": len(cases), "shapes": len({c["shape_id"] for c in cases})}
    for name in ("CL", "CD"):
        errors = np.array([abs(c["coefficients"][name] - c["reference_coefficients"][name]) for c in cases])
        result[name] = {"mae": float(errors.mean()), "max_abs": float(errors.max()),
                        "rmse": float(np.sqrt(np.mean(errors ** 2)))}
    result["CD_drag_counts_mae"] = result["CD"]["mae"] * 10000
    for name in ("Cp", "Cf_stream", "Cf_span"):
        result[name + "_mean_rmse"] = float(np.mean([c["field_errors"][name]["rmse"] for c in cases]))
    result["median_prediction_seconds"] = float(np.median([c["end_to_end_prediction_s"] for c in cases]))
    return result


def run_experiment(data_dir, checkpoint, config_path, output, device="cpu"):
    from .adaptation import train_model
    from .aerotransformer import AeroTransformerPredictor, prediction_fingerprint
    from .confidence import split_geometry_groups, fit_calibration, assess_prediction, evaluate_heldout, describe_geometry
    from .dataset import load_dataset
    from .evaluation import evaluate
    from .validation_report import write_validation_report

    started = time.perf_counter()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new experiment output directory; previous evidence is never overwritten")
    config = read_json(config_path)
    dataset = load_dataset(data_dir)
    dataset_hash = verify_dataset(dataset)
    split = split_geometry_groups([str(int(x)) for x in dataset.index[:, 0]],
                                  seed=config["seed"], counts=config["shape_counts"])
    augment_split(dataset, split)
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "split.json"
    write_json(split_path, split)
    protocol = {"config": config, "dataset_manifest_sha256": dataset_hash,
                "base_weights_sha256": sha256_file(Path(checkpoint) / "best_model_weights"),
                "split_fingerprint": split["fingerprint"], "device": device,
                "selection": "fixed final training epoch; no test-set model or hyperparameter selection",
                "reference": "Published CFD labels; no live CFD timing exists in this stage"}
    write_json(output / "protocol.json", protocol)
    print(f"Fixed protocol: {len(split['train'])} train / {len(split['calibration'])} calibration / {len(split['test'])} test geometries", flush=True)
    training = train_model(data_dir, checkpoint, split_path, output / "adapted_checkpoint",
                           device=device, seed=config["seed"], **config["adaptation"])
    results = {}
    for name, model_path in (("pretrained", checkpoint), ("adapted", output / "adapted_checkpoint")):
        print(f"Evaluating {name} model", flush=True)
        predictor = AeroTransformerPredictor(model_path, device=device)
        evaluation = evaluate(dataset, predictor, output / name, limit=len(dataset), warmup=2)
        if evaluation["mode"] != "real_checkpoint_dataset_evaluation":
            raise ValueError("Synthetic evaluations cannot be reported as real CFD evidence")
        identity = {"model": prediction_fingerprint(predictor.provenance), "dataset": dataset_hash,
                    "split": split["fingerprint"]}
        for index, case in enumerate(evaluation["cases"]):
            # Inputs alone define the support descriptors, never reference fields/errors.
            shape_row = dataset._shape_lookup[int(dataset.index[index, 0])]
            case["geometry_descriptors"] = describe_geometry(np.asarray(dataset.geometry[shape_row]))
            case["prediction_identity"] = identity
        calibration = fit_calibration(partition_cases(evaluation["cases"], split, "calibration"),
                                      identity, tolerances=config["tolerances"], alpha=config["alpha"],
                                      split=split, support_cases=partition_cases(evaluation["cases"], split, "train"))
        heldout = partition_cases(evaluation["cases"], split, "test")
        for case in heldout:
            case["gate"] = assess_prediction(case, calibration)
        write_json(output / name / "evaluation.json", evaluation)
        write_json(output / name / "calibration.json", calibration)
        assessment = evaluate_heldout(heldout, calibration)
        results[name] = {"metrics": metric_summary(heldout), "calibration": calibration,
                         "heldout_assessment": assessment, "heldout_cases": heldout,
                         "evaluation_file": f"{name}/evaluation.json", "identity": identity}
        del predictor
    summary = {"schema_version": 1, "mode": "real_fm_adaptation_and_heldout_cfd_label_validation",
               "protocol": protocol, "split": split, "training": training, "results": results,
               "total_workflow_seconds": time.perf_counter() - started,
               "live_cfd_speedup": None,
               "fallback_mode": "offline routing decisions only; CFD labels are not new solver executions",
               "confidence_scope": "Marginal calibration under exchangeable geometry groups; not a per-case guarantee or a guarantee after selection",
               "limitations": ["Published CRMpert labels, not freshly measured CFD wall times",
                               "Exploratory sample count and empirical input support envelope",
                               "Only CL/CD have calibrated intervals; fields, moments, loads and gradients are not certified",
                               "No MDO or aeroelastic coupling validation",
                               "One released condition per selected geometry; no multi-condition group coverage claim"]}
    write_json(output / "summary.json", summary)
    write_validation_report(summary, output)
    return summary
