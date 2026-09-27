"""Development-only evaluation for the MDO-objective pilot; no test-set API."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import os

import numpy as np

from .io import read_json, write_json


class DevOnlyDataset:
    """Expose exactly declared development rows, not the underlying full dataset."""
    def __init__(self, dataset, resolved_split):
        self._dataset = dataset
        self._rows = list(resolved_split["dev_rows"])
        allowed = set(resolved_split["dev_sample_ids"])
        forbidden = set(resolved_split["calibration_sample_ids"]) | set(resolved_split["test_sample_ids"])
        observed = [int(dataset.sample_ids[row]) for row in self._rows]
        if len(observed) != len(set(observed)) or set(observed) != allowed or set(observed) & forbidden:
            raise ValueError("Development selection does not match the sealed split")
        self.sample_ids = observed
        self.manifest = {**dataset.manifest, "evaluation_partition": "dev",
                         "evaluation_sample_ids": observed,
                         "claim": "Development diagnostics, not held-out test performance"}

    def __len__(self):
        return len(self._rows)

    def sample(self, index):
        if not 0 <= index < len(self._rows):
            raise IndexError(index)
        result = self._dataset.sample(self._rows[index])
        if int(result["sample_id"]) != self.sample_ids[index]:
            raise ValueError("Dataset changed after development selection")
        return result


def macro_metrics(cases):
    """First average conditions within geometry, then weight geometries equally."""
    groups = defaultdict(list)
    for case in cases:
        groups[str(case["shape_id"])].append(case)
    if not groups:
        raise ValueError("No development cases")
    result = {"samples": len(cases), "geometries": len(groups),
              "aggregation": "equal geometry weights, equal condition weights within geometry"}
    for name in ("CL", "CD"):
        errors = [np.array([float(c["coefficients"][name]) - float(c["reference_coefficients"][name])
                            for c in items]) for items in groups.values()]
        if any(not np.isfinite(e).all() for e in errors):
            raise ValueError("Non-finite coefficient error")
        worst = sorted(float(np.max(np.abs(e))) for e in errors)
        result[name] = {"mae": float(np.mean([np.abs(e).mean() for e in errors])),
                        "rmse": float(np.sqrt(np.mean([np.mean(e**2) for e in errors]))),
                        "p95_geometry_max_abs": worst[math.ceil(.95*len(worst))-1],
                        "max_abs": worst[-1]}
    for name in ("Cp", "Cf_stream", "Cf_span"):
        result[name + "_mean_rmse"] = float(np.mean([
            np.mean([c["field_errors"][name]["rmse"] for c in items]) for items in groups.values()]))
    if all("spanwise_lift_rmse" in case for case in cases):
        result["spanwise_lift_mean_rmse"] = float(np.mean([
            np.mean([c["spanwise_lift_rmse"] for c in items]) for items in groups.values()]))
    result["CD_drag_counts_mae"] = result["CD"]["mae"]*10000
    result["normalized_coefficient_score"] = max(result["CL"]["mae"]/.01, result["CD"]["mae"]/.0005)
    result["median_prediction_seconds"] = float(np.median([c["end_to_end_prediction_s"] for c in cases]))
    result["p95_prediction_seconds"] = float(np.percentile([c["end_to_end_prediction_s"] for c in cases], 95))
    return result


def add_spanwise_diagnostics(subset, report, output):
    """Derive development load targets from CFD fields, outside the inference timer."""
    import torch
    from .physics_losses import surface_loads
    output = Path(output)
    for index, case in enumerate(report["cases"]):
        sample = subset.sample(index)
        path = output/case["array_file"]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        tensors = [torch.as_tensor(arrays[key], dtype=torch.float64).unsqueeze(0)
                   for key in ("prediction", "truth")]
        vertices = torch.as_tensor(sample["original_geometry"], dtype=torch.float64).unsqueeze(0)
        with torch.inference_mode():
            pred, truth = [surface_loads(x, vertices, float(sample["condition"][0]), sample["ref_area"])
                           for x in tensors]
        width = truth["spanwise_width"][0].numpy()
        pred_lift, true_lift = pred["spanwise_lift"][0].numpy(), truth["spanwise_lift"][0].numpy()
        case["spanwise_lift_rmse"] = float(np.sqrt(np.sum((pred_lift-true_lift)**2*width)))
        case["spanwise_lift_definition"] = "nondimensional dCL/deta; width-weighted RMSE over represented surface"
        arrays.update(spanwise_lift_prediction=pred_lift, spanwise_lift_truth=true_lift, spanwise_width=width)
        np.savez_compressed(path, **arrays)


def evaluate_dev(data_dir, checkpoint, split_manifest, output, device="cpu", warmup=10):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .dataset import load_dataset
    from .experiment import verify_dataset
    from .mdo_adaptation import resolve_mdo_split
    from .aerotransformer import AeroTransformerPredictor
    from .evaluation import evaluate

    dataset = load_dataset(data_dir)
    dataset_sha = verify_dataset(dataset)
    split = resolve_mdo_split(dataset, split_manifest)
    allowed = set(split["train_sample_ids"]) | set(split["dev_sample_ids"])
    if not set(map(int, dataset.sample_ids)).issubset(allowed):
        raise ValueError("Pilot data directory may contain only frozen train/dev rows")
    subset = DevOnlyDataset(dataset, split)
    predictor = AeroTransformerPredictor(checkpoint, device=device)
    # Fixed FP32 convention shared with training; do not alter model weights.
    predictor.torch.backends.cuda.matmul.allow_tf32 = False
    predictor.torch.backends.cudnn.allow_tf32 = False
    predictor.torch.backends.cudnn.benchmark = False
    predictor.torch.use_deterministic_algorithms(True)
    predictor.torch.set_num_threads(4)
    report = evaluate(subset, predictor, output, limit=len(subset), warmup=warmup)
    report["split_claim"] = "Frozen Dev only; used for method development, not a blind test or confidence calibration"
    report["partition"] = "dev"
    report["protocol_id"] = "CFD-FM-v2-MDO-pilot"
    report["split_sha256"] = split["split_sha256"]
    report["dataset_manifest_sha256"] = dataset_sha
    add_spanwise_diagnostics(subset, report, output)
    report["macro_metrics"] = macro_metrics(report["cases"])
    write_json(Path(output)/"evaluation.json", report)
    write_json(Path(output)/"metrics.json", {
        "protocol_id": report["protocol_id"], "partition": "dev",
        "warning": report["split_claim"], "metrics": report["macro_metrics"],
        "provenance": predictor.provenance, "prediction_timing": report["prediction_timing"],
        "component_timing": report["component_timing"],
        "live_cfd_speedup": None,
    })
    return report


def compare_dev(before_path, after_path, output):
    before, after = read_json(before_path), read_json(after_path)
    for report in (before, after):
        if report.get("partition") != "dev":
            raise ValueError("Only development reports can enter this exploratory comparison")
    if (before["split_sha256"] != after["split_sha256"]
            or before["dataset_manifest_sha256"] != after["dataset_manifest_sha256"]
            or [c["sample_id"] for c in before["cases"]] != [c["sample_id"] for c in after["cases"]]):
        raise ValueError("Both models must use the identical dataset, split and ordered development cases")
    metrics_before, metrics_after = before["macro_metrics"], after["macro_metrics"]
    result = {"protocol_id": "CFD-FM-v2-MDO-pilot", "partition": "dev",
              "claim": "Development comparison only; no blind-test reliability or CFD speedup claim",
              "pretrained": metrics_before, "mdo_adapted": metrics_after,
              "relative_improvements": {name: 1-metrics_after[name]["mae"]/metrics_before[name]["mae"]
                                        if metrics_before[name]["mae"] > 0 else None for name in ("CL", "CD")},
              "live_cfd_speedup": None}
    write_json(output, result)
    return result
