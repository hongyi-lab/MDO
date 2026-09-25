"""Evaluate a real upstream checkpoint against independent CFD reference arrays."""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import write_json
from .metrics import coefficient_errors, field_errors, timing_summary


def evaluate(dataset, predictor, output, limit=8, warmup=1):
    run_started = time.perf_counter()
    if limit < 1 or warmup < 0:
        raise ValueError("limit must be positive; warmup must be nonnegative")
    count = min(limit, len(dataset))
    if count == 0:
        raise ValueError("Dataset is empty")
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory is not empty: {directory}. Choose a new run directory to preserve previous evidence.")
    directory.mkdir(parents=True, exist_ok=True)
    synthetic = bool(getattr(predictor, "provenance", {}).get("synthetic", False))
    first = dataset.sample(0)
    warmup_start = time.perf_counter()
    for _ in range(warmup):
        predictor.predict(first)
    warmup_seconds = time.perf_counter() - warmup_start
    cases, elapsed = [], []
    for index in range(count):
        sample = dataset.sample(index)
        started = time.perf_counter()
        prediction = predictor.predict(sample)
        elapsed.append(time.perf_counter() - started)
        truth = np.asarray(sample["fields"])
        predicted = np.asarray(prediction["fields"])
        if truth.shape != predicted.shape or truth.ndim != 3 or truth.shape[0] != 3:
            raise ValueError("Expected matching [Cp,Cf_stream*150,Cf_span*300] surface arrays")
        coefficients = prediction.get("coefficients", {})
        reference = sample.get("reference_coefficients", {})
        reintegrated = predictor.integrate_reference(sample) if hasattr(predictor, "integrate_reference") else None
        condition = np.asarray(sample["condition"]).reshape(-1)
        if condition.size != 2:
            raise ValueError("Official model condition must be [AoA degrees, Mach]")
        metrics = {
            "Cp": field_errors(predicted[0], truth[0]),
            "Cf_stream": field_errors(predicted[1] / 150, truth[1] / 150),
            "Cf_span": field_errors(predicted[2] / 300, truth[2] / 300),
            "upstream_scaled_fields": field_errors(predicted, truth),
        }
        case = {"sample_id": str(sample["sample_id"]), "shape_id": str(sample["shape_id"]),
                "condition": {"alpha_deg": float(condition[0]), "mach": float(condition[1])},
                "coefficients": coefficients, "reference_coefficients": reference,
                "solver_coefficients": sample.get("solver_coefficients", {}),
                "field_errors": metrics, "coefficient_errors": coefficient_errors(coefficients, reference),
                "reference_reintegrated": reintegrated,
                "integration_consistency_errors": coefficient_errors(reintegrated, reference) if reintegrated else None,
                "model_only_coefficient_errors": coefficient_errors(coefficients, reintegrated) if reintegrated else None,
                "end_to_end_prediction_s": elapsed[-1], "timings": prediction.get("timings", {}),
                "array_file": f"case_{index:04d}.npz"}
        np.savez_compressed(directory / case["array_file"], geometry=sample["geometry"],
                            truth=truth, prediction=predicted)
        cases.append(case)
    report = {"schema_version": 1,
              "mode": "synthetic_interface_smoke" if synthetic else "real_checkpoint_dataset_evaluation",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "reference_source": "Analytic fixture; not CFD" if synthetic else "Published CRMpert CFD surface data; not a live CFD run",
              "coefficient_definition": "Upstream surface reintegration, not raw solver force coefficients",
              "moment_note": "Pinned cfdpost uses reference point (0.25,0,0); CRMpert card wording differs. Check integration_consistency_errors, especially CM.",
              "split_claim": "Convenience subset; do not call this an official held-out benchmark score",
              "case_count": len(cases), "device": str(getattr(predictor, "device", "unknown")),
              "host": platform.platform(),
              "warmup": {"calls": warmup, "wall_s": warmup_seconds},
              "model_loading_s": getattr(predictor, "load_seconds", None),
              "evaluation_wall_s": time.perf_counter() - run_started,
              "prediction_timing": timing_summary(elapsed),
              "component_timing": {key: timing_summary([c["timings"][key] for c in cases])
                                   for key in ("prepare_seconds", "model_seconds", "postprocess_seconds")
                                   if all(key in c["timings"] for c in cases)},
              "timing_scope": "Batch-one predict including tensor conversion, GPU synchronization, CPU output and coefficient integration; excludes model loading, dataset loading and report writing",
              "provenance": getattr(predictor, "provenance", {}),
              "dataset_provenance": getattr(dataset, "manifest", {}),
              "speedup": None,
              "speedup_note": "No matched live CFD timings supplied; no speedup claim is made",
              "cases": cases}
    write_json(directory / "evaluation.json", report)
    return report
