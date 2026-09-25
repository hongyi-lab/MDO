"""Metrics with explicit units and no hidden CPU/GPU-hour conversion."""

from __future__ import annotations

import math
import numpy as np


def field_errors(prediction, truth):
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if pred.shape != target.shape or pred.size == 0:
        raise ValueError(f"Nonempty equal shapes required: {pred.shape} vs {target.shape}")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
        raise ValueError("Non-finite prediction or reference")
    delta = pred - target
    norm = float(np.linalg.norm(target.ravel()))
    return {"mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "relative_l2": float(np.linalg.norm(delta.ravel()) / norm) if norm > 0 else None,
            "max_absolute_error": float(np.max(np.abs(delta)))}


def coefficient_errors(prediction, truth):
    result = {}
    for name in ("CL", "CD", "CM"):
        p, t = prediction.get(name), truth.get(name)
        if p is None or t is None:
            continue
        p, t = float(p), float(t)
        if not math.isfinite(p) or not math.isfinite(t):
            raise ValueError(f"Non-finite {name}")
        result[name] = {"predicted": p, "reference": t, "absolute_error": abs(p - t),
                        "relative_error": abs(p - t) / abs(t) if abs(t) > 1e-12 else None}
        if name == "CD":
            result[name]["error_drag_counts"] = abs(p - t) * 10000
    return result


def timing_summary(seconds):
    values = np.asarray(seconds, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Timing needs finite nonnegative samples")
    return {"count": int(values.size), "median_s": float(np.median(values)),
            "p95_s": float(np.percentile(values, 95)), "sum_s": float(np.sum(values))}


def break_even_queries(offline_seconds, cfd_seconds, surrogate_seconds,
                       verification_seconds=0.0):
    """Serial wall-time estimate only; assumes fixed cost per query."""
    values = (offline_seconds, cfd_seconds, surrogate_seconds, verification_seconds)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Costs must be finite and nonnegative")
    saving = cfd_seconds - surrogate_seconds
    if saving <= 0:
        return None
    return math.floor((offline_seconds + verification_seconds) / saving) + 1
