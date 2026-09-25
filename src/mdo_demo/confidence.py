"""Geometry-disjoint calibration and an inference-only, fail-closed CFD gate.

The calibrated interval has a marginal, geometry-group interpretation under
exchangeability. It is neither a per-query correctness certificate nor a
guarantee of conditional coverage among the queries accepted by this gate.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from typing import Mapping

import numpy as np


_PARTS = ("train", "calibration", "test")
_COEFFICIENTS = ("CL", "CD")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _shape(case):
    value = case["shape_id"]
    if value is None or str(value) == "":
        raise ValueError("shape_id is required")
    return str(value)


def _split_payload(split):
    return {key: split[key] for key in (*_PARTS, "seed")}


def _validate_split(split):
    groups = [split[part] for part in _PARTS]
    if any(not isinstance(group, list) or any(not isinstance(x, str) for x in group)
           for group in groups):
        raise ValueError("Split groups must be lists of string geometry IDs")
    flattened = [item for group in groups for item in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Geometry groups overlap or contain duplicates")
    if split.get("fingerprint") != _digest(_split_payload(split)):
        raise ValueError("Split fingerprint does not match its geometry groups")


def split_geometry_groups(ids, seed=42, counts=None, fractions=(0.4, 0.3, 0.3)):
    """Return deterministic disjoint geometry IDs, independent of input order.

    Repeated IDs identify multiple conditions of one geometry, not independent
    samples. Explicit counts must account for every unique geometry.
    """
    ids = list(ids)
    if any(value is None or str(value) == "" for value in ids):
        raise ValueError("Geometry IDs must be nonempty")
    ids = sorted({str(value) for value in ids})
    if not ids:
        raise ValueError("At least one geometry is required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if counts is not None:
        if set(counts) != set(_PARTS):
            raise ValueError(f"counts must have exactly {_PARTS}")
        sizes = [counts[part] for part in _PARTS]
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in sizes):
            raise ValueError("Split counts must be nonnegative integers")
        if sum(sizes) != len(ids):
            raise ValueError("Split counts must sum to the unique geometry count")
    else:
        fractions = [_finite(x, "split fraction") for x in fractions]
        if len(fractions) != 3 or any(x < 0 for x in fractions) or not math.isclose(sum(fractions), 1):
            raise ValueError("Three nonnegative fractions must sum to one")
        raw = [len(ids) * fraction for fraction in fractions]
        sizes = [math.floor(x) for x in raw]
        order = sorted(range(3), key=lambda index: (-(raw[index] - sizes[index]), index))
        for index in order[:len(ids) - sum(sizes)]:
            sizes[index] += 1
    random.Random(seed).shuffle(ids)
    split, start = {"seed": seed}, 0
    for part, size in zip(_PARTS, sizes):
        split[part] = sorted(ids[start:start + size])
        start += size
    split["fingerprint"] = _digest(_split_payload(split))
    return split


def describe_geometry(geometry):
    """Coarse fixed-axis extent and covariance descriptors; not an OOD proof."""
    array = np.asarray(geometry, dtype=float)
    if array.ndim < 2 or not np.all(np.isfinite(array)):
        raise ValueError("Geometry must contain finite three-dimensional points")
    if array.shape[0] == 3:
        points = array.reshape(3, -1).T
    elif array.shape[-1] == 3:
        points = array.reshape(-1, 3)
    else:
        raise ValueError("Geometry must have a leading or trailing coordinate axis of size 3")
    if len(points) < 3:
        raise ValueError("Geometry needs at least three points")
    extents = np.ptp(points, axis=0)
    if extents[0] <= 0 or np.count_nonzero(extents > 0) < 2:
        raise ValueError("Geometry needs positive streamwise extent and a nondegenerate surface")
    eigenvalues = np.maximum(np.linalg.eigvalsh(np.cov(points, rowvar=False)), 0)
    if eigenvalues[-1] <= 0:
        raise ValueError("Degenerate geometry covariance")
    return {"x_extent": float(extents[0]), "y_extent": float(extents[1]),
            "z_extent": float(extents[2]),
            "covariance_small_ratio": float(eigenvalues[0] / eigenvalues[-1]),
            "covariance_middle_ratio": float(eigenvalues[1] / eigenvalues[-1])}


def _identity(identity):
    if not isinstance(identity, Mapping) or set(identity) != {"model", "dataset", "split"}:
        raise ValueError("prediction_identity must contain exactly model, dataset, split")
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("Prediction identity values must be nonempty fingerprint strings")
    return dict(identity)


def _covariates(case):
    condition = case["condition"]
    if not isinstance(condition, Mapping) or set(condition) != {"alpha_deg", "mach"}:
        raise ValueError("Only alpha_deg and mach are calibrated; additional flow conditions are out of scope")
    result = {"condition.alpha_deg": _finite(condition["alpha_deg"], "alpha_deg"),
              "condition.mach": _finite(condition["mach"], "mach")}
    if result["condition.mach"] <= 0:
        raise ValueError("Mach must be positive")
    descriptors = case["geometry_descriptors"]
    if not isinstance(descriptors, Mapping) or not descriptors:
        raise ValueError("Nonempty geometry_descriptors are required")
    for key, value in descriptors.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Geometry descriptor names must be nonempty strings")
        result[f"geometry.{key}"] = _finite(value, f"geometry.{key}")
    return result


def _score(case, tolerances):
    return max(abs(_finite(case["coefficients"][name], name)
                   - _finite(case["reference_coefficients"][name], f"reference {name}"))
               / tolerances[name] for name in _COEFFICIENTS)


def fit_calibration(cal_cases, prediction_identity, tolerances=None, alpha=0.1,
                    *, split, support_cases=None):
    """Fit group-max split conformal intervals without using held-out truth.

    ``support_cases`` must come from the training partition only. Each case
    needs shape_id, condition and geometry_descriptors. Calibration needs
    coefficients/reference_coefficients for CL/CD. Missing support or too few
    calibration geometries produces an artifact that always requests CFD.
    Tolerances are engineering settings, not established publication criteria.
    """
    _validate_split(split)
    identity = _identity(prediction_identity)
    if identity["split"] != split["fingerprint"]:
        raise ValueError("Identity and split fingerprint mismatch")
    alpha = _finite(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    tolerances = dict(tolerances or {"CL": 0.01, "CD": 0.0005})
    if set(tolerances) != set(_COEFFICIENTS):
        raise ValueError("Only CL and CD are calibrated; CM and Cp are diagnostic")
    tolerances = {name: _finite(tolerances[name], f"{name} tolerance") for name in _COEFFICIENTS}
    if any(value <= 0 for value in tolerances.values()):
        raise ValueError("Tolerances must be positive")
    cal_cases = list(cal_cases)
    support_cases = list(support_cases or [])
    grouped = defaultdict(list)
    seen_samples = set()
    for case in cal_cases:
        shape = _shape(case)
        if shape not in split["calibration"]:
            raise ValueError("Calibration contains a shape outside its partition")
        if "prediction_identity" in case and _identity(case["prediction_identity"]) != identity:
            raise ValueError("Calibration prediction identity mismatch")
        sample = str(case["sample_id"])
        if sample in seen_samples:
            raise ValueError("Duplicate calibration sample_id")
        seen_samples.add(sample)
        grouped[shape].append(_score(case, tolerances))
    if set(grouped) != set(split["calibration"]):
        raise ValueError("Calibration cases must cover every planned calibration geometry")
    support = []
    support_shapes = set()
    for case in support_cases:
        shape = _shape(case)
        if shape not in split["train"]:
            raise ValueError("Support envelope must use training geometries only")
        support_shapes.add(shape)
        support.append(_covariates(case))
    if support and any(set(row) != set(support[0]) for row in support):
        raise ValueError("Support cases must use identical geometry descriptors")
    bounds = {name: [min(row[name] for row in support), max(row[name] for row in support)]
              for name in support[0]} if support else {}
    scores = sorted(max(values) for values in grouped.values())
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    quantile = scores[rank - 1] if rank <= n else None
    status = "ready" if quantile is not None and bounds else (
        "insufficient_calibration_groups" if quantile is None else "support_unavailable")
    calibration = {
        "schema_version": 1, "method": "geometry_group_max_split_conformal",
        "status": status, "prediction_identity": identity,
        "split": _split_payload(split) | {"fingerprint": split["fingerprint"]},
        "alpha": alpha, "nominal_marginal_group_coverage": 1 - alpha,
        "tolerances": tolerances, "calibration_group_count": n,
        "calibration_case_count": len(cal_cases), "finite_sample_rank": rank,
        "quantile_normalized_error": quantile,
        "interval_half_widths": {name: quantile * tolerances[name] if quantile is not None else None
                                 for name in _COEFFICIENTS},
        "group_scores": {shape: max(values) for shape, values in sorted(grouped.items())},
        "group_case_counts": {shape: len(values) for shape, values in sorted(grouped.items())},
        "support_bounds": bounds, "support_shape_ids": sorted(support_shapes),
        "support_source": "training geometries only; axis-aligned empirical envelope",
        "limitations": [
            "Coverage is marginal over exchangeable geometry groups, not a per-query guarantee.",
            "No conditional coverage guarantee applies to the accepted subset.",
            "Calibration and test must share the geometry/condition sampling protocol; group size matters.",
            "The support envelope is a coarse screen, not a validated comprehensive OOD detector.",
            "Unknown overlap with upstream pretraining remains outside this adaptation split guarantee.",
            "CL/CD intervals do not certify pressure fields, structural loads, buffet or optimization gradients.",
            "Default tolerances are engineering settings, not established publication acceptance criteria.",
        ],
    }
    calibration["fingerprint"] = _digest(calibration)
    return calibration


def _validate_calibration(calibration):
    if calibration["schema_version"] != 1 or calibration["method"] != "geometry_group_max_split_conformal":
        raise ValueError("Unsupported calibration schema")
    payload = {key: value for key, value in calibration.items() if key != "fingerprint"}
    if calibration.get("fingerprint") != _digest(payload):
        raise ValueError("Calibration artifact fingerprint mismatch")
    _validate_split(calibration["split"])
    identity = _identity(calibration["prediction_identity"])
    if identity["split"] != calibration["split"]["fingerprint"]:
        raise ValueError("Calibration identity and split mismatch")


def assess_prediction(prediction, calibration):
    """Decide from prediction, identity, covariates and calibration only.

    The function deliberately never reads reference_coefficients, field_errors
    or any other held-out truth. ``fallback_required`` is a request, not proof
    that a live CFD fallback has actually run.
    """
    result = {"decision": "fallback", "accepted": False, "fallback_required": True,
              "reasons": [], "intervals": {}, "calibration_fingerprint": calibration.get("fingerprint")
              if isinstance(calibration, Mapping) else None,
              "scope": "CL/CD marginal geometry-group intervals; no accepted-subset guarantee"}
    try:
        _validate_calibration(calibration)
        if _identity(prediction["prediction_identity"]) != calibration["prediction_identity"]:
            result["reasons"].append("prediction_identity_mismatch")
            return result
        _shape(prediction)
        values = {name: _finite(prediction["coefficients"][name], name) for name in _COEFFICIENTS}
        covariates = _covariates(prediction)
        if calibration["status"] != "ready":
            result["reasons"].append(calibration["status"])
            return result
        if _shape(prediction) in calibration["split"]["calibration"]:
            result["reasons"].append("calibration_geometry_reused_for_inference")
        bounds = calibration["support_bounds"]
        if set(covariates) != set(bounds):
            result["reasons"].append("support_descriptor_schema_mismatch")
        else:
            outside = []
            for name, value in covariates.items():
                low, high = bounds[name]
                rounding = 1e-12 * max(1, abs(low), abs(high))
                if value < low - rounding or value > high + rounding:
                    outside.append(name)
            if outside:
                result["reasons"].append("outside_training_support")
                result["outside_support_covariates"] = outside
        for name, value in values.items():
            half_width = _finite(calibration["interval_half_widths"][name], f"{name} interval")
            if half_width < 0:
                raise ValueError("Interval half width cannot be negative")
            result["intervals"][name] = [value - half_width, value + half_width]
        if calibration["quantile_normalized_error"] > 1:
            result["reasons"].append("calibrated_interval_exceeds_tolerance")
        if not result["reasons"]:
            result.update(decision="surrogate", accepted=True, fallback_required=False)
        return result
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as error:
        result["reasons"].append("invalid_prediction_or_calibration")
        result["detail"] = str(error)
        result["intervals"] = {}
        return result


def evaluate_heldout(cases, calibration):
    """Audit held-out truth after gate decisions; this never refits the gate.

    Coverage here includes accepted AND abstained cases whose intervals exist.
    The accepted-only error rate is descriptive, with no conditional guarantee.
    """
    _validate_calibration(calibration)
    cases = list(cases)
    if not cases:
        raise ValueError("At least one held-out case is required")
    groups, decisions, seen = defaultdict(list), [], set()
    for case in cases:
        shape = _shape(case)
        if shape not in calibration["split"]["test"]:
            raise ValueError("Held-out evaluation contains a geometry outside the test partition")
        sample = str(case["sample_id"])
        if sample in seen:
            raise ValueError("Duplicate held-out sample_id")
        seen.add(sample)
        decision = assess_prediction(case, calibration)
        score = _score(case, calibration["tolerances"])
        q = calibration["quantile_normalized_error"]
        covered = score <= q if q is not None and decision["intervals"] else None
        row = {"sample_id": sample, "shape_id": shape, "decision": decision,
               "observed_normalized_error": score, "observed_within_tolerance": score <= 1,
               "observed_interval_covered": covered,
               "observed_unsafe_acceptance": decision["accepted"] and score > 1}
        decisions.append(row)
        groups[shape].append(row)
    group_covered = [all(row["observed_interval_covered"] for row in rows)
                     for rows in groups.values()
                     if all(row["observed_interval_covered"] is not None for row in rows)]
    accepted = [row for row in decisions if row["decision"]["accepted"]]
    unsafe = sum(row["observed_unsafe_acceptance"] for row in decisions)
    covered_cases = [row["observed_interval_covered"] for row in decisions
                     if row["observed_interval_covered"] is not None]
    return {
        "case_count": len(cases), "test_geometry_count": len(groups),
        "planned_test_geometry_count": len(calibration["split"]["test"]),
        "test_partition_complete": set(groups) == set(calibration["split"]["test"]),
        "accepted_case_count": len(accepted), "fallback_requested_case_count": len(cases) - len(accepted),
        "acceptance_fraction": len(accepted) / len(cases),
        "observed_unsafe_accepted_cases": unsafe,
        "observed_unsafe_fraction_among_accepted": unsafe / len(accepted) if accepted else None,
        "observed_simultaneous_coefficient_case_coverage": sum(covered_cases) / len(covered_cases)
        if covered_cases else None,
        "observed_simultaneous_geometry_group_coverage": sum(group_covered) / len(group_covered)
        if group_covered else None,
        "interval_evaluable_case_count": len(covered_cases),
        "interval_evaluable_geometry_count": len(group_covered),
        "truth_used_for_gate_decision": False,
        "live_cfd_fallback_executed": False,
        "limitations": calibration["limitations"] + [
            f"Only {len(groups)} held-out geometries: observed rates are not reliability certification.",
            "Fallback counts are requests; an offline reference audit is not a timed live CFD fallback.",
        ],
        "cases": decisions,
    }
