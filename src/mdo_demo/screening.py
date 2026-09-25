"""Finite candidate screening demo, explicitly not continuous MDO."""

import math


def screen_candidates(evaluation, mach, min_cl, mach_tolerance=1e-6):
    if (evaluation.get("mode") != "real_checkpoint_dataset_evaluation"
            or evaluation.get("provenance", {}).get("synthetic", False)):
        raise ValueError("Screening evidence requires a real checkpoint evaluation")
    if not all(math.isfinite(v) for v in (mach, min_cl, mach_tolerance)) or mach_tolerance < 0:
        raise ValueError("Finite Mach/CL and nonnegative tolerance required")
    matched = [c for c in evaluation["cases"]
               if abs(c["condition"]["mach"]-mach) <= mach_tolerance]
    if not matched:
        raise ValueError("No cases at the requested Mach; evaluate matching conditions first")
    for c in matched:
        for source in ("coefficients", "reference_coefficients"):
            for name in ("CL", "CD"):
                value = c.get(source, {}).get(name)
                if value is None or not math.isfinite(value):
                    raise ValueError(f"Missing or invalid {name} in {source}")
    predicted_feasible = [c for c in matched if c["coefficients"]["CL"] >= min_cl]
    reference_feasible = [c for c in matched if c["reference_coefficients"]["CL"] >= min_cl]
    if not predicted_feasible:
        raise ValueError("Model predicts no candidate satisfying minimum CL")
    chosen = min(predicted_feasible, key=lambda c: c["coefficients"]["CD"])
    best_reference = min(reference_feasible, key=lambda c: c["reference_coefficients"]["CD"]) if reference_feasible else None
    verified = chosen["reference_coefficients"]["CL"] >= min_cl
    return {"schema_version": 1, "mode": "finite_candidate_screening",
            "claim": "Candidate selection checked against published CFD labels; not a new CFD solve or continuous optimization",
            "constraint": {"mach": mach, "mach_tolerance": mach_tolerance, "minimum_CL": min_cl},
            "candidate_count": len(matched), "predicted_feasible_count": len(predicted_feasible),
            "candidate_sample_ids": [c["sample_id"] for c in matched],
            "model_provenance": evaluation.get("provenance", {}),
            "dataset_provenance": evaluation.get("dataset_provenance", {}),
            "source_evaluation_timestamp": evaluation.get("timestamp_utc"),
            "reference_feasible_count": len(reference_feasible),
            "selected_sample_id": chosen["sample_id"],
            "selected_shape_id": chosen.get("shape_id"),
            "selected_condition": chosen["condition"],
            "predicted": chosen["coefficients"], "reference": chosen["reference_coefficients"],
            "reference_feasible": verified,
            "CL_violation": max(0., min_cl-chosen["reference_coefficients"]["CL"]),
            "best_reference_sample_id": best_reference["sample_id"] if best_reference else None,
            "CD_regret": (chosen["reference_coefficients"]["CD"]-best_reference["reference_coefficients"]["CD"])
                          if best_reference and verified else None,
            "limitations": ["Only sampled shapes/conditions considered", "No thickness, stress, volume or aeroelastic constraints", "No MDO speedup claim"]}
