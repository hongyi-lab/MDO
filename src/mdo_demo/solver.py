"""Independent surrogate service with an explicit, checked CFD fallback hook.

This is a surface-field/CL-CD service, not an MDO nodal-load/adjoint component.
The callback must really run the requested CFD case and report its provenance.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .confidence import assess_prediction, describe_geometry
from .aerotransformer import prediction_fingerprint


FIXED_PHYSICS = {"reynolds": 20_000_000.0, "reynolds_length_m": 1.0,
                 "temperature_k": 300.0, "equations": "RANS-SA"}
COEFFICIENT_DEFINITION = "cfdpost_surface_integrated_CL_CD"
FIELD_SCALES = [1.0, 150.0, 300.0]


def query_fingerprint(query):
    digest = hashlib.sha256()
    for key in ("geometry", "original_geometry", "condition"):
        array = np.asarray(query[key], dtype="<f8")
        expected = {"geometry": (3, 128, 256), "original_geometry": (3, 129, 257), "condition": (2,)}[key]
        if array.shape != expected:
            raise ValueError(f"{key} must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite {key}")
        digest.update(key.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    reference = float(query["ref_area"])
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Positive finite reference area required")
    digest.update(json.dumps({"ref_area": reference, "physics": query["physics"]},
                             sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


class HybridAerodynamicSolver:
    """Reuse a loaded predictor and calibration; abstain instead of guessing CFD.

    ``cfd_backend(query, query_sha256)`` must return status=ok,
    convergence.converged=True, query_sha256, fields, coefficients and the
    same coefficient_definition. Raw ADflow CL/CD are a distinct convention;
    they cannot silently replace the calibrated surface-integrated target.
    """

    def __init__(self, predictor, calibration, dataset_fingerprint, split_fingerprint, cfd_backend=None):
        self.predictor = predictor
        self.calibration = calibration
        self.identity = {"model": prediction_fingerprint(predictor.provenance),
                         "dataset": dataset_fingerprint, "split": split_fingerprint}
        self.cfd_backend = cfd_backend

    def solve(self, query):
        start = time.perf_counter()
        # Explicitly strip any reference labels accidentally supplied by a dataset loader.
        query = {key: query[key] for key in ("geometry", "original_geometry", "condition", "ref_area",
                                             "physics", "sample_id", "shape_id") if key in query}
        fingerprint = query_fingerprint(query)
        gate = {"accepted": False, "fallback_required": True, "reasons": []}
        prediction = None
        if query["physics"] != FIXED_PHYSICS:
            gate["reasons"] = ["physics_outside_released_model_contract"]
        else:
            try:
                prediction = self.predictor.predict(query)
                condition = np.asarray(query["condition"])
                case = {"sample_id": str(query.get("sample_id", fingerprint)),
                        "shape_id": str(query.get("shape_id", fingerprint)),
                        "condition": {"alpha_deg": float(condition[0]), "mach": float(condition[1])},
                        "geometry_descriptors": describe_geometry(query["geometry"]),
                        "coefficients": prediction["coefficients"], "prediction_identity": self.identity}
                gate = assess_prediction(case, self.calibration)
            except (RuntimeError, ValueError, OSError) as exc:
                gate["reasons"] = ["fm_prediction_failed"]
                gate["failure"] = str(exc)
        result = {"query_sha256": fingerprint, "gate": gate, "prediction_identity": self.identity,
                  "coefficient_definition": COEFFICIENT_DEFINITION,
                  "field_names": ["Cp", "Cf_stream_x150", "Cf_span_x300"], "field_scales": FIELD_SCALES,
                  "confidence_scope": "CL/CD marginal calibration only; no field/load/gradient guarantee"}
        if gate["accepted"]:
            result.update(status="fm_accepted", backend="fm", coefficients=prediction["coefficients"],
                          fields=prediction["fields"], cfd_executed=False)
        elif self.cfd_backend is None:
            result.update(status="cfd_required", backend=None, coefficients=None, fields=None,
                          cfd_executed=False)
        else:
            truth = self.cfd_backend(query, fingerprint)
            if (truth.get("status") != "ok" or truth.get("convergence", {}).get("converged") is not True
                    or truth.get("query_sha256") != fingerprint
                    or truth.get("coefficient_definition") != COEFFICIENT_DEFINITION
                    or truth.get("field_scales") != FIELD_SCALES):
                raise RuntimeError("CFD fallback failed convergence, input identity, or coefficient convention checks")
            fields = np.asarray(truth["fields"])
            coefficients = truth["coefficients"]
            if fields.shape != (3, 128, 256) or not np.all(np.isfinite(fields)):
                raise RuntimeError("CFD fallback did not supply matching finite surface fields")
            if any(not np.isfinite(float(coefficients[name])) for name in ("CL", "CD")):
                raise RuntimeError("CFD fallback returned non-finite coefficients")
            result.update(status="cfd_ok", backend="cfd", coefficients=coefficients, fields=fields,
                          cfd_executed=True, cfd_provenance=truth)
        result["total_seconds"] = time.perf_counter() - start
        return result
