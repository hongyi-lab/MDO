"""Public STW rigid-polar transfer probe; this is not a full MDO benchmark.

Uses the actual public OML, with explicit normalization and coverage limits.
No CFD labels enter geometry construction, and no model is trained here.
"""
from pathlib import Path
import re
import numpy as np

UPSTREAM_COMMIT = "48e54b4dcc1c6c696ffc6625c01714f3c3c1244e"


def read_point_zones(path):
    blocks = re.split(r"(?im)^Zone\s+", Path(path).read_text())[1:]
    if not blocks:
        raise ValueError("No Tecplot zones")
    result = []
    for block in blocks:
        lines = block.splitlines()
        if lines[1].strip().upper() != "DATAPACKING=POINT":
            raise ValueError("Only ordered POINT zones supported")
        dims = [int(re.search(r"\b" + axis + r"=(\d+)", lines[0]).group(1))
                for axis in ("I", "J")]
        a = np.fromstring("\n".join(lines[2:]), sep=" ")
        if a.size != dims[0] * dims[1] * 3 or not np.isfinite(a).all():
            raise ValueError("Invalid zone coordinates or size")
        result.append(a.reshape(dims[1], dims[0], 3))
    return result


def chord_stencil(n=128):
    # The same clustcos defaults as pinned cfdpost.modeling.dist_clustcos.
    t = np.linspace(0, 1, n)
    raw = (1 - np.cos(np.pi * (.0079 * (1-t) + .96*t))) / 2
    return (raw - raw[0]) / (raw[-1] - raw[0])


def make_stw_surface(path):
    zones = read_point_zones(path)
    if len(zones) != 12 or zones[0].shape != (4, 1528, 3) or zones[1].shape != zones[0].shape:
        raise ValueError("Unexpected public STW OML topology")
    upper, lower = zones[:2]
    source_span = upper[:, 0, 1]
    if not np.allclose(source_span, np.linspace(0, 14, 4), atol=1e-8):
        raise ValueError("Unexpected STW span")
    if not np.allclose(upper[:, -1], lower[:, -1], atol=1e-10):
        raise ValueError("Upper/lower leading-edge seams do not meet")
    fractions = chord_stencil()
    sampled = []
    for j in range(4):
        # Source lower patch wraps a tiny distance round the nose. Split at
        # the true minimum x, not the CAD patch seam, to preserve that nose.
        ring = np.concatenate([lower[j], upper[j, -2::-1]])
        leading = int(np.argmin(ring[:, 0]))
        x0 = ring[leading, 0]
        x1 = .5*(ring[0, 0] + ring[-1, 0])
        if not np.isclose(x1-x0, 5-3.5*source_span[j]/14, atol=1e-5):
            raise ValueError("Unexpected STW chord")
        branches = [ring[:leading+1][::-1], ring[leading:]]
        sides = []
        for branch in branches:
            if np.any(np.diff(branch[:, 0]) <= 0):
                raise ValueError("Non-monotonic airfoil branch")
            x = x0 + fractions * (x1-x0)
            sides.append(np.column_stack([x, np.full(128, source_span[j]),
                                          np.interp(x, branch[:, 0], branch[:, 2])]))
        lo, up = sides
        # 127 lower + 127 upper + 2 explicit blunt-TE cells = 256 cells.
        sampled.append(np.concatenate([lo[::-1], up[1:], [.5*(up[-1]+lo[-1]), lo[-1]]]))
    sampled = np.asarray(sampled)
    target_span = np.linspace(0, 14, 129)
    native = np.empty((129, 257, 3))
    for i in range(257):
        for k in range(3):
            native[:, i, k] = np.interp(target_span, source_span, sampled[:, i, k])
    native[:, :, 1] = target_span[:, None]
    if not np.allclose(native[:, 0], native[:, -1], atol=1e-14):
        raise ValueError("Surface ring is not closed")
    original = native[..., [0, 2, 1]].transpose(2, 0, 1) / 5.0
    centers = .25*(original[:, 1:, 1:]+original[:, :-1, 1:]
                    +original[:, 1:, :-1]+original[:, :-1, :-1])
    chord = native[:, :, 0].max(1) - native[:, :, 0].min(1)
    area = float(np.sum(.5*(chord[1:]+chord[:-1])*np.diff(target_span)))
    if abs(area/45.5-1) > 1e-5:
        raise ValueError("Reference planform area mismatch")
    audit = {
        "scope": "Rigid baseline wing; no structural optimization or aeroelastic analysis",
        "source": "Public STW wing.dat first two high-resolution main OML patches",
        "native_axes": "x streamwise, y spanwise, z vertical",
        "model_axes": "x streamwise, y vertical, z positive span",
        "normalization_length_m": 5., "area_reference_m2": 45.5,
        "area_reference_normalized": 45.5/25., "sampled_planform_area_m2": area,
        "model_rotation_deg": 0.,
        "rotation_reason": "STW root twist is zero; CRM-specific +6.7166 deg must not be applied",
        "surface_vertex_shape": list(original.shape), "closed_trailing_edge": True,
        "main_span_m": [0., 14.], "rounded_tip_cap_included": False,
        "source_max_span_m": float(max(a[..., 1].max() for a in zones)),
        "sampling": "128 clustcos points per side; 2 blunt-TE cells; 129 uniform span stations",
        "whole_span_root_to_tip": True,
        "sampling_matches_training_verified": False,
        "reference_conditions": {"mach": .77, "altitude_m": 10400},
        "model_training_conditions": {"reynolds": 20000000., "temperature_K": 300.,
                                      "mach_range": [.75, .90], "untwisted_alpha_range_deg": [2., 12.]},
        "exact_physics_match": False,
        "reason": "Frozen model has no Re/temperature input; reference uses cruise altitude; root/tip coverage and sampling differ",
        "score_interpretation": "Uncalibrated transfer discrepancy against public RANS reference, not pure matched-physics model error",
        "mdo_score_eligible": False, "matched_speedup_eligible": False,
    }
    return {"original_geometry": original, "geometry": centers.astype(np.float32),
            "native_vertices_m": native, "ref_area": 45.5/25.}, audit


def polar_scores(predicted, reference):
    p, r = np.asarray(predicted, float), np.asarray(reference, float)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 2 or len(p) < 1:
        raise ValueError("Expected matched (N,2) CL/CD pairs")
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(r == 0):
        raise ValueError("Finite predictions and nonzero references required")
    e = p-r
    return {"n_conditions": len(p), "n_geometries": 1,
            "CL_MAE": float(np.mean(abs(e[:, 0]))),
            "CD_MAE_drag_counts": float(1e4*np.mean(abs(e[:, 1]))),
            "CL_mean_absolute_percent_discrepancy": float(100*np.mean(abs(e[:, 0]/r[:, 0]))),
            "CD_mean_absolute_percent_discrepancy": float(100*np.mean(abs(e[:, 1]/r[:, 1]))),
            "CL_max_absolute_discrepancy": float(np.max(abs(e[:, 0]))),
            "CD_max_absolute_discrepancy_counts": float(1e4*np.max(abs(e[:, 1]))),
            "CL_signed_bias": float(np.mean(e[:, 0])), "CD_signed_bias_counts": float(1e4*np.mean(e[:, 1]))}
