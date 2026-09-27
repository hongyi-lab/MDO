"""Differentiable surface loads and an MDO-oriented supervised adaptation loss.

The force calculation follows the pinned cfdpost BasicWing implementation,
including its surface winding and friction projection. These are aerodynamic
load objectives on a surface reference mesh, not a RANS residual, FEM model,
conservative fluid/structure transfer, or evidence of an MDO speedup.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

CFDPOST_REVISION = "c9fb313a4f3f3912a2e1a1aaf56e9e7a338bfa5a"
DEFAULT_WEIGHTS = {"field": 1.0, "cl": 0.01, "cd": 0.01, "spanwise": 0.1}
DEFAULT_COEFFICIENT_SCALES = (0.01, 0.0005)
DEFAULT_SPANWISE_SCALE = 0.5


def _batch_vector(value, fields: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=fields.dtype, device=fields.device)
    if tensor.ndim == 0:
        tensor = tensor.expand(fields.shape[0])
    if tensor.shape != (fields.shape[0],):
        raise ValueError(f"{name} must be scalar or shape (batch,)")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def surface_loads(fields: torch.Tensor, vertex_geometry: torch.Tensor,
                  aoa_deg, ref_area) -> dict[str, torch.Tensor]:
    """Integrate scaled surface fields on a batch of structured wing meshes.

    Args:
        fields: (B,3,H,W), channels Cp, 150*Cf_tau, 300*Cf_z.
        vertex_geometry: (B,3,H+1,W+1), coordinates x,y,z; z is spanwise.
        aoa_deg: scalar or (B,) angle of attack, in degrees.
        ref_area: scalar or (B,) half-wing reference area, positive.

    Returns:
        coefficients: (B,2), ordered CL, CD.
        spanwise_lift / spanwise_drag: (B,H), dCL/deta and dCD/deta.
        spanwise_width: (B,H), deta; the mesh's represented span is eta=[0,1].
        CM: (B,), diagnostic pitching moment with upstream xRef=0.25.

    Sum(spanwise_lift * spanwise_width) == CL, and similarly for CD.
    The span coordinate uses the mean z of each vertex row. This supports the
    structured, monotonically increasing-span CRMpert reference grid; it is
    not an arbitrary unstructured-surface or structural-load mapper.
    """
    if fields.ndim != 4 or fields.shape[1] != 3 or not fields.is_floating_point():
        raise ValueError("fields must be floating tensor (B,3,H,W)")
    if fields.shape[0] < 1 or fields.shape[2] < 1 or fields.shape[3] < 1:
        raise ValueError("fields must have nonempty batch and spatial dimensions")
    vertices = torch.as_tensor(vertex_geometry, dtype=fields.dtype, device=fields.device)
    expected = (fields.shape[0], 3, fields.shape[2] + 1, fields.shape[3] + 1)
    if vertices.shape != expected:
        raise ValueError(f"vertex_geometry must have shape {expected}; cell-center geometry is insufficient")
    if not bool(torch.isfinite(fields).all()) or not bool(torch.isfinite(vertices).all()):
        raise ValueError("fields and vertex geometry must be finite")
    alpha = _batch_vector(aoa_deg, fields, "aoa_deg") * (math.pi / 180.0)
    area_ref = _batch_vector(ref_area, fields, "ref_area")
    if not bool((area_ref > 0).all()):
        raise ValueError("ref_area must be positive")

    # cfdpost.get_cellinfo_2d: normals use diagonals; area uses two triangles.
    geom = vertices.permute(0, 2, 3, 1)
    p0 = geom[:, :-1, :-1]
    p1 = geom[:, :-1, 1:]
    p2 = geom[:, 1:, 1:]
    p3 = geom[:, 1:, :-1]
    normal = torch.linalg.cross(p2 - p0, p3 - p1, dim=-1)
    normal = normal / (torch.linalg.vector_norm(normal, dim=-1, keepdim=True) + 1e-20)
    cell_area = 0.5 * (
        torch.linalg.vector_norm(torch.linalg.cross(p1 - p0, p2 - p0, dim=-1), dim=-1)
        + torch.linalg.vector_norm(torch.linalg.cross(p2 - p0, p3 - p0, dim=-1), dim=-1))

    # BasicWing._get_xz_cf: normalize chordwise XY tangents at vertex rows,
    # then average neighbouring rows. Do NOT renormalize that average.
    xy_tangent = geom[:, :, 1:, :2] - geom[:, :, :-1, :2]
    xy_tangent = xy_tangent / (torch.linalg.vector_norm(xy_tangent, dim=-1, keepdim=True) + 1e-20)
    xy_tangent = 0.5 * (xy_tangent[:, 1:] + xy_tangent[:, :-1])
    cp = fields[:, 0]
    cf_tau = fields[:, 1] / 150.0
    cf_z = fields[:, 2] / 300.0
    friction = torch.cat((cf_tau.unsqueeze(-1) * xy_tangent, cf_z.unsqueeze(-1)), dim=-1)
    friction = friction - (friction * normal).sum(dim=-1, keepdim=True) * normal
    cell_force = (cp.unsqueeze(-1) * normal + friction) * cell_area.unsqueeze(-1)

    # Upstream _xy_2_cl: CD = Fx*cos(alpha)+Fy*sin(alpha),
    # CL = -Fx*sin(alpha)+Fy*cos(alpha). Integrate each span strip first.
    strip_force = cell_force.sum(dim=2) / area_ref[:, None, None]
    sin_alpha, cos_alpha = torch.sin(alpha)[:, None], torch.cos(alpha)[:, None]
    strip_cl = -strip_force[..., 0] * sin_alpha + strip_force[..., 1] * cos_alpha
    strip_cd = strip_force[..., 0] * cos_alpha + strip_force[..., 1] * sin_alpha
    coefficients = torch.stack((strip_cl.sum(dim=1), strip_cd.sum(dim=1)), dim=1)

    span_z = vertices[:, 2].mean(dim=-1)
    span_width = span_z[:, 1:] - span_z[:, :-1]
    if not bool((span_width > 0).all()):
        raise ValueError("Structured reference mesh requires increasing spanwise z")
    eta_width = span_width / span_width.sum(dim=1, keepdim=True)
    arm = 0.25 * (p0 + p1 + p2 + p3) - fields.new_tensor([0.25, 0.0, 0.0])
    moment = torch.linalg.cross(arm, cell_force, dim=-1).sum(dim=(1, 2)) / area_ref[:, None]
    return {
        "coefficients": coefficients,
        "spanwise_lift": strip_cl / eta_width,
        "spanwise_drag": strip_cd / eta_width,
        "spanwise_width": eta_width,
        "CM": moment[:, 2],
    }


def mdo_loss(prediction: torch.Tensor, target: torch.Tensor,
             vertex_geometry: torch.Tensor, aoa_deg, ref_area, *,
             coefficient_scales=DEFAULT_COEFFICIENT_SCALES,
             spanwise_scale: float = DEFAULT_SPANWISE_SCALE,
             weights: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
    """Field MSE + scaled lift/drag mismatch + span-load distribution mismatch.

    The targets for all load terms are integrated from the SAME training CFD
    surface fields using surface_loads. Historical scalar labels are not mixed
    with the current postprocessor. No extra CFD or structural labels are used.
    Normalization scales/weights are fixed configuration, not fitted using
    calibration/holdout data. Targets are detached before computing the loss.

    Default weights are initial experiment settings, not validated optimal
    values or claims of novelty. CM is deliberately excluded from training.
    """
    if target.shape != prediction.shape:
        raise ValueError("prediction and target fields must have identical shapes")
    scales = torch.as_tensor(coefficient_scales, dtype=prediction.dtype, device=prediction.device)
    if scales.shape != (2,) or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("coefficient_scales must be two finite positive values [CL,CD]")
    if not math.isfinite(float(spanwise_scale)) or spanwise_scale <= 0:
        raise ValueError("spanwise_scale must be finite and positive")
    terms_weights = DEFAULT_WEIGHTS if weights is None else dict(weights)
    if set(terms_weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"weights must contain exactly {tuple(DEFAULT_WEIGHTS)}")
    if any(not math.isfinite(float(value)) or value < 0 for value in terms_weights.values()):
        raise ValueError("weights must be finite and nonnegative")
    if not any(value > 0 for value in terms_weights.values()):
        raise ValueError("At least one loss weight must be positive")
    target = target.detach().to(dtype=prediction.dtype, device=prediction.device)
    predicted_load = surface_loads(prediction, vertex_geometry, aoa_deg, ref_area)
    with torch.no_grad():
        target_load = surface_loads(target, vertex_geometry, aoa_deg, ref_area)
    coefficient_error = (predicted_load["coefficients"] - target_load["coefficients"]) / scales
    span_error = (predicted_load["spanwise_lift"] - target_load["spanwise_lift"]) / spanwise_scale
    terms = {
        "field": torch.mean((prediction - target) ** 2),
        "cl": torch.mean(coefficient_error[:, 0] ** 2),
        "cd": torch.mean(coefficient_error[:, 1] ** 2),
        "spanwise": torch.mean(torch.sum(span_error ** 2 * predicted_load["spanwise_width"], dim=1)),
    }
    terms["total"] = sum(terms_weights[name] * value for name, value in terms.items())
    return terms
