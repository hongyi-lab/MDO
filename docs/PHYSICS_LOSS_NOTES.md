# MDO-oriented adaptation objective, v2

This experiment starts directly with a combined **surface-field, total-load,
and span-load objective**. It does not require a preliminary ordinary
field-only fine-tuning run. Field-only training is a later ablation.

The aim is to adapt the existing pretrained network while giving lift, drag,
and the aerodynamic load distribution explicit supervision. This is a new
experimental objective, not a reproduction of the paper's reported fine-tuning
numbers and not the original v1 training objective. Its benefit must be measured
on the preregistered development split and finally on the untouched holdout.

## Defined objective

For each minibatch, `mdo_loss` returns:

| Key | Definition | Initial weight |
|---|---|---:|
| `field` | Mean squared error over all cells/channels of `[Cp,150*Cf_tau,300*Cf_z]` | 1 |
| `cl` | Mean `((CL_pred-CL_true)/0.01)^2` | 0.01 |
| `cd` | Mean `((CD_pred-CD_true)/0.0005)^2` | 0.01 |
| `spanwise` | Mean over wings of the integral of `((dCL_pred/deta-dCL_true/deta)/0.5)^2 d eta` | 0.1 |
| `total` | Weighted sum of these four terms | — |

Scales and weights are fixed initial choices and must be stored in training
metadata. They are not optimized or selected using calibration or holdout data.
These weights express a research hypothesis; they do not establish an
improvement before experiments. The combined objective may trade one error
against another, so report its components plus independent physical metrics.

All target loads are integrated from the **same training CFD surface fields**
already used for field supervision. This introduces no new CFD labels, extra
simulations, structural stresses, or claimed structural fidelity. In particular,
the historical scalar coefficient columns are not mixed into this loss: the
current postprocessor has a small measured discrepancy with those stored
labels. Using identical integration for predicted and target fields avoids
training against that postprocessor mismatch. Evaluation should still show
stored-label errors separately for comparison with prior results.

## Surface integration

The Torch implementation in `src/mdo_demo/physics_losses.py` follows
[`cfdpost`](https://github.com/YangYunjia/cfdpost/tree/c9fb313a4f3f3912a2e1a1aaf56e9e7a338bfa5a)
at revision `c9fb313a4f3f3912a2e1a1aaf56e9e7a338bfa5a`, specifically
`BasicWing._get_xz_cf`, `BasicWing.aero_force`, and the helpers in
`cfdpost/utils.py`. That upstream implementation is MIT licensed; this project
retains its source attribution and upstream checkout license.

1. Use vertex geometry `(B,3,H+1,W+1)` and cell fields `(B,3,H,W)`. Coordinates
   are x/y/z, with z spanwise. Pressure retains the upstream surface-winding
   sign, rather than imposing a different outward-normal convention.
2. Form each cell normal from the crossed diagonals and area from the sum of
   two triangle areas, exactly as upstream.
3. Divide friction channels by 150 and 300. Compute chordwise XY unit tangents
   at the vertex span rows, then average neighbouring rows **without
   renormalizing the average**. Reconstruct the XYZ friction vector and remove
   its surface-normal component.
4. Sum pressure plus tangential-friction forces. Rotate into lift and drag
   using AoA in degrees and divide by the half-wing reference area.
5. The sum over chord cells in each span strip gives the strip's contribution
   to CL/CD. Divide by the normalized span-strip width to obtain `dCL/deta`
   and `dCD/deta`. Here eta runs from zero to one over the **represented
   reference-grid span**. This distribution integrates back to the global
   coefficient exactly, including on nonuniform span grids.

CM is returned only as a diagnostic, using upstream `xRef=0.25`; it is not in
the training objective because the dataset card and source use inconsistent
wording for the moment reference. The span distribution is an aerodynamic
surface integral, **not a conservative transfer to an FEM mesh**. This code
contains no structural solver or coupled MDA iteration and no RANS residual.

## API

```python
from mdo_demo.physics_losses import mdo_loss, surface_loads

losses = mdo_loss(predicted_fields, true_fields, vertex_geometry, aoa_deg, ref_area)
losses["total"].backward()

loads = surface_loads(predicted_fields, vertex_geometry, aoa_deg, ref_area)
# loads["coefficients"]: (B,2), columns [CL,CD]
# loads["spanwise_lift"]: (B,H), dCL/deta
# loads["spanwise_drag"]: (B,H), dCD/deta
# loads["spanwise_width"]: (B,H), delta_eta
# loads["CM"]: (B,), diagnostic only
```

For an unnormalized physical span-load error, report
`sqrt(mean(sum((pred_dCL_deta-true_dCL_deta)^2 * delta_eta, dim=1)))`.
The `/0.5` normalization belongs to the training loss, not this reported RMSE.

## Local verification on 2026-09-27

Six tests passed in the optional PyTorch CPU validation environment, including
batched geometry/AoA/reference-area handling, load-density integration, zero
loss and detached targets, rejected invalid inputs, finite differences for all
three field channels, and equivalence to the pinned NumPy postprocessor.

The real-data comparison loaded only source sample IDs **0, 302, 620, 922**,
after checking that all are explicitly in the frozen v1 **training** split.
No calibration or holdout fields were read for this implementation check.

| Maximum absolute Torch–NumPy discrepancy | CL | CD | CM |
|---|---:|---:|---:|
| Float64 | 5.00e-15 | 9.82e-16 | 2.44e-15 |
| Float32 | 2.31e-8 | 6.91e-9 | 2.39e-8 |

A full-field directional derivative of the combined loss on training sample 0
was `1.8026417989e-5`; a central finite difference gave `1.8026417273e-5`
(absolute difference `7.16e-13`). This verifies differentiation of the surface
objective with respect to predicted fields. It does not validate total
derivatives of a coupled MDO problem, shape derivatives of a CFD solver, or
the adapted network's eventual accuracy.

The public runtime tests skip their real-asset comparison if the optional
dataset/upstream checkout is absent; a skip is not an equivalence result.
