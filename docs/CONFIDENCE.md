# CFD surrogate calibration and fallback

The confidence gate answers a limited question: are calibrated CL/CD intervals
small enough for the chosen tolerances, with this query inside the empirical
training support? It does **not** guarantee that a particular answer is correct.
It does not certify surface pressure, structural loads, buffet or gradients.

## Data separation

`split_geometry_groups` assigns entire geometries to training, calibration and
test. Multiple conditions of the same geometry remain together. Input ordering
and repeated geometry IDs do not change the assignment. The split fingerprint
records the exact groups and seed.

Only training geometries may fit the adaptation, select its hyperparameters or
construct the support envelope. Freeze the model before examining calibration
errors. Calibration is not another hyperparameter-selection set. Keep the test
set untouched until the final comparison. This new split cannot establish that
an upstream pretrained model never saw the same geometries; check and report
the original model's training provenance separately.

## Calibration rule

For each calibration case, calculate

```
score = max(abs(CL_prediction - CL_reference) / tolerance_CL,
            abs(CD_prediction - CD_reference) / tolerance_CD)
```

Take the maximum score over all sampled conditions of each geometry. With `n`
calibration geometries, select the sorted group score at one-based rank
`ceil((n + 1) * (1 - alpha))`. If that rank exceeds `n`, there is no finite
calibrated interval and the gate requests CFD for every query. Thus 90% nominal
coverage requires at least 9 independent calibration geometries; 95% requires
at least 19. Hundreds of conditions from 8 geometries still count as 8 groups.

The resulting symmetric interval half widths are `q * tolerance_CL` and
`q * tolerance_CD`. CL/CD are calibrated simultaneously. CM remains diagnostic
until the moment-reference convention has been established consistently; Cp
field errors remain separate diagnostics.

Coverage is **marginal across exchangeable geometry groups**, assuming the same
geometry/condition sampling protocol. It is not conditional coverage for a
specific shape, Mach number or the subset accepted by the gate. Changing the
number or selection of conditions per group can change the score distribution.
With a few held-out geometries, an observed coverage percentage is only a small
sample audit, not reliability certification.

The defaults `CL=0.01`, `CD=0.0005` (5 drag counts) are provisional engineering
settings, not published acceptance standards. Choose and record tolerances
before examining held-out errors.

## Inference decision

`assess_prediction` never reads `reference_coefficients`, `field_errors` or
held-out error metrics. It accepts the surrogate only when all checks pass:

1. Model, dataset and split identities match the calibration artifact.
2. Inputs, coefficients and descriptors are finite and the artifact is intact.
3. A finite calibrated interval exists and both half widths are within tolerance.
4. Mach, angle of attack and every geometry descriptor lie within the envelope
   constructed from training geometries only.
5. This is not reuse of a calibration geometry as an independent inference test.

`describe_geometry` provides fixed-axis extents and normalized covariance
eigenvalues. Their axis-aligned min/max envelope is only a coarse support
screen. It cannot detect every out-of-distribution shape or every flow regime
change. Do not interpret passing the envelope as proof of in-distribution flow.

Missing information, changed weights or incompatible support cause fallback.
The current condition schema accepts only angle of attack and Mach. A supplied
additional variable such as Reynolds number is rejected rather than silently
ignored: varying it requires a model/data/calibration protocol that supports it.
If `q > 1`, all queries fall back, including test cases whose true errors happen
to be small. The implementation intentionally does not use test truth to rescue
these decisions.

`fallback_required=True` is a request to the caller, **not evidence that a CFD
solver ran**. An offline audit of published CFD labels is not a live fallback
execution and provides no matched CFD speedup measurement.

## Minimal API

```python
from mdo_demo.confidence import (
    split_geometry_groups, fit_calibration, assess_prediction, evaluate_heldout,
)

split = split_geometry_groups(shape_ids, seed=42,
                              counts={"train": 24, "calibration": 24, "test": 24})
identity = {"model": checkpoint_sha256, "dataset": manifest_sha256,
            "split": split["fingerprint"]}

artifact = fit_calibration(
    calibration_cases, identity, {"CL": 0.01, "CD": 0.0005}, alpha=0.1,
    split=split, support_cases=training_cases,
)

# Every query includes prediction_identity=identity, shape_id, coefficients,
# condition={"alpha_deg": ..., "mach": ...}, and geometry_descriptors.
decision = assess_prediction(query, artifact)

# Test truth is consulted only after inference decisions, to audit their errors.
audit = evaluate_heldout(test_cases, artifact)
```

Calibration cases additionally need `sample_id` and `reference_coefficients`.
Support cases need only `shape_id`, conditions and geometry descriptors. Every
planned calibration geometry must be represented. Evaluation rejects shapes
outside the test partition and reports if only part of that partition was run.

The held-out audit reports simultaneous CL/CD case coverage, geometry-group
coverage, acceptance fraction and observed unsafe acceptances. With no accepted
cases, the accepted-subset failure fraction is `null`, not a misleading zero.
