# Shared aerodynamic–structural baseline: integration pilot

2026-09-27. The user authorized building and checking a shared baseline before
starting new experiments. No new neural-network training is part of this work.

**Build status:** the common interface and genuine TACS single-point chain passed
acceptance. Local checks: 148 run, 147 passed, one TACS-dependent test skipped;
the actual server TACS mechanics test also passed. Both load paths construct
the same 228-node/240-shell-element structural mesh. Raw build evidence is saved
locally under `results/aerostructural_build_20260927`; the public compact record
is [AEROSTRUCTURAL_BUILD_2026-09-27.json](AEROSTRUCTURAL_BUILD_2026-09-27.json).

The bounded queue `oneway_aerostructural_v1_20260927T1255` was launched at
2026-09-27 12:51:12 UTC and its first ADflow evaluation was observed iterating.
The new initial point is native AoA 0 degrees at the same Mach/Re/T; this is a
lower-incidence start for the newly frozen task. No successful CFD convergence
or optimized/verified design is claimed in this build record. The earlier
0-degree FM/TACS build point has 0.210 m displacement against a 0.150 m limit,
so it is correctly recorded as **infeasible**, despite a successful FE solve.

## What is being built

Both paths receive the same physical wing and flight conditions. ADflow or the
frozen AeroTransformer supplies dimensional surface loads. Both use the exact
same native-surface-derived TACS Quad4Shell wingbox, material, load-transfer rule,
constraints and COBYLA optimizer. Only the aerodynamic backend changes.

This first case is **one-way, fixed-planform aero-structural sizing**. Its design
variables are angle of attack, uniform skin thickness and uniform spar/rib web
thickness. Aerodynamic forces affect feasible structural sizing. Structural
displacement is calculated but does not yet return to the aerodynamic geometry.
It is neither the published STW/XRF1 model nor whole-aircraft optimization.

The protocol in `configs/aerostructural_pilot_v1.json` freezes a new integration
task: minimize half-wingbox mass, CL >= 0.5, CD <= 0.06, discrete KS yield index
<= 1, and tip displacement norm <= 0.15 m. The stated aluminium-like properties
are explicit demonstration assumptions, not certified material allowables.
There is no buckling, flutter, geometric nonlinearity or fuel/inertia model.
These limits are set before new optimization results and apply to both paths.
Before any optimization, thickness upper bounds were set to 8 mm after checking
the actual minimum box depth (18.324 mm); a 20 mm skin would overlap the opposing
skin mid-surface. This geometric correction does not tune a result or relax an
acceptance constraint. The initial 5/4 mm build-check point remains unchanged.

## Physical interface

- Native right-handed coordinates: x streamwise, y vertical, negative z outboard.
  The actual root is at z = -0.3 m; coordinates are not silently shifted.
- Re/T and the gas/viscosity model remain fixed. Dynamic pressure is obtained
  from the actual installed `baseclasses.AeroProblem`, not a guessed density.
- Both paths integrate **mainwing patches 1,2,3,5,6,7 only**. Tip and blunt
  trailing-edge patches are excluded on both sides. This is a declared reduced
  load domain, not a claim of complete aircraft surface coverage.
- FM input geometry and angle are transformed together by +6.7166 degrees.
  Predicted panel forces are returned to the native frame; moment is recomputed
  using physical force application points and the same reference point.
- Native CFD fields must pass the actual convergence test and reintegrate to
  the solver's same-family force/moment coefficients before optimization use.
- Distributed force transfer includes offset couples and checks total force,
  total moment and the associated virtual-work mapping. Passing these identities
  does not establish local mesh or load-transfer convergence.
- The structural mesh is built from the same original `wing.xyz` for both paths,
  never from a different FM-resampled shape on just one branch.

Author pretraining's closed-TE/chordwise/span sampling is still not reproduced.
The new common physical interface exposes this as an FM applicability issue,
not proof of model accuracy. Outputs remain explicitly uncalibrated. No claim
of equal-accuracy acceleration follows from code compatibility alone.

## Build gates before experiment

1. Validate coordinate, force/moment and load-transfer tests, including origin
   changes and physical unit scaling; preserve failed evidence.
2. Import the actual TACS runtime and query the actual gas state. The official
   image requires its MDOLab shell initialization and a writable Numba cache.
3. Execute a genuine TACS single-point integration check using a saved physical
   model prediction. This is build verification, not a new training run or a
   CFD accuracy result.
4. Review the frozen task, executable commands and result rejection paths.
5. Start a bounded new CFD/FM pilot only after the build checks pass. A failed
   CFD solve halts its branch; it cannot be replaced by a finite last iterate.

## Runtime and entry points

`scripts/run_aerostructural.py` accepts `--action single|optimize|verify`,
`--backend adflow|fm`, a native `--request`, `--project` and a fresh `--output`.
FM additionally requires the frozen `--checkpoint`. The host model environment
launches the existing official CFD/TACS image through `run_cfd_rootless.py`.
The entire run holds the existing shared `manifests/compute.lock`.

The first optimization budget is 12 design proposals and at most
6 unique aerodynamic evaluations per path, not a wall-clock kill timer. Exact
same-angle aerodynamic results can be reused while varying only structural
thickness because this is explicitly one-way coupling. Failed CFD/FEA is
recorded as blocked; exhausted declared budgets are recorded separately.

The existing native CFD cases have **not** passed 1e-10. Their failed costs are
not successful latency measurements. A new flow case/solver strategy must be
identified and logged when a new reference attempt starts; there is no blind
automatic loop over the old failure.

`scripts/run_aerostructural_pair.py` requires the successful build receipt and
matching tested-code hashes. It runs the CFD branch, the two frozen FM branches,
then three common final verifications. It stops at the first failed stage and
writes `queue.json`; there are at most six distinct optimization CFD cases and
three additional final verifications. No arbitrary wall-time cutoff or hidden
retry is used. The existing experiment monitor now checks every two hours.

## Fair final comparison

Run both paths from the same starting design. Then `--action verify` re-evaluates
each final candidate with converged ADflow and the same TACS configuration.
`scripts/compare_aerostructural.py` refuses mismatched protocols or missing
common verification and exports JSON + CSV containing checked mass, CL/CD,
yield/displacement constraints, feasibility and measured full wall cost.

Report runtime startup, geometry/input preparation, model loading, prediction,
CFD meshing/reuse/solve, force conversion/transfer, FEA, optimization and final
verification. Model-only milliseconds are not end-to-end optimization time.
Existing mesh-generation and model-training costs are reported separately.
The first small fixed-budget pilot reports quality and cost jointly; it does
not establish matched-quality speedup, global optimality or publication claims.

## References

- [Gray & Martins 2025: STW using MPhys, ADflow and TACS](https://mdolab.engin.umich.edu/bibliography/AGray2025a.html)
- [TACS structural solver](https://smdogroup.github.io/tacs/)
- [ADflow solver strategies](https://mdolab-adflow.readthedocs-hosted.com/en/latest/solvers.html)
- [baseclasses flow state from Reynolds number](https://github.com/mdolab/baseclasses/blob/v1.9.0/baseclasses/problems/pyAero_problem.py#L923)

Implementation and verification status must be read from actual test logs and
run manifests; this document alone is not evidence that the full baseline ran.
