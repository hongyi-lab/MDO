# Same-case checkpoint recovery: evidence and limits

Recorded 2026-09-28 Sydney / 2026-09-27 UTC. The common MDO comparison is still
incomplete. This recovery is a bounded CFD diagnostic, not a successful speedup.

## New failure, retained without relabelling

`oneway_aerostructural_late_nk_v2_20260927T144004Z` stopped at its first native
0-degree CFD point at 17:19:36 UTC. No subsequent optimization or verification
stage ran. Source phase wall time was 9571.763110 seconds (159.53 minutes).
ADflow's solve clock was 9548.488652 seconds, its total clock 9567.534416 seconds,
and mesh preparation plus CFD 9568.242690 seconds; these are different scopes.

- 286 major iterations; 9001 internal iterations against a 9000 budget.
- Free-stream residual: 142581965.0852572.
- Final residual: 0.01763690854002143.
- Final/free-stream ratio: **1.2369662972084443e-10**, failing **1e-10**.
- Coefficients finite, no fatal solver error, `solve_failed=true`.

The late-NK change substantially reduced residuals, but the last thousands of
internal iterations remained near 0.017. Repeated NK step lengths near 0.00625
and linear relative residuals near 0.8 support investigating stalled NK steps.
They do not prove a mesh defect, installation fault or floating-point floor.
Simply extending the same iteration budget is not the next experiment.

## Why the new experiment can use an actual checkpoint

This failed run saved `wing_000_vol.cgns` in double precision. The original
`wing_vol.cgns` is only a mesh. Both are retained, with the failed run's complete
logs and outputs in the local ignored evidence archive
`results/overnight_recovery_v2/late_nk_v2_failure_evidence.tar.gz`.

Before execution, the independent CGNS C-API audit verified all 13 original zones
and 3,527,680 interior cells. Original and checkpoint grid coordinates differ by
exactly zero; the predeclared absolute tolerance is 1e-10 metres. All six required
interior state fields are finite double precision; density and pressure are
positive. The cell-centred solution and rind are recorded.

Original mesh SHA256:
`3785e33042e68d992fcea36752ae5b408f2ad8fc2370c69e987a3666905611d3`.

Checkpoint SHA256:
`84826a9fd3250fd86c06ad65ff9feb8eed44b30e9971352f86f2620ec3fe1911`.

The adapter separately binds the checkpoint to its source result, case, physical
conditions, geometry, reference quantities, original mesh and output directory.
It uses the actual **0-degree** request, not the original 2-degree base request.

## One bounded change

`checkpoint_ank_v3_20260927T184659Z` launched at 18:46:59 UTC after tests and the
real checkpoint preflight. It starts a new ADflow instance with the original
`gridFile` and the saved full-volume `restartFile`, writing a fresh output folder.
The source files remain unchanged. It is actual restart, not a free-stream rerun.

The new explicit `robust_rans_ank_polish` recipe uses ANK, disables NK, preserves
`ANKSecondOrdSwitchTol=1e-3` and `nSubiterTurb=10`, and explicitly retains the
decoupled default `ANKCoupledSwitchTol=1e-16`. Its additional budget is **1200
internal iterations**. This budget is our bounded diagnostic choice, not an
official guarantee. RANS/SA, geometry, mesh, flow, references and the **1e-10**
criterion remain unchanged. Line-search settings and residual scaling are not
relaxed. Full-volume double-precision output remains enabled.

The source review uses the installed ADflow revision
`8155e98119ec138f805a916400837cd27c41b961`: the
[solver guide](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/doc/solvers.rst),
`pyADflow.py`, `initializeFlow.F90` and `solvers.F90`. Restart is a constructor
option; there is no assumed public `readRestartFile()` Python method. ADflow
recomputes an independent free-stream residual, then the loaded-state residual.
The code never calls `setResNorms` or substitutes a new denominator.

## Acceptance and cost gates

Audit success and solver convergence are distinct flags. Final acceptance needs
all existing solver/finite-value checks and final residual divided by **both**
the current and source free-stream residuals to be at most 1e-10. The current
free-stream denominator must match the source within predeclared relative 1e-8.
The loaded starting residual must match the saved final residual within 1%; this
is only a checkpoint consistency check, accommodating reconstruction roundoff
near convergence, and does not relax either final threshold.

The result records source plus refinement costs. The host manifest additionally
counts the complete new process, including startup, and adds the old 9571.763110
second stage. Restart-only time cannot be reported as cold CFD latency. Earlier
independent failed attempts and model training remain separately disclosed costs.

This job holds `manifests/compute.lock`, uses eight MPI ranks and one OpenMP/BLAS
thread per rank. Launch load was 6.54 on 32 logical CPUs, available memory about
55.7 GiB. Other agents' tasks are untouched; shared-machine timing is recorded.
Only one new CFD solve is authorized by this diagnostic manifest. There is no
automatic restart of the stopped six-stage MDO queue.

## Checks and next gate

- Local scientific Python: 165 tests, 151 passed, 14 optional-runtime skips.
- Server model Python: 165 tests, 163 passed, 2 optional-runtime skips.
- Official CFD environment: all 11 restart-contract tests passed.
- Actual source bundle, checkpoint, CGNS grid/fields and provenance preflight
  passed in the original container path namespace before solving.
- The initial local attempt used a bundled Python without SciPy and failed two
  tests; its log is retained. Tests then passed in the existing scientific
  environment. That environment issue is separate from numerical convergence.

The new process and its mesh initialization were observed after launch. Completion
and the restart normalization audit were **not yet available at this record**.
Follow `manifests/active_checkpoint_refinement.json`, not the ended v2 queue.

If accepted, next audit the actual new fields, common-surface force integration
and TACS transfer. A subsequent optimization queue needs a tested, versioned CFD
policy applied to both routes' final checks, plus a new real build receipt.
Checkpoint success alone cannot be represented as finished MDO. If it fails,
retain the new evidence and diagnose it; do not silently repeat this recipe.

The eventual primary comparison remains jointly verified final wingbox mass,
constraint margins/feasibility and complete optimization-plus-verification cost,
as specified in `CFD_VS_FM_PRIMARY_COMPARISON.md`. The scope is still one-way,
fixed-planform aero-structural sizing, not whole-aircraft design or full
aeroelastic optimization. No model training was repeated.
