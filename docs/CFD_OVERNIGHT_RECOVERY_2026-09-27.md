# Overnight CFD recovery and continuation

## Authorization and scope

On 2026-09-27 the user explicitly authorized diagnosing and fixing a failed CFD
run tonight, then continuing toward a fair comparison for the next morning.
The existing monitor remains every two hours. Do not ask again whether to fix a
failure within this scope. A running task must finish unchanged; deployment or
a new solve requires the old process to have exited and the shared compute lock
to be available. No new model training, dataset-wide CFD sweep or whole-aircraft
optimization is authorized by this recovery plan.

This is a commitment to investigate and act, not a guarantee of overnight
convergence or publication-ready results.

## Current evidence, not a completed result

The active run is recorded in `manifests/active_aerostructural_run.json` on the
server. At 13:29 UTC, the queue `oneway_aerostructural_v1_20260927T1255` was still
in its first `adflow_optimization` point, after about 38 minutes:

- Native angle 0 degrees, Mach approximately 0.8; frozen geometry and mesh.
- 111 major / 2324 internal iterations; final shown total residual 31.3515.
- The shown residual fell from approximately 69 to 31 over the recent steps.
- Several NK steps were 0.01–0.12 and linear relative residuals were near 0.79.
- The result file still said `running`; no convergence or fair-comparison result
  was available at this snapshot.

ADflow's solver documentation identifies small NK steps and stalled
Eisenstat–Walker tolerance as signs that NK may have been enabled before
transients settled. This is a supported hypothesis, not a proven root cause.
Do not interrupt the current run merely because this pattern appears.

Sources: [official solver guide](https://mdolab-adflow.readthedocs-hosted.com/en/latest/solvers.html)
and [pinned solver documentation](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/doc/solvers.rst).
The retrieved exact excerpts are also available locally in the ignored file
`results/baseline_build_v1/adflow_solver_recovery_source_8155e98.txt`
(SHA256 `b31422de3184c989b5f2c9e98a2b24dbdbc4c3f0cff314a14ca605f35f58d40f`).

## If the current point fails

1. Save the final queue/result JSON, logs, convergence history and wall surface
   arrays. Verify exit state and whether the stop was budget exhaustion,
   numerical stagnation/divergence, a nonfinite field or an execution error.
   Distinguish cumulative internal iterations from nonlinear major iterations.
2. Compare the latest residual trajectory and solver stages with the pinned
   documentation. Read the implemented options in `src/mdo_demo/cfd.py` and the
   caller in `src/mdo_demo/matched_cfd.py`; the latter currently fixes 3000
   internal iterations. Do not blame the mesh or installation without evidence.
3. If the final evidence still supports premature NK switching, a candidate
   repair is to delay `NKSwitchTol` from 1e-5 to 1e-7 and permit up to 9000
   internal iterations for the first diagnostic. The 9000 budget is our bounded
   experiment choice, not an official recommendation or a promise it suffices.
   Retain `ANKSecondOrdSwitchTol=1e-3` to avoid changing that setting at the same
   time. Keep the line search and 1e-10 acceptance threshold. Verify
   compatibility with the installed, pinned ADflow source before deployment.
4. Current runs set `writeVolumeSolution=False`. `wing_vol.cgns` is a mesh, and
   the saved surface CGNS is not a restartable full flow state. Do not label a
   fresh solve as continuation. If checkpoint/restart support is introduced,
   verify mesh/case identity, source hashes, solver API and residual
   normalization; account for all stages and startup costs.
5. Make configuration changes explicit, test them, then deploy only after the
   current queue exits. Use a new run directory and retain the failed record.
   Update tested-code hashes and build evidence through actual checks; never
   edit a receipt merely to bypass its validation. Respect `compute.lock`.
6. First obtain one converged, finite CFD result with audited force integration
   and real TACS transfer. Then continue the bounded common-structure experiment.
   A repeated failure requires a new evidence-based diagnosis, not an unchanged
   retry loop or a relaxed numerical threshold.

## Fairness and reporting

Keep physical geometry, flight conditions, RANS/SA equations, mesh, eight MPI
ranks, structural model, starting design, objective/constraints and optimizer
protocol fixed. Apply any accepted CFD numerical policy consistently to the CFD
optimization route and every final verification, and record its version.

The main comparison remains jointly verified mass, constraint margins and
feasibility, with complete optimization and final-verification time. Count
retry/correction costs and separately disclose solver-development experiments,
original mesh generation and model training. Failed CFD time divided by model
latency is not a successful speedup result. This pilot is one-way fixed-planform
aero-structural sizing, not full aeroelastic or whole-aircraft optimization.

When a stage completes, preserve and inspect its raw outputs. Notify only for a
meaningful completion, failure, recovery launch or required input. If recovery
is still incomplete in the morning, report the actual blocker and changes made,
without inventing a fair-comparison table. Figures retain the existing pending
Python/R choice; raw JSON/CSV and a factual text summary can be prepared without
waiting on that choice.

The separate GitHub dependency omission was fixed in `9dfa56f`, and its Core
checks passed. It is not evidence about CFD convergence and does not require
changes to the active server job.
