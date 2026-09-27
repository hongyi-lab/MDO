# Native 0-degree CFD: bounded late-NK recovery

Recorded 2026-09-28 Australia/Sydney; event times below are UTC.

## Completed attempt and diagnosis

The initial one-way aero-structural queue stopped at its first CFD point at
2026-09-27 13:42:12 UTC. No optimization candidate or final MDO comparison was
produced. Raw files remain preserved locally under `results/overnight_recovery_v2`.

| Quantity | Observed value |
|---|---:|
| Native angle of attack | 0 degrees |
| Mach / Reynolds number | approximately 0.8 / 20 million |
| Major / internal iterations | 133 / 3046 |
| Configured internal budget | 3000 |
| Final residual / free-stream residual | 6.6368866993e-8 |
| Required relative residual | 1e-10 |
| CFD solve wall time | 3041.776 seconds |
| Entire failed optimization stage | 3060.584 seconds (51.01 minutes) |
| MPI ranks | 8 |
| Fatal failure / finite coefficients | false / true |

The stopping reason was exhausted internal iterations without convergence.
Residual reduction and finite coefficients are progress, not acceptance. The
ADflow iteration log repeatedly showed small NK steps and linear tolerance
near 0.79. The [pinned official solver guide](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/doc/solvers.rst)
identifies this pattern as a possible premature NK switch. This supports a
targeted recovery experiment; it does not prove the root cause.

## Versioned repair

The old recipe and v1 protocol remain unchanged. The new
`configs/aerostructural_pilot_late_nk_v2.json` records one common CFD policy for
optimization and all final design verifications:

| Setting | Previous run | New recovery |
|---|---:|---:|
| NKSwitchTol | 1e-5 | 1e-7 |
| Internal iteration budget | 3000 | 9000 |
| ANKSecondOrdSwitchTol | 1e-3 | 1e-3 |
| Relative convergence requirement | 1e-10 | 1e-10 |
| MPI ranks / threads per rank | 8 / 1 | 8 / 1 |
| Save full volume solution | no | yes, double precision |

9000 is our explicit diagnostic budget, not an ADflow recommendation or a
guarantee of success. The line search, geometry, mesh, RANS/SA equations, flow
conditions and MDO design task are unchanged. The fresh solve starts from
free-stream conditions. It does not pretend to restart from the old surface
file. Volume output is saved for possible later, separately verified restart
use; no restart logic is silently activated.

Protocol SHA256:
`ae0270394e2589090279b04ddcc7e1c133f7e6bc7272d91115a70dee250c89cf`.

## Checks actually performed

- Local tests: 154 run, 153 passed, one actual-TACS test skipped locally.
- Server model environment: 154 run, 152 passed; actual TACS and the optional
  local training-asset integration test unavailable in that environment.
- Official structural environment: all 12 structural tests passed, including
  real TACS load scaling, mass, stiffness and root-clamp checks.
- Same-protocol FM-to-TACS initial point completed. Its displacement still
  exceeds the frozen limit, so it remains correctly marked infeasible.
- Reintegrating the failed 0-degree CFD surface through the common native
  interface passed: absolute CL difference 6.86e-7, CD difference 2.89e-7,
  largest moment-coefficient difference 1.10e-6. These are integration checks,
  not a converged CFD reference or model accuracy result.
- Installed ADflow source confirms the new output/solver options exist.
- Invalid budgets are rejected before MPI/mesh work; tests ensure the repair
  does not relax acceptance, change physical identity or expand MPI allocation.

The first server test attempt exposed stale test files; the current test set
was synced after backup. The standalone TACS test invocation then needed its
explicit source import path. Both initial logs are retained along with the
subsequent successful checks. Neither issue explains CFD convergence.

Updated build receipt SHA256:
`c4d3206339441750b7bddab77face7dc07c3ccb833c1642e3cadb308ffaee2fe`.

## New background run and result limits

Run `oneway_aerostructural_late_nk_v2_20260927T144004Z` started at
14:40:04 UTC. Startup was checked: the actual solver printed the 9000 budget,
the delayed NK threshold and its first ANK iteration. It is running, not yet
converged. It retains the six-stage bounded queue: CFD optimization, original
and adapted FM optimization, then common ADflow/TACS verification for all three
candidates. A failed physics stage blocks subsequent stages.

Before starting, shared CPU load was about 5.84, available RAM about 55 GiB,
and the project lock was free. Other agents' jobs were untouched. This is a
shared machine, so contention must be disclosed in timing comparisons.

Final reporting must inspect actual solver options/ranks as well as protocol
hashes, retain failed/development costs, and compare jointly verified mass,
constraints, feasibility and complete optimization plus verification time.
There is still no qualified MDO speedup or final optimized-design comparison.
The scope remains one-way, fixed-planform wingbox sizing.
