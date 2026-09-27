# ADflow fixed-seed warm-start source audit

Date: 2026-09-28. Reviewed ADflow revision: `8155e98119ec138f805a916400837cd27c41b961`.

## Scope and conclusion

A volume checkpoint obtained at native angle of attack 0 degrees can initialize a new solve at native angle of attack +0.2 degrees while retaining the same geometry, volume grid, Mach number, Reynolds number, temperature, reference quantities, steady RANS equations, and SA turbulence model. The checkpoint supplies an initial flow state; the new `AeroProblem` specifies the target boundary conditions. This is a different-condition warm start, not continuation of the identical physical case.

This conclusion concerns the supported API and its reviewed implementation. It does **not** establish that the new target solve has run, converged, or achieved any accuracy or speedup.

## Reviewed source path

All links below refer to the exact revision above.

| Question | Source evidence | Interpretation |
|---|---|---|
| May a restart use a different angle? | [`initializeFlow.F90`, lines 2803-2811](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/initFlow/initializeFlow.F90#L2803-L2811) | The restart routine explicitly documents changing boundary conditions for angle-of-attack and Mach sweeps. This protocol permits only the angle change. |
| Does the target AP apply before reading the checkpoint? | [`pyADflow.py`, line 3373](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L3373) and [lines 3397-3402](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L3397-L3402) | `_setAeroProblemData` runs before the checkpoint state is loaded. |
| Where is the target angle assigned? | [`pyADflow.py`, lines 3467-3471](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L3467-L3471) and [lines 3574-3579](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L3574-L3579) | The angle comes from the new AP, is converted to radians, and updates the inflow direction. |
| Are the reference state and farfield refreshed? | [`pyADflow.py`, lines 3687-3693](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L3687-L3693); [`initializeFlow.F90`, lines 319-340](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/initFlow/initializeFlow.F90#L319-L340) and [lines 108-110](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/initFlow/initializeFlow.F90#L108-L110); [`BCRoutines.F90`, lines 1314-1319](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/solver/BCRoutines.F90#L1314-L1319) | Boundary-data refresh rebuilds the reference state. Farfield velocities use the current target `wInf`, rather than the source checkpoint angle. |
| Does checkpoint reference data override the target angle? | [`variableReading.F90`, lines 2341-2357](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/initFlow/variableReading.F90#L2341-L2357) and [lines 2627-2631](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/initFlow/variableReading.F90#L2627-L2631) | The reviewed reference-data path rescales density, pressure, velocity magnitude units and viscosity. It does not replace the target AP angle. |
| Which residual normalizes convergence? | [`NKSolvers.F90`, line 70](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/NKSolver/NKSolvers.F90#L70); [`solvers.F90`, lines 968-974](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/solver/solvers.F90#L968-L974); [`NKSolvers.F90`, lines 280-331](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/src/NKSolver/NKSolvers.F90#L280-L331) | A fresh process starts with no cached free-stream residual. ADflow temporarily saves the checkpoint, sets the cells to the target free stream, calculates its residual, then restores the checkpoint. The target denominator is independent of the source flow-state residual. |

## Frozen initialization behavior

Each target is launched in a **fresh ADflow process with a fresh AeroProblem**. Do not reuse an in-memory solver whose free-stream residual cache may refer to another target.

No geometry rotation or manual flow-field rotation is applied. The source checkpoint is loaded as an initial guess and reconverged under the target farfield conditions.

The reviewed Python option table has no `restartAngleInflow` option. Related options are `infChangeCorrection=True`, `infChangeCorrectionTol=1e-12`, and `infChangeCorrectionType="offset"` by default; `"rotate"` is another supported correction type. See [`pyADflow.py`, lines 5704-5706](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L5704-L5706).

That correction is applied only when `oldWinf` exists ([lines 3416-3421](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L3416-L3421)). A fresh AP initializes `oldWinf=None` ([lines 6814-6822](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py#L6814-L6822)), and the reviewed checkpoint-load path does not populate it. This protocol explicitly chooses **`infChangeCorrection=False`**, equivalent to the effective behavior of this fresh-AP file-restart path. This choice makes the absence of a flow-field correction explicit and prevents accidental dependence on future in-memory AP reuse.

## Acceptance and provenance requirements

- Validate the unchanged volume grid and geometry, checkpoint integrity and required state fields; verify that only angle differs between source and target physical definitions.
- Record the fixed seed's source case identity and checkpoint SHA256 separately from the target case identity. Retain the exact target angle used by the implementation, including any previously frozen float32 representation.
- Keep the target residual requirement at `1e-10`. Normalize by the newly computed **target** free-stream residual, not by the source 0-degree denominator and not by the loaded checkpoint residual.
- Preserve `L2ConvergenceRel=1e-16`, the frozen residual scaling and physical/numerical policy. Require actual solver success, finite outputs, verified target angle and achieved convergence before accepting a reference.
- Record seed-generation cost and target setup, solve and output costs separately. A warm-start refinement time alone is not a cold-start CFD timing. Downstream total-cost comparisons must state and consistently account for the shared seed cost.
- Source-code support for warm starts is not evidence that a particular target has converged or that a complete MDO comparison has finished.

