# Fixed converged seed: first new-angle validation

This is one bounded native CFD diagnostic, before restoring the fair
aero-structural optimization queue. It changes the initialization strategy, not
the physical design task, acceptance threshold or available CPU ranks.

## Why this is a new interface

The accepted 0-degree CFD result and its common TACS initial analysis are recorded
in `CFD_CHECKPOINT_V3_RESULT.json`. A new angle is a new physical case. The old
same-case restart validator correctly refuses that use; it has not been bypassed.
The new `cfd_seed.py` contract allows only the target angle to change. It verifies
the qualified seed, its original restart audit and parent hashes, checkpoint,
grid/field audit, geometry, Mach/Re/T, reference quantities and integration groups.

The pinned source audit in `CFD_FIXED_SEED_SOURCE_AUDIT.md` establishes that a
fresh ADflow/AeroProblem applies the target angle and boundary data, reads the
old flow as an initial guess, and independently calculates the target's uniform
free-stream residual. No geometry or saved-state rotation is performed. The
adapter explicitly disables `infChangeCorrection`, matching the effective
fresh-AeroProblem file-load behavior and preventing in-memory correction reuse.

Acceptance uses the new target's own residual denominator. It neither divides
by the seed's residual nor requires that the target denominator equal the seed's
value. Actual solved angle, finite coefficients, flags, solver options, MPI
allocation, and final full-volume fields must pass. None of these checks proves
mesh independence or a complete MDO result.

## Frozen one-point experiment

Protocol: `configs/cfd_seed_alpha0p2_probe_v1.json`.
File SHA256: `c3e8062ca8a01738633a1e275fc3c4ce6fe7378cef09318f1a9cb9faea18dc31`.

- Nominal next COBYLA proposal: +0.2 degrees, from rhobeg 0.1 and angle scale 2.
- Actual stored angle: 0.20000000298023224 degrees, due to the existing bundle's
  float32 condition representation. The exact stored value is used and hashed.
- Target bundle: `f8bc0259308cc72836467191c06916fb9446be43ad9aa73f0d19771bc4f5c68c`.
- Fixed 0-degree seed checkpoint:
  `a9f054f11a90672888caabd8ac1904ff7aba075461a12aefbdf70feedf016bc3`.
- Original volume mesh:
  `3785e33042e68d992fcea36752ae5b408f2ad8fc2370c69e987a3666905611d3`.
- ANK-only, 1200 internal-iteration budget, 8 MPI ranks, one thread per rank.
- RANS/SA, same 81-layer mesh and physical parameters; residual threshold 1e-10
  and relative-to-start option 1e-16 remain unchanged.
- Additional full-volume grid/state audit runs on a converged new result.
- No automatic retries, training or automatic complete MDO queue launch.

## Measured-cost contract

The seed cost ledger is bound to the seed result content and checkpoint hashes.
It preserves the actual **9652.238474 seconds** of source-stage plus refinement
process time. The new-angle process is timed independently. Preparation is not
added on every aerodynamic call and is not treated as free.

Any subsequent common numerical policy must use the same immutable seed for
both optimization routes' CFD evaluations and final checks. Report online
optimization/verification costs and shared preparation separately. Apply the
same preparation accounting rule to both end-to-end routes; count it once in
the project aggregate. Earlier independent numerical development, original mesh
generation and model training retain their separate cost records.

## Checks performed before launch

- Local scientific environment: 173 tests, 159 passed, 14 optional-runtime skips.
- Server model environment: 173 tests, 171 passed, 2 optional-runtime skips.
- Official CFD environment: all 8 new seed-contract tests passed.
- Real checkpoint, parent convergence audit, target request, native geometry,
  saved grid/state audit and measured preparation ledger passed preflight inside
  the actual CFD path namespace. No solver ran during that preflight.

Tests include changed Mach/reference/geometry, wrong checkpoint or missing SA
state, unqualified seed, wrong solved angle/options/ranks, and shared-cost errors.
A target with a different free-stream denominator and a large initial residual
is intentionally allowed if its own final residual and physical checks pass.

## Background execution and next gate

Run `seed_alpha0p2_ank_v1_20260927T204129Z` was launched at 20:41:29 UTC.
Its authoritative pointer is `manifests/active_seed_initialization.json`; the
older aerostructural and checkpoint pointers describe ended runs. The host
launcher holds the project lock, checks shared CPU/memory margin and leaves all
other agents' tasks untouched. Source files were backed up and hash-checked
before deployment; no running project process was changed.

At launch, the result is pending. A supported warm-start API and passing unit
tests are not evidence that this target converges. If successful, inspect the
actual target identity, residual history, final field audit and force-to-TACS
path before defining a new common optimization protocol and build receipt.
Do not relabel the old protocol or reuse a previous receipt after code changes.
If unsuccessful, preserve its evidence and diagnose the new failure rather than
repeating the same configuration or silently relaxing the criterion.

The primary result is still jointly verified optimized mass, constraint margins
and complete optimization-plus-verification cost. The present scope remains
one-way fixed-planform wingbox sizing. No complete optimized-design comparison
or equal-quality MDO speedup has been obtained yet.
