# Real CFD on the Linux server

The CFD path is **ADflow on CPU/MPI**. The A6000 is used by the learned model;
no GPU is requested by these CFD commands. Windows can validate requests and run
unit tests, but a successful Windows check is not evidence that a CFD solve ran.

There are two distinct experiments:

1. **Installation/timing smoke case:** a small public MACH-Aero tutorial wing mesh.
   It checks the real solver and CPU resources. It is not CRMpert, and its runtime
   cannot be divided by CRMpert inference time to claim matched acceleration.
2. **CRMpert truth check:** an exact matching CRMpert volume mesh, geometry,
   flow condition, reference quantities and coefficient definition. The generic
   adapter supports it; the missing input-geometry provenance described below
   must be resolved before claiming end-to-end reproduction.

## First CPU result without hunting for a mesh

Prerequisites: Linux x86_64, Docker Engine accessible to the current user,
`curl`, `sha256sum`, and an Internet connection. Run from the repository root:

```bash
bash scripts/bootstrap_cfd.sh
source results/cfd_bootstrap/image.env
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$PWD,target=/home/mdolabuser/mount" \
  --workdir /home/mdolabuser/mount \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e MDO_CFD_IMAGE_DIGEST="$MDO_CFD_IMAGE_DIGEST" \
  "$MDO_CFD_IMAGE_DIGEST" /bin/bash -lc \
  'source "${BASHRC_MDOLAB:?Official image must provide BASHRC_MDOLAB}" && mpirun -np 8 python scripts/run_cfd.py --tutorial-mesh data/cfd/mach_wing_L3.cgns --output results/cfd_smoke_8.json' \
  > results/cfd_smoke_8.log 2>&1
```

The official image supplies its own Python and compiled ADflow stack. Its
`BASHRC_MDOLAB` setup script must be explicitly sourced for noninteractive
headless execution; a login shell alone does not reliably load that environment.
Do not install the repository's GPU dependencies over that environment. The CLI inserts
the local `src` directory directly, so no package installation is needed for CFD.
The default Docker user is the official `mdolabuser`, not root; if the mount is
not writable, grant that user access to the project output directories on the
host. Do not use a privileged container.

Start with 8 MPI processes on the i9-14900. Then repeat with 16 and 24 only if the
8-process run converges and fits RAM. Each invocation starts a fresh solver;
there is no cached solution pretending to be a CFD execution. Repeated runs
still have OS filesystem cache effects. Report the median of several successful
runs and the full range; do not call ranks physical cores on the hybrid CPU.

The smoke case uses the official L3 wing mesh, Mach 0.8, alpha 1.5 degrees,
Re = 20 million based on a 1 m reference length, and 300 K. Reference area is
45.5 m² and chord 3.25 m. **The original tutorial uses altitude=10000 m instead
of this explicit Re/T condition**, so its published coefficient numbers are not
an exact acceptance target. This deliberate smoke-case variation is recorded
in the result provenance. It must not be used as a CRM baseline result.

`bootstrap_cfd.sh` fetches the 5.48 MB mesh from MACH-Aero revision
`47545f536d41bd8075ee7e7dd750fb078edffe63` and verifies SHA256
`72d6d58cf9cbb086a6920dfc67713e31b35991a28c3cd6b96b7392e120d7915b`.
The initially selected Docker tag is mutable; the bootstrap records its actual
RepoDigest in `results/cfd_bootstrap/image.env`, and the run commands use that
immutable digest. Keep this file with experiment results. To reuse a known
image, set `MDO_CFD_IMAGE=mdolab/public@sha256:...` before bootstrapping.

## Explicit case request

Create `request.json` beside your matching volume mesh. Replace the illustrative
values with the exact case metadata; the adapter intentionally supplies no
default reference area, Reynolds number or incidence.

```json
{
  "schema_version": 1,
  "problem_id": "crmpert-exact-case-v1",
  "geometry_id": "shape-0",
  "mesh_path": "wing_vol.cgns",
  "condition": {
    "mach": 0.85,
    "alpha_deg": 2.1837,
    "reynolds": 20000000,
    "reynolds_length_m": 1.0,
    "temperature_k": 300.0
  },
  "reference": {
    "area_m2": 1.5,
    "chord_m": 1.0,
    "moment_center_m": [0.25, 0.0, 0.0]
  },
  "numerics": {
    "preset": "aerotransformer",
    "l2_convergence": 1e-10,
    "max_cycles": 3000
  },
  "provenance": {
    "note": "EXAMPLE ONLY: replace with exact geometry, area and matched sample metadata"
  }
}
```

`geometry_sha256` is an optional top-level SHA256 of a shared geometry artifact
whose relation to both the model surface and CFD mesh has been verified.
Leaving it absent is allowed for a standalone CFD run, but a geometry label
alone is not evidence of matching geometry. A volume-mesh hash is always
computed. Relative mesh paths resolve beside `request.json`, not beside the CLI.

```bash
# Runs on Windows/Linux without importing MPI or ADflow:
python scripts/run_cfd.py --input request.json --output validated.json --validate-only

# Inside the MACH-Aero container, with the same mount as above:
mpirun -np 8 python scripts/run_cfd.py --input request.json --output results/cfd.json
```

For fixed lift, replace `alpha_deg` with `target_cl` and `alpha_initial_deg`.
Optional `numerics.trim_tolerance` defaults to 1e-4 and `trim_max_iterations` to
20. This invokes ADflow's `solveCL`, which searches incidence with repeated CFD
solves. Both final lift error and final flow residual must pass. **A fixed-lift
result is not comparable to an arbitrary fixed-incidence model query.** The
published AeroTransformer training path uses incidence and Mach as conditions;
fixed-incidence mode is the natural first paired test.

## Result and failure contract

Exit code 0 means all convergence checks passed; 2 means a completed but
unconverged solve; 1 means a validation/import/solver error. Errors are written
to the result JSON when possible. MPI failures abort the job instead of waiting
indefinitely on another rank. Before solving, an earlier result is invalidated
with `status: running`; a hard process crash/OOM can leave that marker or no final
JSON. The shell exit code and log must also be checked.

Only accept a CFD truth value when `status == "ok"` and
`convergence.converged == true`. In particular, exhausting `nCycles` does not
mean convergence. The adapter requires finite coefficients, no ADflow fatal or
routine failure flag, and `final_residual / freestream_residual <= tolerance`.
Trim additionally requires `solveCL` success and the requested lift tolerance.

The JSON includes:

- `identity`: condition, reference quantities, equations and volume-mesh hash;
  `case_sha256` fingerprints those fields. Numerics are separately recorded.
- `coefficients`: `CL`, `CD`, `CMx`, `CMy`, `CMz`; `CM` is explicitly an alias of
  `CMz` for the upstream span-along-z convention, not a universal pitch axis.
- `solved_alpha_deg`, residuals, failure flags, last-solve major iterations and
  trim iterations. Major counts exclude the initial history row; they are not
  ADflow's accumulated minor-iteration count. `trim_total_major_iterations`
  sums all CFD histories returned by `solveCL`, when available.
- `memory`: Linux `getrusage(RUSAGE_SELF).ru_maxrss` for every MPI rank, converted
  from KiB to MiB, with the maximum rank peak and sum of rank peaks. These are
  process-lifetime high-water marks. **Their sum is not simultaneous total job
  peak memory**, and shared pages may be counted in multiple ranks. External MPI
  processes and other jobs are excluded; missing measurements stay null.
- `timing`: setup, solve, total wall time, MPI processes and allocated rank-hours.
  Total includes imports/setup, solving, surface output and coefficient extraction,
  but excludes mesh generation, request hashing and process/container launch.
  Solve-only includes ADflow's surface-file writing. Report the scope explicitly.
- Solver version, available Git revision, Docker digest, options, host and
  surface-output directory. No pressure/friction fields are fabricated in JSON;
  ADflow writes its real surface files there for subsequent matching/postprocessing.

No equivalence between CPU rank-hours and GPU-hours is assumed. Record those
resource costs separately, and count mesh/data generation, training, inference,
correction and final verification for the eventual total-cost comparison.

## CRMpert mesh reconstruction: verified capability and missing asset

The official AeroTransformer repository at
`3dc350ff69e354d1451eae468c686a368bd6151f` contains
`simulation/gen-mesh.crmpert.py`, `original_tip.xyz`, `run-adflow.py` and
`single-point.py`. Its mesher uses pyHyp, CGNS utilities and `cst_modeling`.
However, neither that repository nor the public CRMpert file listing supplies
a complete mesher `input.json` or the original CGNS volume meshes.

The published `samples.parquet` includes perturbed CST coefficients, twist and
dihedral information, but its documented columns do not give all of the
mesher's fixed planform constants (`SA` and the required `chords` array).
Do not invent those constants or label a fitted reconstruction an exact
reproduction. A supplied original `input.json` must contain at least `SA`,
`half_span`, `chords`, `twists`, `DAz`, `cst_u` and `cst_l`, in the upstream
mesher's conventions. In particular, its seven control sections and cumulative
twist convention must be preserved. Then:

1. Use the pinned official source in `external/AeroTransformer` and install its
   meshing dependency in a dedicated MACH-Aero image, e.g. the reviewed source
   `git+https://github.com/YangYunjia/cst-modeling3d.git@9500b19a732463c26b32f6860ac6c517c7d212a5`.
   This dependency and mesher have **not** been executed on the local Windows host.
2. Work in a new per-geometry directory with that exact `input.json`. The upstream
   script has a working-directory-relative `../../original_tip.xyz` path; provide
   its supplied tip file at that location. The upstream `single-point.py` expects
   the mesher name `gen-mesh.py`, so invoke `gen-mesh.crmpert.py` directly rather
   than assuming the dispatcher already selects it.
3. Generate the surface on rank 0, then synchronize MPI ranks **before** calling
   `volume_meshing`; the published entry point lacks this explicit barrier.
   For the first mesh use a single process to avoid that race:

   ```bash
   MPLBACKEND=Agg mpirun -np 1 python /path/to/external/AeroTransformer/simulation/gen-mesh.crmpert.py
   ```

4. Read the generated `input.json` for `surface_area`, inspect `mesh_info.json`
   and `volume_meshing.log`, and check positive cell volumes. Save checksums and
   mesh-generation time. Pass `wing_vol.cgns`, the actual area and original
   condition to this adapter. The upstream script fixes Reynolds length and
   chord reference to 1 m and moment x-reference to 0.25 m.

The ready-to-run tutorial route above keeps CFD installation testing independent
of obtaining this missing exact geometry input. An automated exact-CRMpert
meshing claim is intentionally not made by this repository.

## Matching definitions before reporting acceleration

CRMpert metadata distinguishes solver-integrated coefficients (documented
columns 6–8) and coefficients reintegrated from the ML surface (9–11). The model
is evaluated using the latter. Surface interpolation, omitted tip/trailing-edge
surfaces and friction conventions can change drag. Compare field predictions
on the same surface and compare like-defined coefficients first; do not silently
mix these two label sources. A pressure-only integral is not total viscous drag.

There is also a reference discrepancy: the dataset README describes moment
about the leading edge, while the simulation script sets `xRef=0.25` and the
postprocessor also uses 0.25. Prioritize verified `CL`/`CD`; defer `CM` parity
until its axis/reference convention has been confirmed for the selected data.

Official sources inspected for this integration:

- [AeroTransformer simulation source at the pinned revision](https://github.com/tum-pbs/AeroTransformer/tree/3dc350ff69e354d1451eae468c686a368bd6151f/simulation)
- [CRMpert metadata and files](https://huggingface.co/datasets/thuerey-group/CRMpert/tree/main)
- [MACH-Aero Docker installation](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/installInstructions/dockerInstructions.html)
- [Public tutorial wing asset](https://github.com/mdolab/MACH-Aero/blob/47545f536d41bd8075ee7e7dd750fb078edffe63/tutorial/wing/meshing/volume/wing_vol_L3.cgns)
- [ADflow Python API inspected at revision 8155e981](https://github.com/mdolab/adflow/blob/8155e98119ec138f805a916400837cd27c41b961/adflow/pyADflow.py)

The API audit revision is a source reference, **not** a claim that the downloaded
container contains that commit. Every live run records its actual available
ADflow version/revision and image digest.
