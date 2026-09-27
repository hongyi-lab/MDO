# Official CFD runtime without Docker permissions

The September 2026 server does not permit this user to access the Docker socket.
The project therefore runs the unchanged official MACH-Aero image with the host's
existing `bwrap` user namespace support. No `sudo`, Docker group changes, daemon,
host package installation or emulation is required. Container UID 0 maps to the
ordinary host user; it is not host root.

## Recreate the project-owned environment

Layout: `$project/code/MDO` is the checkout; `$project/cfd` holds the runtime and
cases. The host needs Python 3, `bwrap`, `flock`, and working unprivileged user
namespaces. This was checked on Ubuntu 22.04.5, x86-64.

```bash
project="$HOME/Aero/cfd_fm_20260927"
mkdir -p "$project/cfd/packages" "$project/logs" "$project/manifests"
python3 "$project/code/MDO/scripts/pull_cfd_oci.py" \
  --output "$project/cfd/official_oci"
cd "$project/cfd/packages"
apt-get download umoci=0.4.7+ds-2ubuntu0.1
dpkg-deb -x umoci_0.4.7+ds-2ubuntu0.1_amd64.deb ../tools
../tools/usr/bin/umoci unpack --rootless \
  --image "$project/cfd/official_oci:cfd" "$project/cfd/official_bundle"
```

`apt-get download` only downloads the named Ubuntu package. `dpkg-deb -x` extracts
it below this project; neither command installs a host package. The package
version above is the one used for this server, not a promise about future mirrors.

The image is pinned to
`mdolab/public@sha256:75963e625a37d9190b568304732f6970eca2737c200f125b2b2c6b2410bd83a3`.
The downloader verifies the manifest and every config/layer by SHA256 and size.
It changes only Docker/OCI media-type descriptors, retaining all layer bytes.
`provenance.json` records both manifest identities. The image is approximately
3.85 GB compressed, plus its extracted root filesystem. Rootless extraction may
log unsupported extended-attribute warnings; successful unpack/import is checked
separately.

The launcher binds the extracted root filesystem read-only, makes only this
project writable, uses temporary `/tmp`, `/run` and `/dev/shm`, and imports the
official image environment. It limits OpenMP, MKL and OpenBLAS to one thread per
MPI process. The imported official runtime was Python 3.11.14, Open MPI 4.1.6,
ADflow, pyHyp and CGNS 4.3.0 (ADF, 32-bit `cgsize_t`).

## Run a prepared same-geometry case

Create/validate the bundle using [MATCHED_CFD.md](MATCHED_CFD.md). The September
server case uses the explicit new CRM-like planform at alpha 6.71 degrees and
Mach 0.8; it is not a recovered STW or original CRMpert geometry.

```bash
flock -x "$project/manifests/compute.lock" \
  python3 "$project/code/MDO/scripts/run_cfd_rootless.py" \
  --project "$project" -- /bin/bash -lc \
  'source "$BASHRC_MDOLAB" && mpirun -np 8 python scripts/run_matched_cfd.py run \
   --request /home/mdolabuser/mount/cfd/case_alpha6p71_mach0p8/request.json \
   --output /home/mdolabuser/mount/cfd/runs/alpha6p71_mach0p8_L81_new \
   --wall-normal-layers 81' \
  > "$project/logs/cfd_new.log" 2>&1
```

Choose an unused output directory and log. Keep the log and result even if the
case fails. The FM training and timed evaluation driver uses the same exclusive
lock; network-only downloads do not need the lock. No wall-clock timeout replaces
the numerical stopping criterion. Eight MPI processes with one thread each is
the recorded allocation; the first run used Open MPI's default affinity (each
process allowed CPUs 0–31), not eight exclusively pinned physical cores.

## Quality and comparison audit

The mesh recipe has 81 layers and approximately 3.53 million cells. pyHyp's
`computeQualityLayer` uses `MPI_Reduce` to put global minimum volume/quality on
rank zero. The adapter broadcasts those root values and requires both positive.
It must not take a second minimum over uninitialized nonroot values. This exact
adapter bug stopped the first server attempt after meshing, before ADflow; it was
fixed and tested, and the failed attempt is retained as `L81_v1`.

The reviewed installed ADflow commit is
`8155e98119ec138f805a916400837cd27c41b961`. In
`src/solver/surfaceIntegrations.F90`, lines 493–499 explicitly describe Cp
integration for open surfaces, line 523 subtracts `pInf` in the wall force, and
lines 525–526 define `Cp = 2*(p_wall-pInf)/(gammaInf*pInf*MachCoef^2)`.
The source SHA256 is
`786a0c8eb35fc61f4131e60ae3a9f4bcf185caab4f21bdb1c69c8a52bad4836a`.
The result audits this installed-file hash instead of assuming pressure parity
from a wall-family name.

Only CL and CD are primary paired coefficient diagnostics. FM CM remains
diagnostic because the upstream moment convention is unresolved. Mainwing
families exclude tip and blunt trailing edge in both methods, but the native
CFD and resampled FM surfaces still use different discrete integration. Full
field transfer, grid independence, and confidence calibration for this new
geometry are not established. A converged CFD run alone does not establish an
equal-accuracy FM speedup.

The geometry-coordinate audit also identified an author preprocessing frame
rotation (approximately +6.7166 degrees). The currently generated direct FM
input has not yet been reconciled with that transformation. Consequently,
current direct FM/CFD coefficient differences must not be interpreted as model
accuracy. Keep the native CFD result; correct and verify the FM geometry/flow
frame contract before comparing. The native-surface export helper is saved in
`scripts/export_native_cfd_surface.py`; only its PLOT3D center reader has been
checked locally. Its CGNS extraction still needs a real converged output check.

## Follow-up alpha=2 degree case

The independent `case_alpha2p0_mach0p8` case retains the exact native geometry,
Mach, Re, reference quantities and 81-layer mesh recipe. Its case hash is
`3781541896fbe3a8b07f2a64a4fb9d1ff4b8f935d15773e2aba94ec655b30596`.
The live command supports `--reuse-mesh-result <old result.json>` and checks the
source surface hash, physical mesh conditions, mesh recipe, positive quality,
and actual volume-mesh hash before copying the mesh into a fresh output folder.
A failed flow solve does not invalidate a separately verified positive mesh.
New mesh-generation time is zero; checking/copying time is recorded separately,
and previous generation time is retained only as provenance.

`manifests/active_cfd_run.json` on the server points to the current request,
output/result, log, PID and exit-code file. The alpha=2 job uses the same 8 MPI
processes, one thread each, exclusive compute lock, 3000 internal iterations and
1e-10 residual criterion. It is a new attempt, not a replacement for the failed
6.71-degree record. No model job is started by the CFD driver.

Results now include the cumulative internal iteration count, full convergence
history and an evidence-based stop reason. Source Git provenance is taken only
from the explicitly configured ADflow checkout with a verified repository root;
an enclosing Git repository above site-packages is not accepted. This source
revision is not a cryptographic attestation of the compiled binary.

The paired-diagnostics command rejects an explicit unaligned frame/sampling
status. Unknown contracts produce no primary error metric and remain ineligible
for accuracy claims; frame/sampling verification flags are required even for a
primary coefficient diagnostic. Equal-accuracy speedup remains disabled.
