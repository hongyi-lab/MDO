#!/usr/bin/env bash
# Linux host: pulls official MACH-Aero and a small, pinned wing volume mesh.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
command -v docker >/dev/null || { echo 'Docker Engine is required; see docs/CFD.md.' >&2; exit 1; }
command -v curl >/dev/null
command -v sha256sum >/dev/null
image_tag="${MDO_CFD_IMAGE:-mdolab/public:u22-gcc-ompi-stable-amd64}"
docker pull "$image_tag"
image_digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$image_tag")"
test -n "$image_digest"
mkdir -p results/cfd_bootstrap data/cfd
printf 'MDO_CFD_IMAGE_DIGEST=%s\n' "$image_digest" > results/cfd_bootstrap/image.env
mesh_path='data/cfd/mach_wing_L3.cgns'
mesh_sha='72d6d58cf9cbb086a6920dfc67713e31b35991a28c3cd6b96b7392e120d7915b'
if ! printf '%s  %s\n' "$mesh_sha" "$mesh_path" | sha256sum --check --status 2>/dev/null; then
  curl --fail --location --retry 3 \
    'https://raw.githubusercontent.com/mdolab/MACH-Aero/47545f536d41bd8075ee7e7dd750fb078edffe63/tutorial/wing/meshing/volume/wing_vol_L3.cgns' \
    --output "${mesh_path}.download"
  printf '%s  %s\n' "$mesh_sha" "${mesh_path}.download" | sha256sum --check
  mv -- "${mesh_path}.download" "$mesh_path"
fi
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$task_root,target=/home/mdolabuser/mount" \
  --workdir /home/mdolabuser/mount \
  "$image_digest" /bin/bash -lc \
  'source "${BASHRC_MDOLAB:?Official image must provide BASHRC_MDOLAB}" && python -c "import adflow, baseclasses, pyhyp, cgnsutilities; from mpi4py import MPI; print(\"MACH-Aero imports passed\")"' \
  | tee results/cfd_bootstrap/imports.log
printf '\nImage pinned locally: %s\nMesh verified: %s\nFollow docs/CFD.md to run the 8-rank CPU smoke case.\n' "$image_digest" "$mesh_path"
