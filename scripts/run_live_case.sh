#!/usr/bin/env bash
# A NEW explicitly defined shared-geometry CFD/FM case, not a paper reproduction.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
device="${1:-cuda:0}"
ranks="${2:-8}"
layers="${3:-81}"
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || { echo 'MPI ranks must be a positive integer' >&2; exit 2; }
[[ "$layers" =~ ^[0-9]+$ ]] && (( layers >= 9 && (layers - 1) % 8 == 0 )) || { echo 'Layers must be 8*k+1, at least 9' >&2; exit 2; }
test -x .venv/bin/python || { echo 'Run bash scripts/bootstrap_server.sh cuda first' >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
.venv/bin/python -m pip install 'git+https://github.com/YangYunjia/cst-modeling3d.git@9500b19a732463c26b32f6860ac6c517c7d212a5'
if [ ! -f assets/AeroTransformer/ATsurf_S/best_model_weights ]; then
  .venv/bin/python scripts/fetch_assets.py --samples 8 --model ATsurf_S
fi
if [ ! -f results/cfd_bootstrap/image.env ]; then
  bash scripts/bootstrap_cfd.sh
fi
source results/cfd_bootstrap/image.env
run_dir="results/live_case_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/run_matched_cfd.py prepare \
  --input-json configs/new_crm_like_case.json --output "$run_dir/bundle" --allow-reconstructed-case
.venv/bin/python scripts/run_matched_cfd.py predict \
  --request "$run_dir/bundle/request.json" --checkpoint "${MDO_CHECKPOINT:-assets/AeroTransformer/ATsurf_S}" \
  --device "$device" --output "$run_dir/fm_prediction.json"
docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$task_root,target=/home/mdolabuser/mount" \
  --workdir /home/mdolabuser/mount \
  --env "MDO_CFD_IMAGE_DIGEST=$MDO_CFD_IMAGE_DIGEST" --env "MDO_CASE_DIR=$run_dir" \
  --env "MDO_MPI_RANKS=$ranks" --env "MDO_MESH_LAYERS=$layers" \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  "$MDO_CFD_IMAGE_DIGEST" /bin/bash -lc \
  'source "${BASHRC_MDOLAB:?}" && mpirun -np "$MDO_MPI_RANKS" python scripts/run_matched_cfd.py run --request "$MDO_CASE_DIR/bundle/request.json" --output "$MDO_CASE_DIR/cfd" --wall-normal-layers "$MDO_MESH_LAYERS"' \
  2>&1 | tee "$run_dir/cfd.log"
.venv/bin/python scripts/run_matched_cfd.py compare \
  --prediction "$run_dir/fm_prediction.json" --cfd-result "$run_dir/cfd/result.json" \
  --output "$run_dir/paired_diagnostic.json"
.venv/bin/python -m mdo_demo doctor --output "$run_dir/environment.json"
printf '\nPaired diagnostic saved in %s. This new case is not calibrated; equal-accuracy speedup is not yet established.\n' "$run_dir"
