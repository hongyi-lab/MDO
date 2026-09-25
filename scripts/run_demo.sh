#!/usr/bin/env bash
# Real checkpoint and real published CFD labels. Does not claim live CFD speedup.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
device="${1:-cuda:0}"
model="${2:-ATsurf_S}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
if [ ! -x .venv/bin/python ]; then
  echo 'First run bash scripts/bootstrap_server.sh cuda' >&2; exit 2
fi
run_dir="results/$(date -u +%Y%m%dT%H%M%SZ)_${model}"
.venv/bin/python scripts/fetch_assets.py --samples 8 --model "$model"
.venv/bin/python -m mdo_demo evaluate \
  --data assets/CRMpert --checkpoint "assets/AeroTransformer/$model" \
  --device "$device" --limit 8 --warmup 2 --output "$run_dir"
.venv/bin/python -m mdo_demo doctor --output "$run_dir/environment.json"
printf '\nOpen %s/report.html locally after copying the directory from the server.\n' "$run_dir"
