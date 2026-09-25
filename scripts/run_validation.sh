#!/usr/bin/env bash
# The complete offline accuracy + adaptation + calibration experiment.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
device="${1:-cuda:0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
test -x .venv/bin/python || { echo 'Run bash scripts/bootstrap_server.sh cuda first' >&2; exit 2; }
run_dir="results/validation_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python scripts/fetch_assets.py --samples 72 --data-dir assets/CRMpert72 --model ATsurf_S
.venv/bin/python scripts/run_validation.py --device "$device" --output "$run_dir"
.venv/bin/python -m mdo_demo doctor --output "$run_dir/environment.json"
.venv/bin/python -m pip freeze > "$run_dir/environment.freeze.txt"
printf '\nOpen %s/report.html after copying the result folder to your laptop.\n' "$run_dir"
