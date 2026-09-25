#!/usr/bin/env bash
# Run on Linux. Installs an isolated environment; never changes GPU drivers.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
device="${1:-cuda}"
case "$device" in
  cuda) torch_index='https://download.pytorch.org/whl/cu128' ;;
  cpu) torch_index='https://download.pytorch.org/whl/cpu' ;;
  *) echo 'Usage: bash scripts/bootstrap_server.sh [cuda|cpu]' >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Use Python 3.10–3.13; 3.12 recommended"'
command -v git >/dev/null
if [ ! -x .venv/bin/python ]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.11.0' --index-url "$torch_index"
.venv/bin/python -m pip install -r requirements-model.txt -e .
.venv/bin/python scripts/bootstrap_upstream.py
mkdir -p results
.venv/bin/python -m pip freeze > results/environment.freeze.txt
.venv/bin/python -m mdo_demo doctor --output results/doctor.json
if [ "$device" = cuda ]; then
  .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: inspect results/doctor.json"; print(torch.cuda.get_device_name(0))'
fi
printf '\nReady. Run: source .venv/bin/activate\nThen: python scripts/fetch_assets.py --samples 8 --model ATsurf_S\n'
