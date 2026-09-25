"""Read-only environment inventory. It does not install or change drivers."""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command):
    if shutil.which(command[0]) is None:
        return {"available": False}
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"available": True, "returncode": proc.returncode,
                "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": str(exc)}


def inspect_environment(directory="."):
    root = Path(__file__).resolve().parents[2]
    for name in ("floGen", "cfdpost"):
        upstream = root / "external" / name
        if upstream.is_dir() and str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
    disk = shutil.disk_usage(Path(directory).resolve())
    result = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "logical_cpus": os.cpu_count(),
              "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "disk_free_gib": disk.free / 1024 ** 3,
              "modules": {name: importlib.util.find_spec(name) is not None
                          for name in ("numpy", "torch", "flowvae", "cfdpost", "adflow", "mpi4py")},
              "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
              "mpi": command_output(["mpirun", "--version"]),
              "lscpu": command_output(["lscpu"])}
    if result["modules"]["torch"]:
        try:
            import torch
            result["torch"] = {"version": torch.__version__, "built_cuda": torch.version.cuda,
                               "cuda_available": torch.cuda.is_available(),
                               "gpu_count": torch.cuda.device_count()}
            result["torch"]["cpu_threads"] = torch.get_num_threads()
        except Exception as exc:
            result["torch"] = {"error": str(exc)}
    return result
