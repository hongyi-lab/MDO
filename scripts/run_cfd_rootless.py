#!/usr/bin/env python3
"""Run the pinned, unpacked official CFD environment with user-namespace bwrap.

No host sudo, Docker socket, package installation, or private-key access occurs.
The project directory alone is writable inside the otherwise read-only image.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    project = args.project.resolve()
    rootfs = project / "cfd" / "official_bundle" / "rootfs"
    oci = project / "cfd" / "official_oci"
    if not (rootfs / "bin" / "bash").exists():
        raise RuntimeError("Official root filesystem is not ready; finish OCI download and rootless unpack")
    # The official image has no default bind mount point. This adds an empty
    # directory only to our project-owned extracted copy, never the host /home.
    (rootfs / "home" / "mdolabuser" / "mount").mkdir(exist_ok=True)
    manifest = json.loads((oci / "source_manifest.json").read_text())
    config = json.loads((oci / "blobs" / "sha256" / manifest["config"]["digest"].split(":")[1]).read_text())
    provenance = json.loads((oci / "provenance.json").read_text())
    env = dict(item.split("=", 1) for item in config["config"]["Env"])
    env.update(HOME="/home/mdolabuser", USER="root", LOGNAME="root", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR="/tmp/matplotlib",
               PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR="/tmp/mdo_numba_cache",
               OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               MDO_CFD_IMAGE_DIGEST=provenance["source_image"], MDO_CFD_EXECUTION="unprivileged_bwrap_native_cpu")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Supply a command after --")
    argv = ["bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "--unshare-pid", "--die-with-parent",
            "--ro-bind", str(rootfs), "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--tmpfs", "/run", "--tmpfs", "/dev/shm", "--ro-bind", "/etc/hosts", "/etc/hosts",
            "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf",
            "--bind", str(project), "/home/mdolabuser/mount", "--chdir", "/home/mdolabuser/mount/code/MDO", "--clearenv"]
    for key, value in env.items():
        argv.extend(["--setenv", key, value])
    argv.extend(command)
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
