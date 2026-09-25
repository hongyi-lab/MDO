#!/usr/bin/env python3
"""Run a real, headless ADflow case; use external mpirun for parallel execution."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdo_demo.cfd import run_adflow, tutorial_request, validate_request


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Explicit request JSON; relative mesh paths resolve beside this file")
    source.add_argument("--tutorial-mesh", type=Path, help="Pinned MACH-Aero wing L3 CGNS; separate CPU smoke case")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON, written by rank 0")
    parser.add_argument("--validate-only", action="store_true", help="Validate files and physics without importing ADflow")
    args = parser.parse_args()
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
    comm = None
    try:
        if not args.validate_only:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.rank
        request = None
        error = None
        if rank == 0:
            try:
                raw = tutorial_request(args.tutorial_mesh) if args.tutorial_mesh else json.loads(args.input.read_text(encoding="utf-8-sig"))
                request = validate_request(raw, args.input.resolve().parent if args.input else Path.cwd())
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if comm is not None:
            request, error = comm.bcast((request, error), root=0)
        if error:
            raise ValueError(error)
        if args.validate_only:
            if rank == 0:
                write_json(args.output, {"status": "validated_only", "cfd_executed": False, "request": request})
            return 0
        if rank == 0:
            # Invalidate any earlier success before starting another expensive run.
            # An OOM or external kill must not leave an old result looking current.
            write_json(args.output, {"schema_version": 1, "backend": "adflow", "status": "running",
                                     "case_sha256": request["case_sha256"],
                                     "convergence": {"converged": False}})
        result = run_adflow(request, args.output.parent / (args.output.stem + "_surface"), comm=comm)
        if rank == 0:
            write_json(args.output, result)
            print(f"CFD status={result['status']}; output={args.output}", flush=True)
        return 0 if result["status"] == "ok" else 2
    except Exception as exc:
        if rank == 0:
            write_json(args.output, {"schema_version": 1, "backend": "adflow", "status": "error",
                                     "convergence": {"converged": False},
                                     "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        if comm is not None and comm.size > 1:
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
