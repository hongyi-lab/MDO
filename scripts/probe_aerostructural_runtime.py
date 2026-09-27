#!/usr/bin/env python3
"""Import actual structural dependencies and obtain the solver's flow state.

No CFD, training, or structural solve is executed by this dependency probe.
"""
import argparse
import importlib.metadata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdo_demo.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from baseclasses import AeroProblem
    from tacs import TACS, constitutive, elements, functions
    from mpi4py import MPI
    import scipy
    request = read_json(args.request)
    identity = request["identity"]
    c, r = identity["condition"], identity["reference"]
    ap = AeroProblem(name="dependency_probe", mach=c["mach"], alpha=c["alpha_deg"],
                     reynolds=c["reynolds"], reynoldsLength=c["reynolds_length_m"],
                     T=c["temperature_k"], areaRef=r["area_m2"], chordRef=r["chord_m"])
    flow = {name: float(getattr(ap, name)) for name in ("rho", "V", "P", "T", "mu", "a")}
    flow["dynamic_pressure_pa"] = 0.5 * flow["rho"] * flow["V"]**2
    if args.output.exists():
        raise FileExistsError("Preserve existing runtime evidence; use a new output")
    write_json(args.output, {"status": "ok", "source": "actual baseclasses.AeroProblem state", "flow": flow,
                            "condition": c, "baseclasses_version": importlib.metadata.version("mdolab-baseclasses"),
                            "tacs_module": TACS.__file__, "scipy_version": scipy.__version__,
                            "mpi_size": MPI.COMM_WORLD.Get_size(), "solver_executed": False})


if __name__ == "__main__":
    main()
