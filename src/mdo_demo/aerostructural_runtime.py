"""Actual FM/ADflow providers with one common physical TACS wingbox.

Host Python has PyTorch; the existing official rootless image provides ADflow
and TACS. Subprocess startup and file transfer overhead remain in wall time.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import os
import sys
import time

import numpy as np

from .aero_contract import (from_fm_surface, from_native_export, make_request,
                            native_mainwing_vertices, save_loads)
from .aerostructural import PhysicsFailure, condition_bundle, cfd_numerical_policy
from .io import read_json, write_json
from .matched_cfd import load_bundle, prediction_sample
from .structural import FRAME
from .seed_policy import fixed_seed_policy


class UnifiedRuntime:
    def __init__(self, project, base_request, protocol, output, *, backend, checkpoint=None,
                 device="cuda", reuse_mesh_result=None, mpi_ranks=8, solver_preset=None,
                 initialization_manifest=None):
        self.project = Path(project).resolve()
        self.code = self.project / "code" / "MDO"
        self.base_request = Path(base_request).resolve()
        self.protocol = protocol
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.backend = backend
        self.checkpoint = checkpoint
        self.device = device
        self.reuse_mesh_result = Path(reuse_mesh_result).resolve() if reuse_mesh_result else None
        self.mpi_ranks = int(mpi_ranks)
        if not 1 <= self.mpi_ranks <= 32:
            raise ValueError("Invalid declared CPU rank count")
        self.cfd_policy = cfd_numerical_policy(protocol)
        if self.mpi_ranks != self.cfd_policy["mpi_ranks"]:
            raise ValueError("Runtime MPI ranks differ from the frozen CFD policy")
        if solver_preset is not None and solver_preset != self.cfd_policy["preset"]:
            raise ValueError("Runtime solver preset differs from the frozen CFD policy")
        self.solver_preset = self.cfd_policy["preset"]
        self.model = None
        self.resource_wait_seconds = 0.
        manifest = load_bundle(self.base_request)
        self.native_vertices = native_mainwing_vertices(self.base_request.parent / manifest["surface_path"])
        # BOTH providers use this exact native surface to construct the same FE
        # mesh. Using the FM-resampled surface only on its branch is not fair.
        self.base_identity = manifest["identity"]
        self.started = time.perf_counter()
        runtime_path = self.output / "runtime_probe.json"
        self.image_call(["python", self.code / "scripts/probe_aerostructural_runtime.py",
                         "--request", self.base_request, "--output", runtime_path], self.output / "runtime_probe.log")
        self.probe = read_json(runtime_path)
        self.dynamic_pressure_pa = self.probe["flow"]["dynamic_pressure_pa"]
        self.setup_wall_seconds = time.perf_counter() - self.started
        self.fixed_seed = None
        self.shared_seed_preparation = None
        if fixed_seed_policy(protocol) is not None:
            if initialization_manifest is None or reuse_mesh_result is not None:
                raise ValueError("Fixed-seed protocol requires its manifest and forbids adaptive mesh/result reuse")
            from .fixed_seed_runtime import FixedSeedCFD
            self.fixed_seed = FixedSeedCFD(self, initialization_manifest)
            self.shared_seed_preparation = self.fixed_seed.ledger
            # Both FM and CFD routes require the identical qualified reference
            # capability; preparation costs are charged once at comparison.
            self.fixed_seed.preflight(self.base_request)
        elif initialization_manifest is not None:
            raise ValueError("Initialization manifest requires the new frozen protocol")
        self.setup_wall_seconds = time.perf_counter() - self.started

    def _inside(self, argument):
        if isinstance(argument, Path):
            relative = argument.resolve().relative_to(self.project)
            return str(Path("/home/mdolabuser/mount") / relative).replace("\\", "/")
        return str(argument)

    def image_call(self, args, log_path):
        return self.image_process(args, log_path)["wall_seconds"]

    def image_process(self, args, log_path, *, allowed_exit_codes=(0,)):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(self.code / "scripts/run_cfd_rootless.py"), "--project", str(self.project),
                   "--", "/bin/bash", "-c", 'source "$BASHRC_MDOLAB"; exec "$@"', "mdo-runtime"]
        command += [self._inside(arg) for arg in args]
        start = time.perf_counter()
        if fixed_seed_policy(self.protocol) is not None and args and str(args[0]) == 'mpirun':
            from .shared_resources import wait_for_resources
            self.resource_wait_seconds += wait_for_resources(self.project, log_path.with_suffix('.resources.jsonl'), ranks=8)
        environment = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        with log_path.open("w", encoding="utf-8") as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False, env=environment)
        duration = time.perf_counter() - start
        receipt = {"exit_code": run.returncode, "wall_seconds": duration, "log": str(log_path)}
        write_json(log_path.with_suffix(log_path.suffix + ".process.json"), receipt)
        if run.returncode not in allowed_exit_codes:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1800:]
            raise PhysicsFailure(f"Physical runtime exited {run.returncode}; evidence={log_path}\n{tail}")
        return receipt

    def physical_request(self, manifest):
        identity = manifest["identity"]
        condition = dict(identity["condition"], dynamic_pressure_pa=self.dynamic_pressure_pa)
        span = self.native_vertices[2].mean(axis=1)
        return make_request(manifest["case_sha256"], identity["surface_sha256"], condition=condition,
                            reference=identity["reference"], coverage={"families": ["mainwing"],
                            "closed_TE": False, "span_bounds_m": [float(span[-1]), float(span[0])]})

    def aero(self, alpha, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        request_path = condition_bundle(self.base_request, alpha, output / "bundle")
        manifest = load_bundle(request_path)
        request = self.physical_request(manifest)
        write_json(output / "physical_request.json", request)
        if self.backend == "fm":
            if fixed_seed_policy(self.protocol) is not None:
                from .shared_resources import wait_for_resources
                self.resource_wait_seconds += wait_for_resources(self.project, output / 'resources.jsonl',
                                                                 need_gpu=True, ranks=1)
            from .aerotransformer import AeroTransformerPredictor
            load_seconds = 0.
            if self.model is None:
                if self.checkpoint is None:
                    raise ValueError("FM provider requires an explicit frozen checkpoint")
                self.model = AeroTransformerPredictor(self.checkpoint, self.device)
                load_seconds = self.model.load_seconds
            sample, _ = prediction_sample(request_path)
            prediction = self.model.predict(sample)
            loads = from_fm_surface(request, sample, prediction["fields"],
                        timing={**prediction["timings"], "model_load_seconds": load_seconds,
                                "scope": "prediction plus upstream integral; native conversion, loading and structural analysis recorded separately"},
                        provenance=prediction["provenance"])
            np.savez_compressed(output / "prediction.npz", fields=prediction["fields"],
                                original_geometry=sample["original_geometry"], condition=sample["condition"])
        elif self.backend == "adflow" and self.fixed_seed is not None:
            result_path, export, export_meta, receipt = self.fixed_seed.run(request_path, output)
            result = read_json(result_path)
            loads = from_native_export(request, request_path.parent / manifest["surface_path"],
                                       export, export_meta, result)
            loads["fixed_seed_receipt"] = receipt
        elif self.backend == "adflow":
            run_dir = output / "cfd"
            args = ["mpirun", "--bind-to", "none", "-np", str(self.mpi_ranks), "python",
                    self.code / "scripts/run_matched_cfd.py", "run", "--request", request_path,
                    "--output", run_dir, "--solver-preset", self.solver_preset,
                    "--max-cycles", str(self.cfd_policy["max_cycles"])]
            if self.reuse_mesh_result is not None:
                args += ["--reuse-mesh-result", self.reuse_mesh_result]
            self.image_call(args, output / "cfd.log")
            result_path = run_dir / "result.json"
            result = read_json(result_path)
            if result.get("status") != "ok" or not result.get("convergence", {}).get("converged"):
                raise PhysicsFailure(f"CFD did not meet the frozen convergence criterion: {result_path}")
            surfaces = sorted((run_dir / "solver_surface").glob("*_surf.cgns"))
            if len(surfaces) != 1:
                raise PhysicsFailure("Expected exactly one final CFD surface solution")
            export = output / "native_surface.npz"
            self.image_call(["python", self.code / "scripts/export_native_cfd_surface.py", "--request", request_path,
                             "--result", result_path, "--surface", surfaces[0], "--output", export], output / "export.log")
            loads = from_native_export(request, request_path.parent / manifest["surface_path"],
                                       export, export.with_suffix(".json"), result)
            self.reuse_mesh_result = result_path
        else:
            raise ValueError("Backend must be fm or adflow")
        loads["case_sha256"] = manifest["case_sha256"]
        loads["contract_status"] = {"interface_compatible": loads["interface_compatible"],
                                    "accuracy_validated": loads["accuracy_validated"],
                                    "coverage": loads["coverage"], "limitations": loads["limitations"]}
        loads["timing"]["provider_wall_seconds"] = time.perf_counter() - start
        save_loads(loads, output / "loads.json")
        if not loads["interface_compatible"] or loads["status"] != "ok":
            raise PhysicsFailure("Aerodynamic loads did not pass physical-interface checks")
        return loads

    def structure(self, loads, design, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        input_path = output / "input.npz"
        np.savez_compressed(input_path, surface_vertices_m=self.native_vertices,
                            load_points_m=loads["panel_points_m"], load_forces_N=loads["panel_forces_N"],
                            moment_reference_m=loads["moment_center_m"])
        config = {"schema_version": 1, "coordinate_frame": FRAME, "material": self.protocol["material"],
                  "mesh": self.protocol["mesh"], "thickness": {name: design[name] for name in ("skin_m", "web_m")}}
        config_path = output / "config.json"
        write_json(config_path, config)
        self.image_call(["python", self.code / "scripts/run_structure.py", "--input-npz", input_path,
                         "--config", config_path, "--output-dir", output], output / "structure.log")
        result = read_json(output / "structure_result.json")
        result["provider_wall_seconds"] = time.perf_counter() - start
        result["common_geometry_source"] = self.base_identity["surface_sha256"]
        write_json(output / "structure_result.json", result)
        return result
