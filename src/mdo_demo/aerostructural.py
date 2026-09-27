"""Shared, bounded one-way aero-structural sizing experiment.

This is an integration pilot, not a reproduction of STW or a two-way
aeroelastic optimization. A real aerodynamic provider and a real structural
provider are required; unavailable/failed physics never become mock results.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np

from .io import json_default, read_json, write_json
from .matched_cfd import canonical_hash, load_bundle

VARIABLES = ("alpha_deg", "skin_m", "web_m")


def cfd_numerical_policy(cfg: dict) -> dict:
    """One recorded CFD policy shared by optimization and all verification."""
    policy = dict(cfg.get("cfd_solver", {"preset": "robust_rans", "max_cycles": 3000,
                                        "l2_convergence": 1e-10, "mpi_ranks": 8}))
    if set(policy) != {"preset", "max_cycles", "l2_convergence", "mpi_ranks"}:
        raise ValueError("CFD policy requires explicit preset, budget, tolerance and ranks")
    if policy["preset"] not in {"robust_rans", "robust_rans_late_nk"}:
        raise ValueError("Unsupported shared CFD policy preset")
    if type(policy["max_cycles"]) is not int or policy["max_cycles"] <= 0:
        raise ValueError("CFD policy max_cycles must be a positive integer")
    if policy["l2_convergence"] != 1e-10:
        raise ValueError("CFD policy must retain the frozen 1e-10 tolerance")
    if type(policy["mpi_ranks"]) is not int or policy["mpi_ranks"] != 8:
        raise ValueError("CFD policy must retain the shared eight MPI ranks")
    return policy


def validate_protocol(raw: dict) -> dict:
    cfg = copy.deepcopy(raw)
    cfd_numerical_policy(cfg)
    if cfg.get("schema_version") != 1 or cfg.get("coupling") != "one_way":
        raise ValueError("Only the explicitly declared one_way integration pilot is implemented")
    if cfg.get("objective") != "wingbox_mass_kg":
        raise ValueError("The shared objective must be wingbox_mass_kg")
    if set(cfg["design"]) != set(VARIABLES):
        raise ValueError("Require the same alpha_deg, skin_m, web_m design variables")
    for name in VARIABLES:
        spec = cfg["design"][name]
        vals = [spec[k] for k in ("initial", "lower", "upper", "scale")]
        if not np.isfinite(vals).all() or not spec["lower"] <= spec["initial"] <= spec["upper"]:
            raise ValueError(f"Invalid design bounds: {name}")
        if spec["scale"] <= 0 or spec["lower"] >= spec["upper"]:
            raise ValueError(f"Invalid variable scale/range: {name}")
        if name.endswith("_m") and spec["lower"] <= 0:
            raise ValueError("Shell thickness bounds must be positive")
    for key in ("lift_coefficient_min", "drag_coefficient_max", "ks_failure_max",
                "tip_displacement_max_m", "objective_scale_kg"):
        if not np.isfinite(cfg["limits"][key]) or cfg["limits"][key] <= 0:
            raise ValueError(f"Positive finite limit required: {key}")
    if cfg["optimizer"]["method"] != "COBYLA":
        raise ValueError("The first paired pilot uses the same COBYLA optimizer")
    for key in ("max_evaluations", "max_aero_evaluations"):
        if type(cfg["optimizer"][key]) is not int or cfg["optimizer"][key] < 1:
            raise ValueError(f"Positive integer budget required: {key}")
    for key in ("rhobeg", "tol", "catol"):
        if not np.isfinite(cfg["optimizer"][key]) or cfg["optimizer"][key] <= 0:
            raise ValueError(f"Positive finite optimizer setting required: {key}")
    cfg["protocol_sha256"] = canonical_hash({k: v for k, v in cfg.items() if k != "protocol_sha256"})
    return cfg


def design_from_scaled(x, cfg):
    values = np.asarray(x, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Three finite design coordinates required")
    return {name: float(values[i] * cfg["design"][name]["scale"]) for i, name in enumerate(VARIABLES)}


def constraint_margins(aero, structural, cfg):
    """Dimensionless margins: nonnegative is feasible, same for both backends."""
    limits = cfg["limits"]
    values = {
        "lift": float(aero["CL"]) / limits["lift_coefficient_min"] - 1.,
        "drag": 1. - float(aero["CD"]) / limits["drag_coefficient_max"],
        "yield": 1. - float(structural["ks_failure"]) / limits["ks_failure_max"],
        "tip_displacement": 1. - float(structural["tip_displacement_m"]) / limits["tip_displacement_max_m"],
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Nonfinite physical constraints cannot be scored")
    mass = float(structural["mass_kg"])
    if not np.isfinite(mass) or mass <= 0 or float(structural["tip_displacement_m"]) < 0:
        raise ValueError("Invalid structural mass/displacement")
    if float(structural["ks_failure"]) < 0:
        raise ValueError("Invalid negative structural failure index")
    return values


def condition_bundle(base_request: Path, alpha_deg: float, output: Path) -> Path:
    """Reuse identical surface geometry, while rehashing the changed flow case."""
    import shutil
    from .io import sha256_file
    base_request = Path(base_request).resolve()
    manifest = load_bundle(base_request)
    if output.exists():
        raise FileExistsError("Use a fresh condition bundle")
    output.mkdir(parents=True)
    manifest = copy.deepcopy(manifest)
    npz_path = output / manifest["fm_input_path"]
    with np.load(base_request.parent / manifest["fm_input_path"], allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    arrays["condition"][0] = float(alpha_deg)
    # The exact representable angle sent to both solvers is the same float32.
    actual_alpha = float(arrays["condition"][0])
    np.savez(npz_path, **arrays)
    surface_path = output / manifest["surface_path"]
    shutil.copyfile(base_request.parent / manifest["surface_path"], surface_path)
    native = read_json(base_request.parent / manifest["native_input_path"])
    native["condition"]["alpha_deg"] = actual_alpha
    native_path = output / manifest["native_input_path"]
    write_json(native_path, native)
    identity = manifest["identity"]
    identity["condition"]["alpha_deg"] = actual_alpha
    identity["native_input_sha256"] = sha256_file(native_path)
    identity["fm_input_sha256"] = sha256_file(npz_path)
    identity["surface_sha256"] = sha256_file(surface_path)
    manifest["case_sha256"] = canonical_hash(identity)
    manifest["source_bundle_case_sha256"] = load_bundle(base_request)["case_sha256"]
    manifest["surface_generation_seconds"] = 0.0
    manifest["surface_reuse_note"] = "Fixed planform in one-way pilot; flow condition changes only"
    request = output / "request.json"
    write_json(request, manifest)
    load_bundle(request)
    return request


class PhysicsFailure(RuntimeError):
    """A recorded physical/dependency failure; never an objective penalty."""


class EvaluationBudgetExceeded(RuntimeError):
    """Predeclared common budget reached, distinct from failed physics."""


class PilotEvaluator:
    """Provider callbacks isolate different runtimes while sharing everything else.

    aero_provider(alpha, directory) -> {coefficients, ...load data...}
    structure_provider(aero, design, directory) -> physical structural metrics.
    Providers must raise on numerical failure. CFD responses are cached only at
    exactly identical angles within this fixed-geometry/condition experiment.
    """
    def __init__(self, protocol, backend, output, aero_provider, structure_provider):
        self.cfg = validate_protocol(protocol)
        self.backend = backend
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.aero_provider = aero_provider
        self.structure_provider = structure_provider
        self.aero_cache = {}
        self.design_cache = {}
        self.history = []
        self.aero_call_count = 0
        self.structural_call_count = 0
        self.started = time.perf_counter()
        write_json(self.output / "protocol.json", self.cfg)

    def evaluate(self, x):
        design = design_from_scaled(x, self.cfg)
        key = tuple(float(v) for v in x)
        if key in self.design_cache:
            return self.design_cache[key]
        if len(self.history) >= self.cfg["optimizer"]["max_evaluations"]:
            raise EvaluationBudgetExceeded("The predeclared design evaluation budget is exhausted")
        number = len(self.history)
        directory = self.output / f"eval_{number:04d}"
        directory.mkdir()
        record = {"evaluation": number, "backend": self.backend, "design": design,
                  "protocol_sha256": self.cfg["protocol_sha256"], "status": "running"}
        began = time.perf_counter()
        budget_error = None
        try:
            violations = []
            for name, value in design.items():
                spec = self.cfg["design"][name]
                violations.append(max(spec["lower"] - value, value - spec["upper"], 0.) / spec["scale"])
            violation = max(violations)
            if violation > 0:
                # COBYLA may evaluate outside bounds while constructing its local
                # approximation. This response describes domain membership only;
                # no aerodynamic or structural result is fabricated or scored.
                record.update(status="domain_rejected", physical_evaluation=False,
                              scaled_bound_violation=violation,
                              optimizer_domain_penalty=1.e6 + violation,
                              optimizer_domain_constraints=[-violation] * 4,
                              feasible=False, common_verification_complete=False)
                return self._save_record(record, began, directory, key)
            # This is also the angle used by the float32 FM input and ADflow request.
            alpha = float(np.float32(design["alpha_deg"]))
            alpha_spec = self.cfg["design"]["alpha_deg"]
            if not alpha_spec["lower"] <= alpha <= alpha_spec["upper"]:
                raise ValueError("Float32 angle conversion crosses the physical domain boundary")
            record["requested_alpha_deg"] = design["alpha_deg"]
            design["alpha_deg"] = alpha
            cache_key = alpha.hex()
            cache_hit = cache_key in self.aero_cache
            if not cache_hit:
                if self.aero_call_count >= self.cfg["optimizer"]["max_aero_evaluations"]:
                    raise EvaluationBudgetExceeded("The predeclared aerodynamic evaluation budget is exhausted")
                self.aero_call_count += 1
                aero = self.aero_provider(alpha, directory / "aero")
                if aero.get("status") != "ok":
                    raise PhysicsFailure(f"Aerodynamic provider did not succeed: {aero.get('status')}")
                self.aero_cache[cache_key] = aero
            aero = self.aero_cache[cache_key]
            self.structural_call_count += 1
            structural = self.structure_provider(aero, design, directory / "structure")
            if structural.get("status") != "ok":
                raise PhysicsFailure(f"Structural provider did not succeed: {structural.get('status')}")
            margins = constraint_margins(aero["coefficients"], structural, self.cfg)
            minimum = min(margins.values())
            success = dict(status="ok", physical_evaluation=True, aerodynamic_cache_hit=cache_hit,
                          aerodynamic_case_sha256=aero.get("case_sha256"),
                          coefficients=aero["coefficients"], structural=structural,
                          constraint_margins=margins, minimum_margin=minimum,
                          feasible=minimum >= -self.cfg["optimizer"]["catol"],
                          mass_kg=structural["mass_kg"],
                          common_verification_complete=False,
                          aerodynamic_contract=aero.get("contract_status"),
                          aerodynamic_evidence={k: copy.deepcopy(aero.get(k)) for k in
                              ("backend", "reference_eligible", "solver_convergence", "integration_audit", "request")})
            success["aerodynamic_evidence"]["provenance"] = {
                "solver_result_content_sha256": aero.get("provenance", {}).get("solver_result_content_sha256")}
            # Reject nonfinite auxiliary provider fields before they can corrupt
            # the persistent record (the scored metrics are checked above).
            json.dumps(success, allow_nan=False, default=json_default)
            record.update(success)
        except EvaluationBudgetExceeded as exc:
            budget_error = exc
            record.update(status="budget_exhausted", error_type=type(exc).__name__, error=str(exc),
                          feasible=False, common_verification_complete=False)
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                          feasible=False, common_verification_complete=False)
        self._save_record(record, began, directory, key)
        if budget_error is not None:
            raise budget_error
        if record["status"] != "ok":
            raise PhysicsFailure(record["error"])
        return record

    def _save_record(self, record, began, directory, key):
        record["evaluation_wall_seconds"] = time.perf_counter() - began
        record["cumulative_wall_seconds"] = time.perf_counter() - self.started
        self.history.append(record)
        write_json(directory / "evaluation.json", record)
        with (self.output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, default=json_default) + "\n")
        if record["status"] in ("ok", "domain_rejected"):
            self.design_cache[key] = record
        return record

    def run(self, *, optimize=True):
        cfg = self.cfg
        x0 = np.array([cfg["design"][v]["initial"] / cfg["design"][v]["scale"] for v in VARIABLES])
        result = {"schema_version": 1, "backend": self.backend, "coupling": "one_way",
                  "scope": "new fixed-planform aero-structural sizing integration pilot; not STW reproduction",
                  "protocol_sha256": cfg["protocol_sha256"], "status": "running",
                  "optimizer_converged": False, "common_verification_complete": False,
                  "mdo_speedup_verified": False}
        try:
            self.evaluate(x0)
            if optimize:
                from scipy.optimize import minimize
                bounds = [(cfg["design"][v]["lower"] / cfg["design"][v]["scale"],
                           cfg["design"][v]["upper"] / cfg["design"][v]["scale"]) for v in VARIABLES]
                def objective(x):
                    record = self.evaluate(x)
                    if record["status"] == "domain_rejected":
                        return record["optimizer_domain_penalty"]
                    return record["mass_kg"] / cfg["limits"]["objective_scale_kg"]
                def constraints(x):
                    record = self.evaluate(x)
                    if record["status"] == "domain_rejected":
                        return np.array(record["optimizer_domain_constraints"])
                    return np.array(list(record["constraint_margins"].values()))
                options = {k: cfg["optimizer"][k] for k in ("rhobeg", "tol", "catol")}
                options["maxiter"] = cfg["optimizer"]["max_evaluations"]
                optimized = minimize(objective, x0, method="COBYLA", bounds=bounds,
                                     constraints=[{"type": "ineq", "fun": constraints}], options=options)
                result.update(optimizer_converged=bool(optimized.success),
                              optimizer_message=str(optimized.message))
            result["status"] = "completed" if not optimize or result["optimizer_converged"] else "budget_limited"
        except EvaluationBudgetExceeded as exc:
            result.update(status="budget_limited", error=str(exc))
        except PhysicsFailure as exc:
            result.update(status="blocked", error=str(exc))
        except Exception as exc:
            result.update(status="blocked", error_type=type(exc).__name__, error=str(exc))
        valid = [r for r in self.history if r["status"] == "ok"]
        feasible = [r for r in valid if r["feasible"]]
        result["best_feasible"] = min(feasible, key=lambda r: r["mass_kg"]) if feasible else None
        result["best_candidate"] = (min(valid, key=lambda r: (-r["minimum_margin"], r["mass_kg"]))
                                    if valid and not feasible else result["best_feasible"])
        result.update(evaluation_count=len(self.history), aerodynamic_call_count=self.aero_call_count,
                      aerodynamic_successful_call_count=len(self.aero_cache),
                      structural_call_count=self.structural_call_count,
                      successful_physical_evaluation_count=len(valid),
                      domain_rejection_count=sum(r["status"] == "domain_rejected" for r in self.history),
                      feasible_candidate_found=bool(feasible),
                      optimization_wall_seconds=time.perf_counter() - self.started,
                      timing_scope="wall time since evaluator creation including provider calls, transfers and structure; any earlier initialization and final common verification are not included")
        write_json(self.output / "result.json", result)
        return result
