"""Exploratory MDO-oriented training on the unchanged, frozen v1 data split.

Only real released CRMpert training fields enter the optimizer. Development
evaluation is a separate command; calibration/test fields are never accessed.
This changed objective is an exploratory v2 pilot, not the registered v1 run.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .dataset import load_dataset
from .io import sha256_file


GROUPS = ("train", "dev", "calibration", "test")
FROZEN_SPLIT_SHA256 = "4b14512e2c53efa731d56b9b8b5b911dc4c7cdcecdbfbd931503bb6a322e460a"
LARGE_WEIGHTS_SHA256 = "4fa8fb4e3986cdb7a4a24f1f74410c54412beb5191857c76fdd4c4f0c5dc8bfa"
DEFAULT_LOSS = {
    "coefficient_scales": (0.01, 0.0005), "spanwise_scale": 0.5,
    "weights": {"field": 1.0, "cl": 0.01, "cd": 0.01, "spanwise": 0.1},
}
LOSS_KEYS = ("total", "field", "cl", "cd", "spanwise")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ids(values, name):
    if not isinstance(values, list) or not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{name} must contain nonempty nonnegative integer IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate IDs")
    return values


def _validate_partition_metadata(raw, dataset, full_index=None):
    """Metadata-only checks, including absent holdout fields in compact data."""
    groups = raw.get("splits", {})
    if set(groups) != set(GROUPS):
        raise ValueError("Exactly train/dev/calibration/test splits are required")
    seen_samples, seen_shapes = set(), set()
    result = {}
    total_samples = raw["dataset"]["total_samples"]
    total_geometries = raw["dataset"]["total_geometries"]
    for name in GROUPS:
        group = groups[name]
        ids = _ids(group.get("source_sample_ids"), f"{name} source_sample_ids")
        shapes = _ids(group.get("geometry_ids"), f"{name} geometry_ids")
        if group.get("n_samples") != len(ids) or group.get("n_geometries") != len(shapes):
            raise ValueError(f"{name} counts disagree with split metadata")
        if max(ids) >= total_samples or max(shapes) >= total_geometries:
            raise ValueError(f"{name} IDs exceed the pinned release bounds")
        if seen_samples.intersection(ids):
            raise ValueError("Four-way sample IDs overlap")
        if seen_shapes.intersection(shapes):
            raise ValueError("Four-way geometry IDs overlap")
        seen_samples.update(ids)
        seen_shapes.update(shapes)
        result[f"{name}_sample_ids"] = list(ids)
        result[f"{name}_shape_ids"] = list(shapes)
    if len(seen_shapes) != total_geometries:
        raise ValueError("Four-way geometry partition does not cover the pinned release")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0
           for x in dataset.sample_ids):
        raise ValueError("Dataset source sample IDs must be nonnegative integers")
    source_ids = [int(x) for x in dataset.sample_ids]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Dataset source sample IDs must be unique")
    lookup = {sample_id: row for row, sample_id in enumerate(source_ids)}
    # Only train/dev rows must be physically present. Inspect index metadata for
    # any available holdout rows, never dataset.sample() or their flow fields.
    for name in GROUPS:
        ids = result[f"{name}_sample_ids"]
        absent = set(ids) - lookup.keys()
        if name in ("train", "dev") and absent:
            raise ValueError(f"Missing exact {name} IDs in dataset: {sorted(absent)[:12]}")
        actual_shapes = set()
        for sample_id in ids:
            if sample_id in lookup:
                value = dataset.index[lookup[sample_id], 0]
                if not np.isfinite(value) or int(value) != value:
                    raise ValueError("Dataset geometry IDs must be finite integers")
                actual_shapes.add(int(value))
                if int(value) not in result[f"{name}_shape_ids"]:
                    raise ValueError(f"{name} sample-to-geometry mapping disagrees with frozen split")
        if not absent and actual_shapes != set(result[f"{name}_shape_ids"]):
            raise ValueError(f"{name} geometry coverage disagrees with frozen split")
        if name in ("train", "dev"):
            result[f"{name}_rows"] = [lookup[x] for x in ids]
        if full_index is not None:
            shapes = set(map(int, full_index[ids, 0]))
            if shapes != set(result[f"{name}_shape_ids"]):
                raise ValueError(f"{name} IDs disagree with full pinned index")
            available = [sample_id for sample_id in ids if sample_id in lookup]
            if available and not np.array_equal(
                    np.asarray(dataset.index[[lookup[x] for x in available], :12]),
                    np.asarray(full_index[available, :12])):
                raise ValueError("Local index values differ from pinned release metadata")
    return result


def resolve_mdo_split(dataset, split_manifest, full_index=None):
    """Audit frozen 450/350 IDs and all four groups without reading flow fields.

    ``full_index`` is an optional path to the complete, hash-pinned index.npy.
    Without it, the exact frozen split metadata plus available local index rows
    are validated; missing calibration/test arrays are deliberately not required.
    """
    file_hash = None
    if isinstance(split_manifest, (str, Path)):
        path = Path(split_manifest)
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        file_hash = sha256_file(path)
    else:
        raw = split_manifest
    if not isinstance(raw, dict) or raw.get("protocol_id") != "CFD-FM-v1":
        raise ValueError("The frozen CFD-FM-v1 manifest is required")
    fingerprint = _json_hash({"dataset": raw.get("dataset"), "splits": raw.get("splits")})
    if fingerprint != FROZEN_SPLIT_SHA256:
        raise ValueError("Dataset/split differs from the frozen v1 train450/dev350 partition")
    metadata = None
    if full_index is not None:
        if sha256_file(Path(full_index)) != raw["dataset"]["index_sha256"]:
            raise ValueError("Full index hash differs from pinned release")
        metadata = np.load(full_index, mmap_mode="r", allow_pickle=False)
        if metadata.ndim != 2 or metadata.shape[0] != raw["dataset"]["total_samples"] or metadata.shape[1] < 12:
            raise ValueError("Invalid full pinned index shape")
    result = _validate_partition_metadata(raw, dataset, metadata)
    allowed_ids = set(result["train_sample_ids"]) | set(result["dev_sample_ids"])
    if set(dataset.sample_ids) != allowed_ids:
        raise ValueError("The pilot dataset must contain exactly frozen train+dev IDs; no other fields are allowed")
    return {**result, "split_sha256": fingerprint, "source_split_file_sha256": file_hash,
            "dataset": raw["dataset"],
            "metadata_validation": "full_pinned_index" if metadata is not None else "frozen_manifest_and_available_index_rows",
            "field_access": "Training accesses only train_sample_ids; separate evaluation may access dev IDs"}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _event(output, stage, **values):
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
    with (Path(output) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
    _write_json(Path(output) / "training_status.json", record)


def _seed(torch, seed):
    # Set before CUDA model/optimizer construction; unsupported deterministic
    # operations must fail explicitly instead of silently changing the protocol.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(torch, device):
    result = {"cpu_peak_rss_bytes": None, "cpu_current_rss_bytes": None,
              "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    try:
        import resource
        result["cpu_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    except ImportError:
        pass
    try:
        import psutil
        memory = psutil.Process().memory_info()
        result["cpu_current_rss_bytes"] = int(memory.rss)
        if hasattr(memory, "peak_wset"):
            result["cpu_peak_rss_bytes"] = int(memory.peak_wset)
    except ImportError:
        pass
    if device.type == "cuda":
        result.update(gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                      gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    return result


def _batch(dataset, row, torch, device):
    sample = dataset.sample(int(row))  # The caller passes a resolved TRAIN row only.
    arrays = {key: np.array(sample[key], dtype=np.float32, order="C", copy=True)
              for key in ("geometry", "condition", "fields", "original_geometry")}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite training sample")
    tensors = {key: torch.from_numpy(value).unsqueeze(0).to(device) for key, value in arrays.items()}
    tensors["ref_area"] = torch.tensor([sample["ref_area"]], dtype=torch.float32, device=device)
    tensors["sample_id"] = int(sample["sample_id"])
    return tensors


def _step(model, batch, optimizer, torch, loss_options):
    from .physics_losses import mdo_loss
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch["geometry"], code=batch["condition"])
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if prediction.shape != batch["fields"].shape or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Invalid surface prediction during MDO training")
    losses = mdo_loss(prediction, batch["fields"], batch["original_geometry"],
                      batch["condition"][:, 0], batch["ref_area"], **loss_options)
    if not bool(torch.stack([torch.isfinite(losses[name]).all() for name in LOSS_KEYS]).all()):
        raise RuntimeError("Non-finite MDO training loss")
    losses["total"].backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError("No gradients produced by MDO objective")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
        raise RuntimeError("Non-finite model parameters after optimizer update")
    return {key: float(losses[key].detach().cpu()) for key in LOSS_KEYS}, float(norm.detach().cpu())


def resource_probe(model, dataset, train_rows, *, torch, device, output, lr=1e-5,
                   warmup_steps=5, measured_steps=20, loss_options=None):
    """Measure actual forward/backward/Adam steps, then restore weights and RNG.

    Probe work is charged and logged, but cannot become the trained checkpoint.
    Peak CPU RSS is the process lifetime high-water mark, not an allocation delta.
    """
    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError("Probe needs nonnegative warmup and positive measured steps")
    loss_options = loss_options or DEFAULT_LOSS
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    original_mode = model.training
    original_flags = [p.requires_grad for p in model.parameters()]
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings, measured_ids = [], []
    started = time.perf_counter()
    try:
        for step in range(warmup_steps + measured_steps):
            _sync(torch, device)
            one_start = time.perf_counter()
            batch = _batch(dataset, train_rows[step % len(train_rows)], torch, device)
            losses, grad_norm = _step(model, batch, optimizer, torch, loss_options)
            _sync(torch, device)
            duration = time.perf_counter() - one_start
            if step >= warmup_steps:
                timings.append(duration)
                measured_ids.append(batch["sample_id"])
            _event(output, "probe_step", step=step + 1, warmup=step < warmup_steps,
                   sample_id=batch["sample_id"], losses=losses, grad_norm=grad_norm, wall_s=duration,
                   **_memory(torch, device))
        result = {"status": "complete", "warmup_steps": warmup_steps, "measured_steps": measured_steps,
                  "batch_size": 1, "precision": "float32", "optimizer": "Adam", "lr": lr,
                  "device": str(device), "torch_version": str(torch.__version__),
                  "torch_cpu_threads": torch.get_num_threads(),
                  "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "measured_source_sample_ids": measured_ids, "step_wall_s": timings,
                  "median_step_wall_s": float(np.median(timings)),
                  "mean_step_wall_s": float(np.mean(timings)),
                  "probe_wall_s": time.perf_counter() - started, **_memory(torch, device),
                  "timing_scope": "batch read/copy, forward, objective, backward, clipping, Adam update and finite checks; excludes event-file writes",
                  "memory_scope": "CPU process-lifetime peak; GPU peak since probe start including optimizer state",
                  "checkpoint_policy": "All probe parameter/buffer updates and torch RNG are restored"}
        return result
    finally:
        model.load_state_dict(state, strict=True)
        model.zero_grad(set_to_none=True)
        model.train(original_mode)
        for parameter, flag in zip(model.parameters(), original_flags):
            parameter.requires_grad_(flag)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        del optimizer, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item, torch) for item in value)
    return value


def fit_mdo_epochs(model, dataset, train_rows, *, torch, device, output, epochs=5,
                   lr=1e-5, seed=7, contract=None, resume=False, loss_options=None):
    """Full-parameter fit with deterministic epoch order and epoch-boundary resume.

    This generic inner loop also supports tiny CPU fixtures. Public train_model
    enforces the official Large weights, CFD provenance, and frozen full split.
    """
    loss_options = loss_options or DEFAULT_LOSS
    output = Path(output)
    contract = contract or {}
    model.float().train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    history, start_epoch = [], 0
    latest = output / "latest_training.pt"
    if resume and latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved.get("contract") != contract:
            raise ValueError("Resume contract differs: weights, data, split, loss, seed and optimizer must match")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        start_epoch, history = saved["completed_epochs"], saved["history"]
        if start_epoch > epochs:
            raise ValueError("Requested epochs cannot precede the committed resume checkpoint")
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        _event(output, "resume", completed_epochs=start_epoch, requested_epochs=epochs,
               checkpoint="latest_training.pt", note="Uncommitted epoch work is replayed; events retain attempt history")
    started = time.perf_counter()
    attempt = datetime.now(timezone.utc).isoformat()
    for epoch in range(start_epoch, epochs):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(train_rows))
        sums = dict.fromkeys(LOSS_KEYS, 0.0)
        norms = []
        epoch_started = time.perf_counter()
        for batch_number, position in enumerate(order):
            _sync(torch, device)
            batch_started = time.perf_counter()
            batch = _batch(dataset, train_rows[int(position)], torch, device)
            losses, grad_norm = _step(model, batch, optimizer, torch, loss_options)
            _sync(torch, device)
            for key in LOSS_KEYS:
                sums[key] += losses[key]
            norms.append(grad_norm)
            _event(output, "train_step", attempt=attempt, epoch=epoch + 1, batch=batch_number + 1,
                   source_sample_id=batch["sample_id"], seed=seed, losses=losses,
                   grad_norm_before_clip=grad_norm, wall_s=time.perf_counter() - batch_started)
        row = {"epoch": epoch + 1, "samples_seen": len(train_rows), "batches": len(train_rows),
               "training_loss_online": {key: sums[key] / len(train_rows) for key in LOSS_KEYS},
               "grad_norm_mean": float(np.mean(norms)), "grad_norm_max": float(np.max(norms)),
               "wall_s_before_checkpoint": time.perf_counter() - epoch_started, **_memory(torch, device)}
        history.append(row)
        saved = {"schema_version": 1, "contract": contract, "completed_epochs": epoch + 1,
                 "model_state": _cpu_tree(model.state_dict(), torch),
                 "optimizer_state": _cpu_tree(optimizer.state_dict(), torch), "history": history,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        temporary = latest.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(latest)
        _write_json(output / "training_history.json", history)
        _event(output, "epoch_committed", **row, checkpoint="latest_training.pt")
    _sync(torch, device)
    return {"history": history, "fit_wall_s_this_invocation": time.perf_counter() - started,
            "completed_epochs": epochs, "resumed_from_epoch": start_epoch,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "resume_granularity": "Completed epochs; latest_training.pt includes optimizer and torch RNG"}


def _training_data_fingerprint(dataset, rows):
    manifest = dataset.manifest
    if manifest.get("dataset", manifest.get("repo")) != "thuerey-group/CRMpert" or manifest.get("revision") != "88ece28b846fd1d9870933252db556cf97d30ae0":
        raise ValueError("Training requires provenance for the real pinned CRMpert release")
    if manifest.get("field_scales") != [1.0, 150.0, 300.0]:
        raise ValueError("Dataset field scaling is missing or incompatible")
    digest = hashlib.sha256()
    for row in rows:
        sample = dataset.sample(row)
        if sample["geometry"].shape != (3, 128, 256) or sample["fields"].shape != (3, 128, 256):
            raise ValueError("Official Large model requires geometry and fields (3,128,256)")
        if sample["original_geometry"].shape != (3, 129, 257) or sample["condition"].shape != (2,):
            raise ValueError("Invalid reference vertex geometry or [AoA,Mach] condition")
        if not math.isfinite(sample["ref_area"]) or sample["ref_area"] <= 0:
            raise ValueError("Invalid reference half-wing area")
        digest.update(json.dumps([sample["sample_id"], sample["shape_id"], sample["ref_area"]]).encode())
        for key in ("geometry", "fields", "original_geometry", "condition"):
            array = np.asarray(sample[key])
            if not np.isfinite(array).all():
                raise ValueError("Non-finite training arrays")
            digest.update(np.asarray(array, dtype=np.float32, order="C").tobytes())
    return digest.hexdigest()


def train_model(data_dir, base_checkpoint, split_manifest, output, *, device="cuda",
                epochs=5, lr=1e-5, seed=7, probe_steps=20, probe_warmup=5,
                probe_only=False, resume=False, full_index=None):
    """Run the MDO-only pilot; never ordinary field-only adaptation or test scoring."""
    for name, value, minimum, maximum in (("epochs", epochs, 1, 30), ("probe_steps", probe_steps, 1, 1000),
                                          ("probe_warmup", probe_warmup, 0, 1000), ("seed", seed, 0, 2**32 - 1)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    output, base_checkpoint = Path(output), Path(base_checkpoint)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not resume)):
        raise ValueError("Use a new/empty output directory, or --resume for an existing pilot")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        _event(output, "validate", experiment="exploratory_v2_pilot", seed=seed)
        dataset = load_dataset(data_dir)
        split = resolve_mdo_split(dataset, split_manifest, full_index)
        from .experiment import verify_dataset
        _event(output, "verify_dataset_assets", train_samples=450, dev_samples=350,
               purpose="Integrity hashes only; optimizer accesses training fields only")
        dataset_manifest_hash = verify_dataset(dataset)
        data_hash = _training_data_fingerprint(dataset, split["train_rows"])
        if sha256_file(base_checkpoint / "best_model_weights") != LARGE_WEIGHTS_SHA256:
            raise ValueError("This pilot must initialize from official ATsurf_L weights, not an adapted/smaller model")
        from .aerotransformer import AeroTransformerPredictor, MODEL_REVISION
        import torch
        _seed(torch, seed)
        _event(output, "load_official_large", seed=seed)
        predictor = AeroTransformerPredictor(base_checkpoint, device=device)
        asset = predictor.provenance.get("asset_manifest", {})
        if asset.get("model") != "ATsurf_L" or asset.get("revision") != MODEL_REVISION or not predictor.provenance.get("matches_asset_manifest"):
            raise ValueError("Official Large model/config asset manifest is required")
        if not all(predictor.provenance[name].get("matches_pin") for name in ("flogen", "cfdpost")):
            raise ValueError("The upstream floGen/cfdpost source must match pinned unmodified revisions")
        predictor.model.float()
        loss_record = json.loads(json.dumps(DEFAULT_LOSS))
        contract = {"experiment": "exploratory_v2_pilot", "base_weights_sha256": LARGE_WEIGHTS_SHA256,
                    "base_config_sha256": predictor.provenance["config_sha256"], "split_sha256": split["split_sha256"],
                    "training_data_sha256": data_hash, "loss": loss_record, "seed": seed, "lr": lr,
                    "dataset_manifest_sha256": dataset_manifest_hash,
                    "batch_size": 1, "precision": "float32", "mode": "full", "optimizer": "Adam",
                    "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
                    "physics_loss_code_sha256": sha256_file(Path(__file__).with_name("physics_losses.py")),
                    "trainer_code_sha256": sha256_file(Path(__file__))}
        contract_path = output / "run_contract.json"
        if resume:
            if not contract_path.exists() or json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise ValueError("Existing output does not match this immutable pilot run contract")
        _write_json(contract_path, contract)
        _write_json(output / "run_metadata.json", {
            "experiment": "exploratory_v2_pilot", "formal_v1_result": False,
            "run_contract": contract, "base_provenance": predictor.provenance,
            "split": {key: value for key, value in split.items() if key not in ("train_rows", "dev_rows")},
            "dataset_manifest_sha256": dataset_manifest_hash,
            "requested_epochs_this_invocation": epochs, "max_supported_epochs": 30,
            "checkpoint_selection": "Fixed final requested epoch; development evaluation is separate",
            "targets": "Same training CFD fields reintegrated for CL/CD and spanwise loads",
            "calibration_and_test_fields_present": False,
            "initial_loss_weights": loss_record,
        })
        probe_path = output / "resource_probe.json"
        if not (resume and probe_path.exists()):
            _event(output, "probe_start", warmup_steps=probe_warmup, measured_steps=probe_steps)
            probe = resource_probe(predictor.model, dataset, split["train_rows"], torch=torch,
                                   device=predictor.device, output=output, lr=lr,
                                   warmup_steps=probe_warmup, measured_steps=probe_steps)
            _write_json(probe_path, probe)
        else:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        if probe_only:
            _event(output, "probe_complete", checkpoint_exported=False, probe="resource_probe.json")
            return {"status": "probe_complete", "experiment": "exploratory_v2_pilot", "probe": probe}
        _seed(torch, seed)
        _event(output, "training_start", epochs=epochs, train_samples=450, dev_samples=350,
               calibration_and_test_fields_accessed=False)
        stats = fit_mdo_epochs(predictor.model, dataset, split["train_rows"], torch=torch,
                              device=predictor.device, output=output, epochs=epochs, lr=lr,
                              seed=seed, contract=contract, resume=resume)
        predictor.model.eval()
        temporary = output / "best_model_weights.tmp"
        torch.save(_cpu_tree(predictor.model.state_dict(), torch), temporary)
        temporary.replace(output / "best_model_weights")
        (output / "model_config").write_bytes((base_checkpoint / "model_config").read_bytes())
        manifest = {"schema_version": 1, "status": "complete", "kind": "aerotransformer_mdo_adaptation",
                    "experiment": "exploratory_v2_pilot", "formal_v1_result": False,
                    "model_name": "AeroTransformer", "model": "ATsurf_L", "mode": "full",
                    "architecture_changed": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_selection": "Fixed last requested epoch; no development/test checkpoint selection",
                    "filename_note": "best_model_weights is the upstream loader filename, not best-dev selection",
                    "base_provenance": predictor.provenance, "run_contract": contract,
                    "weights_sha256": sha256_file(output / "best_model_weights"),
                    "config_sha256": sha256_file(output / "model_config"),
                    "split": {key: value for key, value in split.items() if key not in ("train_rows", "dev_rows")},
                    "train_sample_ids": split["train_sample_ids"], "dev_sample_ids": split["dev_sample_ids"],
                    "training_data_sha256": data_hash, "dataset_manifest_sha256": dataset_manifest_hash,
                    "dataset_provenance": dataset.manifest,
                    "loss": {**loss_record,
                             "field": "Mean squared error of scaled Cp,150Cf_stream,300Cf_span",
                             "cl_cd": "CL and CD errors divided by fixed .01/.0005 then squared",
                             "spanwise": "Width-weighted L2 of nondimensional dCL/deta error divided by .5",
                             "normalization_source": "Fixed initial engineering scales; no dev/calibration/test fitted statistics",
                             "weights_reason": "Coefficient losses have tight engineering scales; 0.01 coefficient weights limit their initial amplification; no claimed optimal weighting",
                             "targets": "Coefficients and loads reintegrated from the same training CFD fields; historical coefficient labels not used"},
                    "hyperparameters": {"epochs": epochs, "batch_size": 1, "lr": lr, "seed": seed,
                                        "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                                        "weight_decay": 0, "max_grad_norm": 1, "precision": "float32",
                                        "amp": False, "tf32": False, "schedule": "constant",
                                        "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"},
                    "resource_probe": probe, "device": str(predictor.device), "torch_version": str(torch.__version__),
                    "torch_cpu_threads": torch.get_num_threads(), **stats,
                    "total_wall_s_this_invocation": time.perf_counter() - started,
                    "heldout_policy": "Optimizer uses training fields only; train/dev files hashed for integrity; dev scoring is separate; calibration/test fields absent and untouched",
                    "claim": "Training/probe completion is not held-out accuracy, MDO feasibility, or CFD speedup"}
        _write_json(output / "manifest.json", manifest)
        _event(output, "complete", completed_epochs=epochs, manifest="manifest.json")
        return manifest
    except BaseException as exc:
        _event(output, "error", error=f"{type(exc).__name__}: {exc}",
               resume_policy="Resume last committed epoch; probe work is never a checkpoint")
        raise
