"""Fixed-budget AeroTransformer adaptation using explicitly designated training data.

Calibration/test arrays are never accessed by the fit loop or used to select a
checkpoint. The exported state_dict retains the upstream architecture exactly.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from .dataset import load_dataset
from .io import sha256_file, write_json


def _ids(values, name, *, nonempty=False):
    if not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{name} must be a list of nonnegative integer source sample IDs")
    if len(set(values)) != len(values) or (nonempty and not values):
        raise ValueError(f"{name} must contain unique IDs" + (" and cannot be empty" if nonempty else ""))
    return values


def _geometry_ids(values, name):
    if not isinstance(values, list) or any(
            not (type(value) is int or (isinstance(value, str) and value.isdecimal()))
            for value in values):
        raise ValueError(f"{name} must list integer or numeric-string geometry IDs")
    return _ids([int(value) for value in values], name)


def resolve_training_split(dataset, split_manifest):
    """Validate IDs and geometry separation from metadata, without reading fields.

    Accept a JSON path, a dictionary, or an explicit training-ID list. Optional
    train/calibration/test keys are geometry-ID lists, matching root's split tool.
    """
    file_hash = None
    if isinstance(split_manifest, (str, Path)):
        path = Path(split_manifest)
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        file_hash = sha256_file(path)
    else:
        raw = split_manifest
    if isinstance(raw, list):
        raw = {"schema_version": 1, "train_sample_ids": raw}
    if not isinstance(raw, dict) or raw.get("schema_version", 1) != 1:
        raise ValueError("Split must be a schema_version=1 object or a training sample-ID list")
    if "train_sample_ids" not in raw:
        raise ValueError("Split requires explicit train_sample_ids; no random split is invented")
    source_ids = [int(x) for x in dataset.sample_ids]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Dataset source sample IDs must be unique")
    lookup = {sample_id: row for row, sample_id in enumerate(source_ids)}
    normalized = {"schema_version": 1}
    used_ids, used_shapes = set(), set()
    for name in ("train", "calibration", "test"):
        values = _ids(raw.get(f"{name}_sample_ids", []), f"{name}_sample_ids", nonempty=name == "train")
        missing = set(values) - lookup.keys()
        if missing:
            raise ValueError(f"{name} IDs absent from dataset: {sorted(missing)}")
        shapes = sorted({int(dataset.index[lookup[sample_id], 0]) for sample_id in values})
        if set(values) & used_ids:
            raise ValueError("Training/calibration/test sample IDs overlap")
        if set(shapes) & used_shapes:
            raise ValueError("Training/calibration/test geometry IDs overlap; split by geometry")
        if name in raw:
            declared = _geometry_ids(raw[name], name)
            if set(declared) != set(shapes):
                raise ValueError(f"Declared {name} geometry IDs do not match its sample IDs")
        normalized[f"{name}_sample_ids"] = list(values)
        normalized[f"{name}_shape_ids"] = shapes
        used_ids.update(values)
        used_shapes.update(shapes)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return {**normalized, "split_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_split_file_sha256": file_hash,
            "source_split_fingerprint": raw.get("fingerprint"),
            "train_rows": [lookup[x] for x in normalized["train_sample_ids"]],
            "selection": "Explicit training IDs only; calibration/test fields not loaded for training"}


def configure_trainable(model, mode):
    """Tune the actual upstream output projection, or all model parameters."""
    if mode not in ("head", "full"):
        raise ValueError("mode must be head or full")
    names = dict(model.named_parameters())
    head_names = [name for name in names if name.startswith("final_layer.out_proj.")]
    if mode == "head" and not head_names:
        raise ValueError("Expected upstream final_layer.out_proj parameters; refusing a guessed head")
    for name, parameter in names.items():
        parameter.requires_grad_(mode == "full" or name in head_names)
    # Freezing weights is insufficient for models with mutable running statistics
    # or dropout: keep the frozen backbone in eval mode throughout head training.
    model.train(mode == "full")
    if mode == "head":
        model.final_layer.out_proj.train()
    trainable = [name for name, parameter in names.items() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    return trainable


def _synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit_model(model, samples, *, torch, device, epochs, batch_size, lr, seed, mode):
    """Internal loop, also testable with a tiny architecture; samples are train only."""
    trainable_names = configure_trainable(model, mode)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if mode == "head" and name not in trainable_names}
    optimizer = torch.optim.Adam(parameters, lr=lr)
    generator = np.random.default_rng(seed)
    history = []
    _synchronize(torch, device)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = generator.permutation(len(samples))
        total_loss, examples, batches = 0.0, 0, 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), batch_size):
            current = [samples[int(index)] for index in order[start:start + batch_size]]
            geometry = torch.from_numpy(np.stack([sample["geometry"] for sample in current])).to(device)
            condition = torch.from_numpy(np.stack([sample["condition"] for sample in current])).to(device)
            target = torch.from_numpy(np.stack([sample["fields"] for sample in current])).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(geometry, code=condition)
            if isinstance(predicted, (tuple, list)):
                predicted = predicted[0]
            if predicted.shape != target.shape or not bool(torch.isfinite(predicted).all()):
                raise RuntimeError("Training produced invalid surface predictions")
            loss = torch.mean((predicted - target) ** 2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if not any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("No training gradients were produced")
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite training gradients")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise RuntimeError("Non-finite adapted weights")
            total_loss += float(loss.detach().cpu()) * len(current)
            examples += len(current)
            batches += 1
        _synchronize(torch, device)
        history.append({"epoch": epoch + 1, "training_mse_online": total_loss / examples,
                        "samples_seen": examples, "batches": batches,
                        "wall_s": time.perf_counter() - epoch_start})
    _synchronize(torch, device)
    fit_seconds = time.perf_counter() - started
    if mode == "head":
        final_state = model.state_dict()
        if any(not torch.equal(value, final_state[name].detach().cpu()) for name, value in frozen.items()):
            raise RuntimeError("Frozen backbone state changed during head-only adaptation")
    return {"history": history, "fit_wall_s": fit_seconds,
            "trainable_parameter_names": trainable_names,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "frozen_state_verified_unchanged": True if mode == "head" else None}


def train_model(data_dir, base_checkpoint, split_manifest, output, device="cpu",
                epochs=10, mode="head", batch_size=1, lr=1e-4, seed=7):
    """Fit a fixed final-epoch checkpoint; return its manifest, never a test score."""
    started = time.perf_counter()
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if mode not in ("head", "full"):
        raise ValueError("mode must be head or full")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Training output must be a new or empty directory")
    dataset = load_dataset(data_dir)
    split = resolve_training_split(dataset, split_manifest)
    # Eagerly load only train cases. This avoids accidentally fetching test labels
    # through a common Dataset __getitem__ during training/model selection.
    samples = []
    data_hash = hashlib.sha256()
    for row in split["train_rows"]:
        sample = dataset.sample(row)
        selected = {key: np.array(sample[key], dtype=np.float32, order="C", copy=True)
                    for key in ("geometry", "condition", "fields")}
        if selected["geometry"].shape != (3, 128, 256) or selected["fields"].shape != (3, 128, 256):
            raise ValueError("Released AeroTransformer requires training geometry/fields (3,128,256)")
        if selected["condition"].shape != (2,) or any(not np.isfinite(value).all() for value in selected.values()):
            raise ValueError("Training geometry, [AoA,Mach], and fields must be finite")
        data_hash.update(json.dumps([sample["sample_id"], sample["shape_id"]]).encode())
        for key in ("geometry", "condition", "fields"):
            data_hash.update(selected[key].tobytes(order="C"))
        samples.append(selected)
    from .aerotransformer import AeroTransformerPredictor
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    predictor = AeroTransformerPredictor(base_checkpoint, device=device)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_status.json", {"status": "running", "mode": mode})
    base_checkpoint = Path(base_checkpoint)
    try:
        stats = _fit_model(predictor.model, samples, torch=torch, device=predictor.device,
                           epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, mode=mode)
        predictor.model.eval()
        state = {name: value.detach().cpu() for name, value in predictor.model.state_dict().items()}
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in state.values()):
            raise RuntimeError("Non-finite state cannot be exported")
        temporary = output / "best_model_weights.tmp"
        torch.save(state, temporary)
        temporary.replace(output / "best_model_weights")
        (output / "model_config").write_bytes((base_checkpoint / "model_config").read_bytes())
        manifest = {"schema_version": 1, "status": "complete", "kind": "aerotransformer_adaptation",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model_name": "AeroTransformer", "architecture_changed": False,
                    "checkpoint_selection": "Fixed final epoch; no calibration/test selection or early stopping",
                    "mode": mode, "base_provenance": predictor.provenance,
                    "base_weights_sha256": predictor.provenance["weights_sha256"],
                    "base_config_sha256": predictor.provenance["config_sha256"],
                    "weights_sha256": sha256_file(output / "best_model_weights"),
                    "config_sha256": sha256_file(output / "model_config"),
                    "split": {key: value for key, value in split.items() if key != "train_rows"},
                    "train_sample_ids": split["train_sample_ids"], "train_shape_ids": split["train_shape_ids"],
                    "split_sha256": split["split_sha256"],
                    "training_data_sha256": data_hash.hexdigest(),
                    "dataset_provenance": dataset.manifest,
                    "loss": "Mean squared error over the three upstream scaled surface channels",
                    "loss_channels": ["Cp", "150*Cf_stream", "300*Cf_span"],
                    "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                                        "seed": seed, "optimizer": "Adam", "weight_decay": 0.0,
                                        "max_grad_norm": 1.0, "precision": "float32", "amp": False},
                    "device": str(predictor.device), "torch_version": torch.__version__,
                    "torch_cpu_threads": torch.get_num_threads(),
                    "reproducibility": "Fixed seed and batch order; bitwise equality across devices is not guaranteed",
                    "timing_scope": "fit includes batch transfers, forward/backward/update and finite checks; total also includes data/model loading and checkpoint export, but excludes CFD label generation",
                    **stats, "total_wall_s": time.perf_counter() - started,
                    "accuracy_claim": "Training loss is not held-out accuracy; evaluate on separate geometry groups",
                    "cost_claim": "No inference or CFD speedup is implied by adapting weights"}
        write_json(output / "manifest.json", manifest)
        write_json(output / "training_status.json", {"status": "complete", "manifest": "manifest.json"})
        return manifest
    except BaseException as exc:
        write_json(output / "training_status.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise
