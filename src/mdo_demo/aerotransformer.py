"""Small adapter around the pinned upstream AeroTransformer implementation.

No substitute architecture, fake weights, or synthesized CFD truth is used.
PyTorch and upstream imports are lazy so dataset inspection works on a laptop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from .dataset import FIELD_SCALES, physical_fields

FLOGEN_REVISION = "ff3abda23e10e1073c07ffd78dad96979e940c77"
CFDPOST_REVISION = "c9fb313a4f3f3912a2e1a1aaf56e9e7a338bfa5a"
MODEL_REVISION = "698d70095a00d6f4a25f870e8de0772fc12b68f7"
RELEASED_WEIGHT_HASHES = {
    "3a89a6aa01a37a61695c6ea8de51cf115b2af043c780ed4eec725312f7e80336",
    "46b8c1bd2cc08c02653da30aaed412263cab321e9464c53325599ca7e74c2fde",
    "4fa8fb4e3986cdb7a4a24f1f74410c54412beb5191857c76fdd4c4f0c5dc8bfa",
}


def _add_upstream_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("floGen", "cfdpost"):
        path = root / "external" / name
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _source_provenance(module: Any, expected_revision: str) -> dict[str, Any]:
    path = Path(module.__file__).resolve()
    result: dict[str, Any] = {"module_path": str(path), "expected_revision": expected_revision,
                              "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        source_root = next(parent for parent in path.parents if (parent / ".git").exists())
        result["actual_revision"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        result["tracked_files_modified"] = bool(subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL).strip())
    except (StopIteration, OSError, subprocess.CalledProcessError):
        result["actual_revision"] = None
        result["tracked_files_modified"] = None
    result["matches_pin"] = result["actual_revision"] == expected_revision and result["tracked_files_modified"] is False
    return result


def integrate_coefficients(sample: dict[str, Any], fields: np.ndarray) -> dict[str, float]:
    """Upstream BasicWing integration on the vertex reference surface.

    Fields use the published (Cp, 150*Cf_tau, 300*Cf_span) representation.
    The pinned upstream moment implementation uses ref_point=[0.25, 0, 0].
    The dataset card's leading-edge description is inconsistent with that
    implementation; report CM as diagnostic until the convention is resolved.
    """
    _add_upstream_paths()
    try:
        from cfdpost.wing.basic import BasicWing
    except ImportError as exc:
        raise RuntimeError("cfdpost dependencies missing; run scripts/bootstrap_server.sh") from exc
    wing = BasicWing(
        paras={"ref_area": float(sample["ref_area"])},
        aoa=float(sample["condition"][0]),
        iscentric=True,
        normal_factors=tuple(FIELD_SCALES),
    )
    wing.read_formatted_surface(
        geometry=np.array(sample["original_geometry"], copy=True),
        data=np.array(fields, dtype=np.float64, copy=True),
        isinitg=False,
        isnormed=True,
    )
    wing.aero_force()
    return dict(zip(("CL", "CD", "CM"), map(float, wing.coefficients)))


class AeroTransformerPredictor:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu"):
        start = perf_counter()
        _add_upstream_paths()
        try:
            import torch
            from flowvae.app.wing import models as model_module
            from flowvae.ml_operator.config import ModelConfig
            from cfdpost.wing import basic as post_module
        except ImportError as exc:
            raise RuntimeError("Model dependencies missing; run scripts/bootstrap_server.sh") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU; use --device cpu only for laptop validation")
        checkpoint_dir = Path(checkpoint_dir)
        weights_path = checkpoint_dir / "best_model_weights"
        config_path = checkpoint_dir / "model_config"
        if not weights_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Missing model_config/best_model_weights in {checkpoint_dir}; run scripts/fetch_assets.py")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_name") != "AeroTransformer":
            raise ValueError("This adapter only accepts an AeroTransformer model_config")
        self.model = ModelConfig(config_path=str(config_path)).create().to(self.device)
        # HF publishes a raw tensor state_dict, not a training checkpoint.
        # weights_only avoids executing arbitrary pickle globals.
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.provenance = {
            "model": checkpoint_dir.name,
            "model_repo": "thuerey-group/AeroTransformer",
            "expected_model_revision": MODEL_REVISION,
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "flogen": _source_provenance(model_module, FLOGEN_REVISION),
            "cfdpost": _source_provenance(post_module, CFDPOST_REVISION),
            "device": str(self.device),
            "torch_version": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        self.provenance["evaluation_mode"] = (
            "pretrained_zero_shot_no_local_finetuning"
            if self.provenance["weights_sha256"] in RELEASED_WEIGHT_HASHES
            else "custom_or_unverified_weights_training_history_unknown")
        manifest_path = checkpoint_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.provenance["asset_manifest"] = manifest
            self.provenance["matches_asset_manifest"] = (
                manifest.get("weights_sha256") == self.provenance["weights_sha256"]
                and manifest.get("config_sha256") == self.provenance["config_sha256"])
        self.load_seconds = perf_counter() - start

    def integrate_reference(self, sample: dict[str, Any]) -> dict[str, float]:
        """Reintegrate CFD fields to expose postprocessing/label differences."""
        return integrate_coefficients(sample, sample["fields"])

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        self._synchronize()
        start = perf_counter()
        geometry = np.asarray(sample["geometry"], dtype=np.float32)
        condition = np.asarray(sample["condition"], dtype=np.float32)
        if geometry.shape != (3, 128, 256) or condition.shape != (2,):
            raise ValueError("Released model requires geometry (3,128,256) and condition [AoA_deg, Mach]")
        if not np.all(np.isfinite(geometry)) or not np.all(np.isfinite(condition)):
            raise ValueError("Model inputs must be finite")
        inputs = torch.from_numpy(np.array(geometry, copy=True)).unsqueeze(0).to(self.device)
        code = torch.from_numpy(np.array(condition, copy=True)).unsqueeze(0).to(self.device)
        self._synchronize()
        prepared = perf_counter()
        with torch.inference_mode():
            output = self.model(inputs, code=code)
            if isinstance(output, (tuple, list)):
                output = output[0]
        self._synchronize()
        inferred = perf_counter()
        fields = output[0].detach().cpu().numpy()
        if fields.shape != geometry.shape or not np.all(np.isfinite(fields)):
            raise RuntimeError("Model returned an invalid surface field")
        physical = physical_fields(fields)
        coefficients = integrate_coefficients(sample, fields)
        finished = perf_counter()
        return {
            "fields": fields,
            "cp": physical[0],
            "cf_stream": physical[1],
            "cf_span": physical[2],
            "coefficients": coefficients,
            "timings": {
                "prepare_seconds": prepared - start,
                "model_seconds": inferred - prepared,
                "postprocess_seconds": finished - inferred,
                "total_seconds": finished - start,
            },
            "provenance": self.provenance,
        }
