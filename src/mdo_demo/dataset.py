"""Read the released CRMpert arrays without changing their physical conventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FIELD_SCALES = np.array([1.0, 150.0, 300.0], dtype=np.float64)
FIELD_NAMES = ("Cp", "Cf_stream_x150", "Cf_span_x300")
COEFFICIENT_NAMES = ("CL", "CD", "CM")


def physical_fields(fields: np.ndarray) -> np.ndarray:
    """Undo only the documented friction scaling; Cp is unchanged."""
    fields = np.asarray(fields)
    if fields.ndim != 3 or fields.shape[0] != 3:
        raise ValueError("Expected fields with shape (3, span_cells, chord_cells)")
    return fields / FIELD_SCALES[:, None, None]


class CRMpertDataset:
    """Memory-mapped full release or compact subset written by fetch_assets.py.

    In a compact subset, index[:, 0] remains the *original* shape ID. The
    manifest supplies geometry_shape_ids and sample_ids; IDs are never silently
    reassigned. No split is invented or claimed to reproduce a paper fold.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.manifest: dict[str, Any] = {}
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            self.index = np.load(self.data_dir / "index.npy", mmap_mode="r", allow_pickle=False)
            self.fields = np.load(self.data_dir / "data.npy", mmap_mode="r", allow_pickle=False)
            self.geometry = np.load(self.data_dir / "geom0.npy", mmap_mode="r", allow_pickle=False)
            self.original_geometry = np.load(self.data_dir / "origingeom.npy", mmap_mode="r", allow_pickle=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError("Missing CRMpert assets; run python scripts/fetch_assets.py --samples 8") from exc
        if self.index.ndim != 2 or self.index.shape[1] < 12:
            raise ValueError("CRMpert index must have at least 12 columns")
        if self.fields.ndim != 4 or self.fields.shape[1] != 3 or len(self.fields) != len(self.index):
            raise ValueError("data.npy must be (N_samples, 3, H, W) matching index.npy")
        if self.geometry.ndim != 4 or self.geometry.shape[1:] != self.fields.shape[1:]:
            raise ValueError("geom0.npy must be (N_shapes, 3, H, W) matching the fields")
        expected_vertices = (len(self.geometry), 3, self.fields.shape[2] + 1, self.fields.shape[3] + 1)
        if self.original_geometry.shape != expected_vertices:
            raise ValueError(f"origingeom.npy must be vertex geometry with shape {expected_vertices}")
        shape_ids = self.manifest.get("geometry_shape_ids", list(range(len(self.geometry))))
        self.sample_ids = self.manifest.get("sample_ids", list(range(len(self.index))))
        if len(shape_ids) != len(self.geometry) or len(set(shape_ids)) != len(shape_ids):
            raise ValueError("Invalid or duplicate geometry_shape_ids in manifest")
        if len(self.sample_ids) != len(self.index) or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Invalid or duplicate sample_ids in manifest")
        self._shape_lookup = {int(shape_id): i for i, shape_id in enumerate(shape_ids)}
        original_ids = self.index[:, 0]
        if not np.all(np.isfinite(self.index[:, :12])):
            raise ValueError("Non-finite CRMpert index values")
        if not np.all(original_ids == np.floor(original_ids)):
            raise ValueError("Shape IDs must be integers")
        if any(int(i) not in self._shape_lookup for i in original_ids):
            raise ValueError("index.npy refers to a shape absent from geometry_shape_ids")
        if np.any(self.index[:, 4] <= 0):
            raise ValueError("Reference half-wing area must be positive")

    def __len__(self) -> int:
        return len(self.index)

    def sample(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = self.index[index]
        shape_id = int(row[0])
        geometry_row = self._shape_lookup[shape_id]
        fields = np.array(self.fields[index], dtype=np.float32)
        geometry = np.array(self.geometry[geometry_row], dtype=np.float32)
        if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(geometry)):
            raise ValueError(f"Non-finite fields or geometry in sample {self.sample_ids[index]}")
        return {
            "sample_id": int(self.sample_ids[index]),
            "shape_id": shape_id,
            "condition_id": int(row[1]),
            "geometry": geometry,
            "original_geometry": np.array(self.original_geometry[geometry_row], dtype=np.float64),
            "condition": np.array(row[2:4], dtype=np.float32),
            "condition_names": ("angle_of_attack_deg", "mach"),
            "fields": fields,
            "field_names": FIELD_NAMES,
            "ref_area": float(row[4]),
            "half_span": float(row[5]),
            "solver_coefficients": dict(zip(COEFFICIENT_NAMES, map(float, row[6:9]))),
            "reference_coefficients": dict(zip(COEFFICIENT_NAMES, map(float, row[9:12]))),
            "split": self.manifest.get("split", "unassigned_release_samples"),
        }

    __getitem__ = sample


def load_dataset(data_dir: str | Path) -> CRMpertDataset:
    return CRMpertDataset(data_dir)
