#!/usr/bin/env python3
"""Fetch pinned public weights and a small *real* CRMpert subset.

HTTP Range is mandatory for the large .npy arrays. A server that ignores Range
causes an error, not an accidental multi-gigabyte download. Only selected rows
are transferred; their original source sample/shape IDs are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np

MODEL_REVISION = "698d70095a00d6f4a25f870e8de0772fc12b68f7"
DATA_REVISION = "88ece28b846fd1d9870933252db556cf97d30ae0"
WEIGHTS = {
    "ATsurf_S": (4331444, "3a89a6aa01a37a61695c6ea8de51cf115b2af043c780ed4eec725312f7e80336"),
    "ATsurf_M": (15155636, "46b8c1bd2cc08c02653da30aaed412263cab321e9464c53325599ca7e74c2fde"),
    "ATsurf_L": (58218804, "4fa8fb4e3986cdb7a4a24f1f74410c54412beb5191857c76fdd4c4f0c5dc8bfa"),
}
DATA_SHA256 = {
    "index.npy": "4d20000abb8d77671870d1798be34361000fa5219a930025d24fd3229a4c6481",
    "data.npy": "fbda0f974e7630352fbe281a107ee726f0aaca547b884fc6a7334b3cc6ad275e",
    "geom0.npy": "c9803a843516c1fc5ffb4bec61efe12096fc0d0062d9a6ac1b2feb711749d879",
    "origingeom.npy": "89cb144a9e5736c4c9e42a8ef1158243f6b2c6ea84ecef052ad3b4252203e64f",
}
DATA_BASE = f"https://huggingface.co/datasets/thuerey-group/CRMpert/resolve/{DATA_REVISION}"
MODEL_BASE = f"https://huggingface.co/thuerey-group/AeroTransformer/resolve/{MODEL_REVISION}"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def request_bytes(url: str, *, start: int | None = None, size: int | None = None,
                  maximum: int = 64 * 1024 * 1024) -> bytes:
    headers = {"User-Agent": "MDO-reproducible-demo/0.1"}
    if start is not None:
        assert size is not None and size > 0
        headers["Range"] = f"bytes={start}-{start + size - 1}"
        # Distinguish cached ranges at HTTP intermediaries.
        url += f"?download=true&range_start={start}&range_size={size}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if start is not None:
                    expected = f"bytes {start}-{start + size - 1}/"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise RuntimeError("Server did not honor requested HTTP Range; aborting large download")
                    payload = response.read(size + 1)
                    if len(payload) != size:
                        raise RuntimeError("Incomplete or oversized ranged response")
                    return payload
                length = response.headers.get("Content-Length")
                if length and int(length) > maximum:
                    raise RuntimeError(f"Refusing asset larger than {maximum} bytes")
                payload = response.read(maximum + 1)
                if len(payload) > maximum:
                    raise RuntimeError(f"Refusing asset larger than {maximum} bytes")
                return payload
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def download_small(url: str, destination: Path, expected_sha: str | None = None,
                   expected_size: int | None = None) -> None:
    if destination.exists() and expected_sha and sha256(destination) == expected_sha:
        return
    payload = request_bytes(url)
    if expected_size is not None and len(payload) != expected_size:
        raise RuntimeError(f"Wrong size for {destination.name}")
    if expected_sha and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise RuntimeError(f"SHA256 mismatch for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def npy_header(url: str) -> tuple[tuple[int, ...], np.dtype, int]:
    raw = io.BytesIO(request_bytes(url, start=0, size=1024))
    version = np.lib.format.read_magic(raw)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(raw)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(raw)
    else:
        raise ValueError(f"Unsupported NPY version {version}")
    if fortran or dtype.hasobject or dtype.kind not in "fi":
        raise ValueError("Only C-contiguous numeric NPY arrays are supported")
    return shape, dtype, raw.tell()


def fetch_rows(filename: str, rows: list[int], destination: Path) -> dict:
    url = f"{DATA_BASE}/{filename}"
    shape, dtype, offset = npy_header(url)
    if not rows or min(rows) < 0 or max(rows) >= shape[0]:
        raise ValueError(f"Invalid row selection for {filename}")
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    out = np.empty((len(rows), *shape[1:]), dtype=dtype)
    ranges = []
    for local_id, source_id in enumerate(rows):
        start = offset + source_id * row_bytes
        raw = request_bytes(url, start=start, size=row_bytes)
        out[local_id] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
        ranges.append({"source_row": source_id, "start": start, "length": row_bytes,
                       "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"{filename}: row {local_id + 1}/{len(rows)}", flush=True)
    with destination.with_suffix(".npy.part").open("wb") as handle:
        np.save(handle, out, allow_pickle=False)
    destination.with_suffix(".npy.part").replace(destination)
    return {"source_shape": list(shape), "source_dtype": str(dtype), "source_sha256": DATA_SHA256[filename],
            "source_whole_file_hash_verified": False, "ranges": ranges, "local_sha256": sha256(destination)}


def select_sample_ids(index: np.ndarray, count: int) -> list[int]:
    """One deterministic condition on each of several evenly spaced wings."""
    _, first_rows = np.unique(index[:, 0], return_index=True)
    first_rows.sort()
    if count < 1 or count > len(first_rows):
        raise ValueError(f"--samples must be between 1 and {len(first_rows)} distinct shapes")
    positions = np.linspace(0, len(first_rows) - 1, count, dtype=int)
    return list(map(int, first_rows[positions]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-ids", help="Comma-separated original CRMpert row IDs, overrides --samples")
    parser.add_argument("--model", choices=tuple(WEIGHTS), default="ATsurf_S")
    parser.add_argument("--root", type=Path, default=Path("assets"))
    parser.add_argument("--data-dir", type=Path, help="Alternative subset destination")
    args = parser.parse_args()
    model_dir = args.root / "AeroTransformer" / args.model
    model_size, model_sha = WEIGHTS[args.model]
    download_small(f"{MODEL_BASE}/{args.model}/best_model_weights", model_dir / "best_model_weights", model_sha, model_size)
    download_small(f"{MODEL_BASE}/{args.model}/model_config", model_dir / "model_config")
    download_small(f"{MODEL_BASE}/README.md", model_dir / "UPSTREAM_MODEL_CARD.md")
    model_manifest = {"repo": "thuerey-group/AeroTransformer", "revision": MODEL_REVISION,
                      "model": args.model, "weights_sha256": model_sha, "license": "MIT",
                      "config_sha256": sha256(model_dir / "model_config")}
    (model_dir / "manifest.json").write_text(json.dumps(model_manifest, indent=2) + "\n", encoding="utf-8")
    cache = args.root / "metadata" / DATA_REVISION
    download_small(f"{DATA_BASE}/index.npy", cache / "index.npy", DATA_SHA256["index.npy"])
    index = np.load(cache / "index.npy", allow_pickle=False)
    ids = ([int(i) for i in args.sample_ids.split(",")] if args.sample_ids else select_sample_ids(index, args.samples))
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(index):
        raise ValueError("Sample IDs must be unique valid CRMpert rows")
    geometry_ids = list(map(int, np.unique(index[ids, 0])))
    data_dir = args.data_dir or args.root / "CRMpert"
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("sample_ids") != ids or previous.get("revision") != DATA_REVISION:
            raise RuntimeError("Existing subset differs. Choose a new --data-dir to preserve existing results.")
        if all((data_dir / name).is_file() and sha256(data_dir / name) == meta["local_sha256"]
               for name, meta in previous.get("files", {}).items()) and len(previous.get("files", {})) == 4:
            print(f"Verified existing subset: {data_dir} ({len(ids)} samples)")
            return
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "index.npy", index[ids], allow_pickle=False)
    files = {"index.npy": {"source_sha256": DATA_SHA256["index.npy"], "source_whole_file_hash_verified": True,
                            "local_sha256": sha256(data_dir / "index.npy")}}
    files["data.npy"] = fetch_rows("data.npy", ids, data_dir / "data.npy")
    for filename in ("geom0.npy", "origingeom.npy"):
        files[filename] = fetch_rows(filename, geometry_ids, data_dir / filename)
    download_small(f"{DATA_BASE}/README.md", data_dir / "UPSTREAM_DATASET_CARD.md")
    manifest = {"schema_version": 1, "dataset": "thuerey-group/CRMpert", "revision": DATA_REVISION,
                "license": "CC-BY-SA-4.0", "sample_ids": ids, "geometry_shape_ids": geometry_ids,
                "split": "zero_shot_convenience_subset_not_official_paper_split",
                "selection": "explicit_rows" if args.sample_ids else "evenly_spaced_shapes_first_condition",
                "field_scales": [1.0, 150.0, 300.0], "files": files,
                "notice": "Real released CFD labels; local subset is not a new CFD run. Do not claim solver wall time from these arrays."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: model={model_dir}, dataset={data_dir}, {len(ids)} samples / {len(geometry_ids)} wings")


if __name__ == "__main__":
    main()
