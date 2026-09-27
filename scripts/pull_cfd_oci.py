#!/usr/bin/env python3
"""Download the official public MACH-Aero image to OCI without a Docker daemon.

This only downloads and verifies blobs. Use umoci --rootless to unpack safely;
never extract image tar members directly into the host filesystem.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error


def fetch_json(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60) as response:
        data = response.read()
        return json.loads(data), data, dict(response.headers)


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default="mdolab/public")
    parser.add_argument("--reference", default="sha256:75963e625a37d9190b568304732f6970eca2737c200f125b2b2c6b2410bd83a3")
    args = parser.parse_args()
    root = args.output.resolve()
    blobs = root / "blobs" / "sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    token, _, _ = fetch_json("https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + args.repository + ":pull")
    auth = {"Authorization": "Bearer " + token["token"]}
    manifest, raw, headers = fetch_json("https://registry-1.docker.io/v2/" + args.repository + "/manifests/" + args.reference,
        {**auth, "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"})
    if args.reference.startswith("sha256:") and digest(raw) != args.reference:
        raise RuntimeError("Image manifest does not match the required digest")
    source_digest = digest(raw)
    (root / "source_manifest.json").write_bytes(raw)
    items = [manifest["config"]] + manifest["layers"]
    print(f"Source {args.repository}@{source_digest}; {sum(x['size'] for x in items)/1e9:.3f} GB compressed", flush=True)

    def download(item):
        name = item["digest"].split(":", 1)[1]
        target = blobs / name
        if target.is_file() and target.stat().st_size == item["size"]:
            cached_hash = hashlib.sha256()
            with target.open("rb") as stream:
                while True:
                    chunk = stream.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    cached_hash.update(chunk)
            if cached_hash.hexdigest() == name:
                return
        local_auth = dict(auth)
        for attempt in range(4):
            checksum = hashlib.sha256()
            try:
                url = "https://registry-1.docker.io/v2/" + args.repository + "/blobs/" + item["digest"]
                request = urllib.request.Request(url, headers=local_auth)
                temporary = target.with_suffix(".download")
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        checksum.update(chunk)
                        stream.write(chunk)
                if temporary.stat().st_size != item["size"] or checksum.hexdigest() != name:
                    raise RuntimeError("Layer hash or size mismatch: " + name)
                temporary.replace(target)
                print(f"Verified {name[:12]} {item['size']/1e6:.1f} MB", flush=True)
                return
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 401:
                    renewed, _, _ = fetch_json("https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + args.repository + ":pull")
                    local_auth = {"Authorization": "Bearer " + renewed["token"]}
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    # Duplicate Docker layers are fetched once.
    unique = list({x["digest"]: x for x in items}.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, unique))
    manifest["mediaType"] = "application/vnd.oci.image.manifest.v1+json"
    manifest["config"]["mediaType"] = "application/vnd.oci.image.config.v1+json"
    for layer in manifest["layers"]:
        layer["mediaType"] = "application/vnd.oci.image.layer.v1.tar+gzip"
    oci = json.dumps(manifest, separators=(",", ":")).encode()
    oci_digest = digest(oci)
    (blobs / oci_digest.split(":")[1]).write_bytes(oci)
    index = {"schemaVersion": 2, "manifests": [{"mediaType": manifest["mediaType"], "size": len(oci),
        "digest": oci_digest, "annotations": {"org.opencontainers.image.ref.name": "cfd"}}]}
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (root / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}', encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps({"source_image": args.repository + "@" + source_digest,
        "oci_manifest_digest": oci_digest, "source_layers_unchanged": True,
        "conversion": "Docker schema2 mediaType descriptors converted to OCI; all original layer/config bytes verified and unchanged"}, indent=2), encoding="utf-8")
    print("OCI download completed: " + str(root), flush=True)


if __name__ == "__main__":
    main()
