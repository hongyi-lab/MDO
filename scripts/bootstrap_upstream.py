#!/usr/bin/env python3
"""Check out fixed upstream revisions, preserving any existing user changes."""

import json
from pathlib import Path
import subprocess


def git(*args, cwd=None):
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/upstream.lock.json").read_text(encoding="utf-8"))
    destination = root/"external"
    destination.mkdir(exist_ok=True)
    for entry in config["repositories"]:
        path = destination/entry["directory"]
        if path.exists():
            if not (path/".git").exists():
                raise RuntimeError(f"Existing non-Git directory: {path}; preserve it and choose a clean checkout")
            remote = git("remote", "get-url", "origin", cwd=path).removesuffix(".git").lower()
            if remote != entry["url"].removesuffix(".git").lower():
                raise RuntimeError(f"Unexpected origin in {path}")
            if git("status", "--porcelain", cwd=path):
                raise RuntimeError(f"Upstream checkout has local changes: {path}; no changes were discarded")
            if git("rev-parse", "HEAD", cwd=path) != entry["commit"]:
                raise RuntimeError(f"Upstream revision differs in {path}; expected {entry['commit']}. Preserve your work and use a fresh directory.")
        else:
            git("clone", "--no-checkout", "--filter=blob:none", entry["url"], str(path))
            git("checkout", "--detach", entry["commit"], cwd=path)
        print(f"{entry['directory']}: {entry['commit']} ({entry['license']})", flush=True)


if __name__ == "__main__":
    main()
