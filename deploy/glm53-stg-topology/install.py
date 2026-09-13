"""Verify pinned runtime sources without modifying SGLang."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def package_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("SGLang is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def verify(root, bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    mismatches = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != entry["sha256"]:
            mismatches.append(entry["path"])
    for relative in manifest.get("forbidden_paths", []):
        if (root / relative).exists():
            mismatches.append(relative + " (unexpected experiment overlay)")
    if mismatches:
        raise RuntimeError(
            f"Unsupported SGLang source; requires {manifest['base_commit']}: " + ", ".join(mismatches)
        )
    print(json.dumps({"base_commit": manifest["base_commit"], "verified_files": len(manifest["files"]), "source_modifications": 0}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Compatibility alias: this installer always only verifies")
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    verify(args.package_root or package_root(), Path(__file__).resolve().parent)
