"""Install the small Humming overlay on the pinned PP2 upstream runtime."""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def package_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("SGLang is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def install(root, bundle=BUNDLE, *, verify_only=False):
    root, bundle = Path(root), Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    failures, pending = [], []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest path must stay inside the SGLang package")
        target = root / relative
        actual = digest(target)
        changed = item["base_sha256"] != item["sha256"]
        if changed and digest(bundle / "runtime" / relative) != item["sha256"]:
            failures.append(str(relative) + " (overlay payload)")
        allowed = {item["sha256"]}
        if not verify_only:
            allowed.add(item["base_sha256"])
        if actual not in allowed:
            failures.append(str(relative) + " (runtime)")
        elif changed and actual != item["sha256"]:
            pending.append(relative)
    for relative in manifest["forbidden_paths"]:
        if (root / relative).exists():
            failures.append(relative + " (unrelated experiment overlay)")
    if failures:
        raise RuntimeError(
            "Requires the pinned upstream or this exact overlay; no files changed: "
            + ", ".join(failures)
        )
    # Validate every source and payload before the first write.
    for relative in pending:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / "runtime" / relative, target)
    print(
        json.dumps(
            {
                "profile": manifest["profile"],
                "base_commit": manifest["base_commit"],
                "verified_files": len(manifest["files"]),
                "installed_files": len(pending),
                "gpu_validated": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    install(
        args.package_root or package_root(),
        args.bundle_root,
        verify_only=args.verify_only,
    )
