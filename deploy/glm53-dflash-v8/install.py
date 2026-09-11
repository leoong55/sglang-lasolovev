"""Install the reviewed overlay only on matching SGLang source files."""

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
import tempfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed sglang package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def atomic_write(path, data):
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def install(root, bundle, verify_only=False):
    manifest = json.loads((bundle / "base-files.json").read_text())
    pending = []
    # Validate every input before changing any file.
    for row in manifest["files"]:
        relative = Path(row["path"]).relative_to("python/sglang")
        target = root / relative
        old = target.read_bytes() if target.exists() else None
        new = (bundle / "overlay" / row["path"]).read_bytes()
        if digest(new) != row["patched_sha256"]:
            raise RuntimeError(f"Damaged overlay: {relative}")
        compile(new, str(target), "exec")
        current = digest(old) if old is not None else None
        if current == row["patched_sha256"]:
            continue
        if verify_only or current != row["base_sha256"]:
            raise RuntimeError(
                f"Source mismatch: {target}; expected base {manifest['base_commit']} or this exact patch"
            )
        pending.append((target, old, new))
    changed = []
    try:
        for target, old, new in pending:
            atomic_write(target, new)
            changed.append((target, old))
            py_compile.compile(str(target), doraise=True)
    except Exception:
        for target, old in reversed(changed):
            if old is None:
                target.unlink(missing_ok=True)
            else:
                atomic_write(target, old)
        raise
    print(
        f"glm53-dflash-v8: verified {len(manifest['files'])} files; installed {len(changed)}; package={root}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    install(
        args.package_root or package_root(),
        Path(__file__).resolve().parent,
        args.verify_only,
    )
