"""Apply exact-version text hunks only after verifying every protected file."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path


def package_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("SGLang is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_hunks(original, diff):
    source = original.splitlines(keepends=True)
    output, cursor, i = [], 0, 0
    lines = diff.splitlines(keepends=True)
    while i < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
        if not match:
            i += 1
            continue
        start = int(match[1]) - 1
        if start < cursor:
            raise RuntimeError("Overlapping hunks")
        output.extend(source[cursor:start])
        cursor = start
        i += 1
        while i < len(lines) and not lines[i].startswith(("@@", "diff --git")):
            line = lines[i]
            if line.startswith((" ", "-")):
                if cursor >= len(source) or source[cursor] != line[1:]:
                    raise RuntimeError(
                        "Patch context mismatch; refusing a fuzzy application"
                    )
                cursor += 1
            if line.startswith((" ", "+")):
                output.append(line[1:])
            i += 1
    return "".join(output + source[cursor:])


def install(root, bundle, verify_only=False):
    manifest = json.loads((bundle / "manifest.json").read_text())
    patch = (bundle / "runtime.patch").read_text()
    sections = {}
    for part in patch.split("diff --git ")[1:]:
        header, content = part.split("\n", 1)
        source_path = header.split(" b/", 1)[1]
        sections[source_path.removeprefix("python/sglang/")] = content
    writes = []
    # Complete validation and prepare every replacement before changing files.
    for entry in manifest["files"]:
        relative = entry["path"]
        path = root / relative
        data = path.read_bytes() if path.exists() else None
        actual = digest(data) if data is not None else None
        if actual == entry["patched_sha256"]:
            continue
        if verify_only or actual != entry["base_sha256"]:
            raise RuntimeError(
                f'Unsupported SGLang source: {relative}: {actual}; requires {manifest["base_commit"]}'
            )
        if relative in sections:
            replacement = apply_hunks(data.decode(), sections[relative]).encode()
        else:
            replacement = (bundle / "overlay" / relative).read_bytes()
        if digest(replacement) != entry["patched_sha256"]:
            raise RuntimeError(f"Corrupt patch payload: {relative}")
        compile(replacement, str(path), "exec")
        writes.append((path, replacement))
    for path, replacement in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pp4-install-")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(replacement)
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(
        f'PP4 patch verified: base={manifest["base_commit"]}, files={len(manifest["files"])}, installed={len(writes)}'
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    install(package_root(), Path(__file__).resolve().parent, args.verify_only)
