"""Assemble the exact runtime diff, verified overlay and manual build kit."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASE = "0bcd822377da7b5718e674eaf9c870d349424dd1"
PREVIOUS = "eeda8f9e174c3ade29e3532a5e77b7e89b01e7a7"
PR_HEAD = "c6aeb8b9d9128b816777e2b64cbf6603344e8fe1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory outside the checkout")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        parser.error("Output must be outside the source checkout")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    def git(*argv):
        return subprocess.check_output(["git", *argv], cwd=repo)
    patch = git("diff", "--binary", BASE, "--", "python/sglang")
    paths = git("diff", "--name-only", BASE, "--", "python/sglang").decode().splitlines()
    if not paths:
        parser.error("No runtime changes relative to the pinned base")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(source.parent / "glm53-cp8-dcp4-v1/tests", output / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    records = []
    for path in paths:
        before, after = git("show", f"{BASE}:{path}"), (repo / path).read_bytes()
        target = output / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        records.append(dict(path=path, base_sha256=hashlib.sha256(before).hexdigest(),
                            patched_sha256=hashlib.sha256(after).hexdigest()))
    (output / "runtime.patch").write_bytes(patch)
    (output / "v1-to-v2.patch").write_bytes(git("diff", "--binary", PREVIOUS, "--", "python/sglang"))
    (output / "REVISION.json").write_text(json.dumps(dict(
        source_commit=git("rev-parse", "HEAD").decode().strip(),
        source_tree=git("write-tree").decode().strip(),
        base_commit=BASE, previous_commit=PREVIOUS, upstream_pr_head=PR_HEAD,
        gpu_validated=False, image_built_here=False, profile="glm53-cp8-dcp4-q8-v2"), indent=2) + "\n")
    (output / "base-files.json").write_text(json.dumps(dict(base_commit=BASE, upstream_pr=36990,
        upstream_pr_head=PR_HEAD, files=records), indent=2) + "\n")
    shutil.copy2(repo / "LICENSE", output / "LICENSE")
    shutil.copy2(repo / "test/registered/dcp/test_dsa_dcp_kv_gather.py",
                 output / "tests/test_dsa_dcp_kv_gather_integration.py")
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    archive = output.with_name(output.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    print(archive)


if __name__ == "__main__":
    main()
