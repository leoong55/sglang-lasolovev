"""Assemble the exact runtime diff, verified overlay and manual build kit."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASE = "0bcd822377da7b5718e674eaf9c870d349424dd1"
PREVIOUS = "258df599b5a53c756125f40cb61b08739dedb908"
PR_HEAD = "c6aeb8b9d9128b816777e2b64cbf6603344e8fe1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New output directory outside the checkout"
    )
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

    if git(
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        "python/sglang",
        str(source.relative_to(repo)),
    ).strip():
        parser.error(
            "Commit runtime and this build kit first so REVISION.json identifies the exact source"
        )
    patch = git("diff", "--binary", BASE, "--", "python/sglang")
    paths = (
        git("diff", "--name-only", BASE, "--", "python/sglang").decode().splitlines()
    )
    if not paths:
        parser.error("No runtime changes relative to the pinned base")
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    records = []
    added = set(
        git("diff", "--diff-filter=A", "--name-only", BASE, "--", "python/sglang")
        .decode()
        .splitlines()
    )
    for path in paths:
        before = None if path in added else git("show", f"{BASE}:{path}")
        after = (repo / path).read_bytes()
        target = output / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        records.append(
            dict(
                path=path,
                base_sha256=hashlib.sha256(before).hexdigest()
                if before is not None
                else None,
                patched_sha256=hashlib.sha256(after).hexdigest(),
            )
        )
    (output / "runtime.patch").write_bytes(patch)
    (output / "v3-to-stock-deepep.patch").write_bytes(
        git(
            "diff",
            "--binary",
            PREVIOUS,
            "--",
            "python/sglang",
            str(source.relative_to(repo)),
        )
    )
    (output / "REVISION.json").write_text(
        json.dumps(
            dict(
                source_commit=git("rev-parse", "HEAD").decode().strip(),
                source_tree=git("rev-parse", "HEAD^{tree}").decode().strip(),
                base_commit=BASE,
                previous_commit=PREVIOUS,
                upstream_pr_head=PR_HEAD,
                gpu_validated=False,
                image_built_here=False,
                profile="glm53-stock-deepep-v5",
            ),
            indent=2,
        )
        + "\n"
    )
    (output / "base-files.json").write_text(
        json.dumps(
            dict(
                base_commit=BASE,
                upstream_pr=36990,
                upstream_pr_head=PR_HEAD,
                files=records,
            ),
            indent=2,
        )
        + "\n"
    )
    shutil.copy2(repo / "LICENSE", output / "LICENSE")
    shutil.copy2(
        repo / "test/registered/dcp/test_dsa_dcp_kv_gather.py",
        output / "tests/test_dsa_dcp_kv_gather_integration.py",
    )
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            hashes.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}"
            )
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    archive = output.with_name(output.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    print(archive)


if __name__ == "__main__":
    main()
