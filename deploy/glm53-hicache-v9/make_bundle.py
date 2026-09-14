"""Assemble the exact runtime diff, verified overlay and manual build kit."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASE = "0bcd822377da7b5718e674eaf9c870d349424dd1"
PREVIOUS = "6526bef6bf959e203a0cc23dacdb2553f81c1e3e"
PR_HEAD = "c6aeb8b9d9128b816777e2b64cbf6603344e8fe1"
V98 = "2aa3716d0e9f4d55dca0e5c31dbef2b8c55a3dcf"
V99 = "6bd52126422de3671558f3e412bba0cd57188aa8"
V910 = "1f1171df23c98755c7d479d8befe071a9e13ecbe"
V912 = "3e2b2c5c5176fbf665631b6a1f4961b337707965"


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
        def source_hash(revision):
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{revision}:{path}"], cwd=repo,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
            return hashlib.sha256(git("show", f"{revision}:{path}")).hexdigest() if exists else None
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
                previous_sha256=source_hash(V912),
                compatible_sha256=[source_hash(V910)],
            )
        )
    (output / "runtime.patch").write_bytes(patch)
    (output / "v9.12-to-v9.13.patch").write_bytes(
        git("diff", "--binary", V912, "--", "python/sglang", str(source.relative_to(repo)))
    )
    (output / "v8-to-hicache.patch").write_bytes(
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
                profile="glm53-hicache-v9.13",
                moe_backend_default="cutlass",
                moe_backend_opt_in="humming",
                humming_version="0.1.12",
                previous_working_commit=V912,
                compatible_image_commits=[V910, V912],
                cutlass_extension_included=False,
                default_build_base="glm53-hicache-v9.12-0bcd822377da",
                speculative_profiles={"cp8-dcp4": ["off", "DFLASH"], "tp8": ["off", "EAGLE"]},
                bounded_dflash_block_sizes=[2, 4, 8],
                dflash_block_size_default=8,
                dflash_scheduler_metadata_fix=True,
                dflash_graph_policy_default="warn",
                humming_ep_aware_default=True,
                prefill_padding_max_factor=1.25,
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
