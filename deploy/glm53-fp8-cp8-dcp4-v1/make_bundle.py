"""Build the cumulative v9.14-derived image kit from a committed checkout."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASE = "219206f7db9f7ffb24dad2605edf72af62d4939e"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    kit = Path(__file__).resolve().parent
    repo = kit.parents[1]
    out = a.output.resolve()
    if out.exists() or out == repo or repo in out.parents:
        p.error("Use a new output directory outside the checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        p.error("Commit the reviewed source first")
    subprocess.run(
        ["python3", str(repo / "deploy/glm53-hicache-v9/make_bundle.py"), str(out)],
        check=True,
    )
    shutil.copytree(
        kit, out / "fp8", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("Dockerfile", "build.sh"):
        shutil.copy2(kit / name, out / name)
    rev = json.loads((out / "REVISION.json").read_text())
    rev.update(
        profile="glm53-fp8-cp8-dcp4-v1",
        fork_base_commit=BASE,
        image_built_here=False,
        gpu_validated=False,
        target_quantization="fp8 (checkpoint metadata)",
        moe_backend_default="auto (native FP8 with A2A none: Triton)",
        moe_backend_opt_in=None,
        humming_required=False,
        shared_experts_fusion="native default",
        speculative_profiles={"cp8-dcp4": ["off", "DFLASH"]},
        default_build_base="lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9",
    )
    (out / "REVISION.json").write_text(json.dumps(rev, indent=2) + "\n")
    # Include the immediate v9.14 parent as an exact accepted overlay source.
    manifest = json.loads((out / "base-files.json").read_text())
    for row in manifest["files"]:
        r = subprocess.run(
            ["git", "show", BASE + ":" + row["path"]], cwd=repo, capture_output=True
        )
        row["compatible_sha256"].append(
            hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
        )
    (out / "base-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sums = [
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}"
        for f in sorted(out.rglob("*"))
        if f.is_file() and f.name != "SHA256SUMS"
    ]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with tarfile.open(out.with_name(out.name + ".tar.gz"), "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print(out)


if __name__ == "__main__":
    main()
