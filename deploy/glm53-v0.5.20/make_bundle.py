"""Export a committed overlay against exactly upstream v0.5.20; no image build."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

BASE = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
BASE_IMAGE = "lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f"
# Unpatched API boundaries must also match the release before installing.
GUARDS = (
    "srt/layers/cp/utils.py",
    "srt/speculative/draft_worker_common.py",
    "srt/model_executor/runner/decode_cuda_graph_runner.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory outside the checkout")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    output = args.output.resolve()
    if output == repo or repo in output.parents or output.exists():
        parser.error("Choose a new output directory outside the checkout")
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists():
        parser.error("Archive already exists")

    def git(*argv):
        return subprocess.check_output(["git", *argv], cwd=repo)

    if git("status", "--porcelain", "--untracked-files=normal").strip():
        parser.error("Commit the source first: REVISION.json must identify the exact tree")
    paths = git("diff", "--name-only", BASE, "HEAD", "--", "python/sglang").decode().splitlines()
    added = set(git("diff", "--name-only", "--diff-filter=A", BASE, "HEAD", "--", "python/sglang").decode().splitlines())
    if not paths or any(not (repo / path).is_file() for path in paths):
        parser.error("Expected a nonempty runtime overlay without deletions")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    digest = lambda data: hashlib.sha256(data).hexdigest()
    records = []
    for path in paths:
        after = git("show", f"HEAD:{path}")
        target = output / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        records.append(dict(path=path, base_sha256=None if path in added else digest(git("show", f"{BASE}:{path}")), patched_sha256=digest(after)))
    guards = {path: digest(git("show", f"{BASE}:python/sglang/{path}")) for path in GUARDS}
    (output / "base-files.json").write_text(json.dumps(dict(base_commit=BASE, guards=guards, files=records), indent=2) + "\n")
    (output / "runtime.patch").write_bytes(git("diff", "--binary", BASE, "HEAD", "--", "python/sglang"))
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(source_commit=git("rev-parse", "HEAD").decode().strip(), source_tree=git("rev-parse", "HEAD^{tree}").decode().strip())
    (output / "REVISION.json").write_text(json.dumps(provenance, indent=2) + "\n")
    series_base = provenance.get("prefill_series_base")
    if series_base:
        (output / "prefill-series.mbox").write_bytes(git("format-patch", "--stdout", "--binary", f"{series_base}..HEAD"))
        (output / "prefill-series-history.txt").write_bytes(git("log", "--reverse", "--format=%H %s", f"{series_base}..HEAD"))
    shutil.copy2(repo / "LICENSE", output / "LICENSE")
    (output / "SHA256SUMS").write_text("".join(f"{digest(p.read_bytes())}  {p.relative_to(output)}\n" for p in sorted(output.rglob("*")) if p.is_file()))
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest(archive.read_bytes())}  {archive.name}\n")
    print(archive)


if __name__ == "__main__":
    main()
