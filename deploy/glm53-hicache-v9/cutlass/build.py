"""Build seven CUTLASS variants sequentially; no GPU is required at image build."""
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

PIN = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
SHA256 = "09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5"


def log(message):
    print(f"[glm53-cutlass] {message}", flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def heartbeat(stage):
    """Keep long compiler/download steps visible even when Ninja is silent."""
    started = time.monotonic()
    stopped = threading.Event()

    def report():
        while not stopped.wait(15):
            log(f"{stage}: still running, elapsed={time.monotonic() - started:.0f}s")

    log(f"{stage}: starting")
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def fetch_archive(root):
    archive = root / "cutlass.tar.gz"
    if archive.exists():
        log("Checking cached CUTLASS archive SHA256")
        if file_sha256(archive) != SHA256:
            raise RuntimeError("CUTLASS archive SHA256 mismatch; remove the cached archive and retry")
        return archive

    partial = root / "cutlass.tar.gz.part"
    digest = hashlib.sha256()
    downloaded = 0
    next_report = 4 * 1024 * 1024
    try:
        with heartbeat("Downloading pinned CUTLASS source"), urllib.request.urlopen(
            f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/{PIN}", timeout=120
        ) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    log(f"Downloaded {downloaded / 1024**2:.1f} MiB")
                    next_report += 4 * 1024 * 1024
        if digest.hexdigest() != SHA256:
            raise RuntimeError("CUTLASS archive SHA256 mismatch")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    log(f"Download verified: {downloaded / 1024**2:.1f} MiB, SHA256={SHA256}")
    return archive


def main():
    root = Path(__file__).resolve().parent
    enabled = os.environ.get("GLM53_BUILD_CUTLASS", "1")
    if enabled not in ("0", "1"):
        raise ValueError("GLM53_BUILD_CUTLASS must be 0 or 1")
    target = root / "glm53_cutlass.so"
    metadata = root / "BUILD.json"
    # Never leave a stale binary/build receipt behind a skipped or failed build.
    target.unlink(missing_ok=True)
    metadata.unlink(missing_ok=True)
    if enabled == "0":
        metadata.write_text(json.dumps(dict(enabled=False, variants=0,
            reason="Humming/stock image; optional CUTLASS tuning extension omitted"), indent=2) + "\n")
        log("CUTLASS tuning extension skipped; Humming and stock CUTLASS remain available")
        return

    log("Preparing seven separate CUDA compilation units; MAX_JOBS=1, target=sm_90a")
    log("An unavailable GPU during docker build is expected; the CUDA toolkit/nvcc is required")
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc").is_file():
        raise RuntimeError("Building the GLM53 extension requires the CUDA toolkit/nvcc")
    log(f"torch={torch.__version__}, torch CUDA={torch.version.cuda}, toolkit={CUDA_HOME}")
    archive = fetch_archive(root)
    with heartbeat("Extracting CUTLASS headers"), tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    cutlass = root / f"cutlass-{PIN}"
    source = root / "csrc"
    if not source.exists():
        # Source-checkout usage; make_bundle copies these into the build kit.
        source = root.parents[2] / "python/sglang/kernels/aot/csrc"
    build = root / "build"
    build.mkdir(exist_ok=True)
    # Force one job, including when the base image exports a larger MAX_JOBS.
    # Merely lowering this on the old single-.cu build did not bound its peak.
    os.environ["MAX_JOBS"] = "1"
    flags = ["-O3", "-std=c++17", "-use_fast_math", "--expt-relaxed-constexpr",
             "--expt-extended-lambda", "-DCUTE_USE_PACKED_TUPLE=1",
             "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
             # Match the native sgl_kernel build, undo cpp_extension defaults.
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
             "-gencode=arch=compute_90a,code=sm_90a"]
    # Explicit architecture prevents probing for GPUs inside docker build.
    sources = [root / "glm53_w4a8.cpp"] + [root / f"variant_{i}.cu" for i in range(1, 8)]
    log("Compilation units: " + ", ".join(path.name for path in sources) + "; then link")
    with heartbeat("Compiling/linking CUTLASS with one worker"):
        path = load(name="glm53_cutlass", sources=[str(path) for path in sources],
                    extra_include_paths=[str(source), str(cutlass / "include"),
                                         str(cutlass / "tools/util/include")],
                    extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags,
                    build_directory=str(build), is_python_module=False, verbose=True)
    log("Checking operator registration (no GPU kernels are launched)")
    # Registration and ABI load are checked without launching a CUDA kernel.
    assert hasattr(torch.ops.glm53_cutlass, "w4a8_mm")
    shutil.copy2(path, target)
    shutil.copy2(cutlass / "LICENSE.txt", root / "CUTLASS-LICENSE.txt")
    metadata.write_text(json.dumps(dict(
        enabled=True, build_layout="one-variant-per-cu", max_jobs=1,
        cutlass_commit=PIN, cutlass_sha256=SHA256, torch=str(torch.__version__),
        cuda=torch.version.cuda, variants=7, gpu_validated=False,
        library_sha256=file_sha256(target)), indent=2) + "\n")
    log("Library ready: glm53_cutlass.so; cleaning temporary source/object files")
    shutil.rmtree(build)
    shutil.rmtree(cutlass)
    archive.unlink()
    log("Build complete; all seven variants are included")


if __name__ == "__main__":
    main()
