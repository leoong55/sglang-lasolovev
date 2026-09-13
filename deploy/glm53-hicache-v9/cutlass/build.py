"""Build only the GLM53 CUTLASS extension; no GPU is required at image build."""
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

PIN = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
SHA256 = "09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5"


def main():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    root = Path(__file__).resolve().parent
    if CUDA_HOME is None:
        raise RuntimeError("Building the GLM53 extension requires the CUDA toolkit/nvcc")
    archive = root / "cutlass.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(
            f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/{PIN}", timeout=120
        ) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("CUTLASS archive SHA256 mismatch")
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    cutlass = root / f"cutlass-{PIN}"
    source = root / "csrc"
    if not source.exists():
        # Source-checkout usage; make_bundle copies these into the build kit.
        source = root.parents[2] / "python/sglang/kernels/aot/csrc"
    build = root / "build"
    build.mkdir(exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    flags = ["-O3", "-std=c++17", "-use_fast_math", "--expt-relaxed-constexpr",
             "--expt-extended-lambda", "-DCUTE_USE_PACKED_TUPLE=1",
             "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
             "-gencode=arch=compute_90a,code=sm_90a"]
    # Explicit architecture prevents probing for GPUs inside docker build.
    path = load(name="glm53_cutlass", sources=[str(root / "glm53_w4a8.cu")],
                extra_include_paths=[str(source), str(cutlass / "include"),
                                     str(cutlass / "tools/util/include")],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags,
                build_directory=str(build), is_python_module=False, verbose=True)
    target = root / "glm53_cutlass.so"
    shutil.copy2(path, target)
    shutil.copy2(cutlass / "LICENSE.txt", root / "CUTLASS-LICENSE.txt")
    # Registration and ABI load are checked without launching a CUDA kernel.
    assert hasattr(torch.ops.glm53_cutlass, "w4a8_mm")
    (root / "BUILD.json").write_text(json.dumps(dict(
        cutlass_commit=PIN, cutlass_sha256=SHA256, torch=str(torch.__version__),
        cuda=torch.version.cuda, variants=7, gpu_validated=False,
        library_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2) + "\n")
    shutil.rmtree(build)
    shutil.rmtree(cutlass)
    archive.unlink()


if __name__ == "__main__":
    main()
