"""Separate GPU-free image checks from runtime CUDA/DeepEP checks."""

import importlib.util


def check_build():
    for name in ("torch", "triton", "sgl_kernel", "deep_ep"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Required package is missing: {name}")
    print(
        "v6 build dependencies present (GPU imports deferred to pod startup)",
        flush=True,
    )


def check_runtime():
    import torch
    from deep_ep import Buffer
    import sgl_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("v6 requires CUDA devices at runtime")
    for name in ("dispatch", "combine", "low_latency_dispatch", "low_latency_combine"):
        if not hasattr(Buffer, name):
            raise RuntimeError(f"DeepEP Buffer is missing {name}")
    for name in ("cutlass_w4a8_moe_mm", "get_cutlass_w4a8_moe_mm_data"):
        if not hasattr(sgl_kernel, name):
            raise RuntimeError(f"sgl_kernel is missing {name}")
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("This experiment targets Hopper (SM90)")
    print(
        "v6 runtime: Hopper / DeepEP normal+LL / W4AFP8 kernels available", flush=True
    )


if __name__ == "__main__":
    check_build()
