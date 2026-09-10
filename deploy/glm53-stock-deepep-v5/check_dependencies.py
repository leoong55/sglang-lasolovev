"""Inspect package presence at build time; import GPU extensions only at runtime."""

import argparse
from importlib.util import find_spec


def check_build():
    # Top-level find_spec does not execute package __init__. The packaged
    # DeepEP runs CUDA driver/device checks in __init__, even before Buffer use.
    missing = [name for name in ("deep_ep", "sgl_kernel") if find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing installed packages: " + ", ".join(missing))
    print(
        "glm53-stock-deepep-v5: deep_ep and sgl_kernel found without importing; "
        "GPU/API checks deferred to container startup",
        flush=True,
    )


def check_runtime():
    # Do not suppress prerequisite or ABI failures: the GPU container must pass
    # the real imports before model loading begins.
    from deep_ep import Buffer, Config
    from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data

    for method in (
        "capture",
        "low_latency_dispatch",
        "low_latency_combine",
        "get_low_latency_rdma_size_hint",
        "dispatch",
        "combine",
        "get_dispatch_layout",
        "get_dispatch_config",
        "get_combine_config",
    ):
        if not hasattr(Buffer, method):
            raise RuntimeError(
                f"The base image's DeepEP Buffer lacks {method}; normal and low-latency APIs required"
            )
    if (
        not callable(Config)
        or not callable(cutlass_w4a8_moe_mm)
        or not callable(get_cutlass_w4a8_moe_mm_data)
    ):
        raise RuntimeError("Missing DeepEP/CUTLASS W4AFP8 entrypoints")
    print(
        "glm53-stock-deepep-v5: runtime DeepEP normal/low-latency Buffer and CUTLASS W4AFP8 imports verified",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--build", action="store_true")
    stage.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    (check_build if args.build else check_runtime)()
