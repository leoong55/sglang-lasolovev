# SPDX-License-Identifier: Apache-2.0
"""Measured, exact-batch SM90 W4A8 launch choices; stock kernels by default.

Only the standard EP8 GLM MoE path calls this module. Weight representation,
static activation scales, routing and workspaces stay on the original path.
The optional extension is built into the image, never JIT compiled in decode.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)
VARIANTS = {
    0: "stock",
    1: "CO_128x16x512_c1",
    2: "CO_128x32x512_c1",
    3: "CO_128x64x512_c1",
    4: "PP_64x16x512_c1",
    5: "PP_64x32x512_c1",
    6: "CO_128x32x512_c2",
    7: "CO_128x16x512_c2",
}
DEFAULT_LIBRARY = "/opt/glm53-cp8-dcp4-v1/cutlass/glm53_cutlass.so"
_benchmark_pair = None


def validate_config(data):
    if data.get("schema") != 1 or data.get("geometry") != [6144, 2048, 256, 32, 8]:
        raise ValueError("CUTLASS tuning requires schema=1 and GLM53 EP8 geometry")
    if data.get("sm") != 90 or not isinstance(data.get("pairs"), dict):
        raise ValueError("CUTLASS tuning requires SM90 and exact-batch pairs")
    for batch, pair in data["pairs"].items():
        if (not isinstance(batch, str) or not batch.isdecimal()
                or str(int(batch)) != batch or not 1 <= int(batch) <= 64
                or not isinstance(pair, list) or len(pair) != 2
                or any(type(v) is not int or v not in VARIANTS for v in pair)):
            raise ValueError(f"Invalid CUTLASS batch/variant pair: {batch}: {pair}")
    return data


@lru_cache(maxsize=1)
def load_library():
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Load GLM53 CUTLASS extension before CUDA graph capture")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("GLM53 CUTLASS extension requires SM90")
    path = Path(os.environ.get("SGLANG_GLM53_CUTLASS_LIBRARY", DEFAULT_LIBRARY))
    if not path.is_file():
        raise RuntimeError(f"GLM53 CUTLASS library missing; rebuild v9.11 image: {path}")
    torch.ops.load_library(str(path))
    return torch.ops.glm53_cutlass.w4a8_mm


@lru_cache(maxsize=1)
def config():
    path = os.environ.get("SGLANG_GLM53_CUTLASS_CONFIG")
    if not path:
        return None
    import torch

    data = validate_config(json.loads(Path(path).read_text()))
    if data.get("torch") != str(torch.__version__):
        raise ValueError("Retune CUTLASS after changing the PyTorch/image version")
    if data.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("Retune CUTLASS on this GPU model")
    load_library()
    logger.info("GLM53 CUTLASS measured config: %s; batches=%s", path, sorted(data["pairs"]))
    return data


def prepare():
    """Called during weight initialization, before graph capture."""
    config()


def select_pair(batch, hidden, intermediate, local_experts, ep_size, topk):
    if (hidden, intermediate, local_experts, ep_size, topk) != (6144, 2048, 32, 8, 8):
        return (0, 0)
    # No extrapolation into prefill: a profile of C40 does not tune M=16384.
    if not 1 <= batch <= 64:
        return (0, 0)
    if _benchmark_pair is not None:
        return _benchmark_pair
    data = config()
    return tuple(data["pairs"].get(str(batch), (0, 0))) if data else (0, 0)


def gemm(variant, *args):
    if variant == 0:
        from sgl_kernel import cutlass_w4a8_moe_mm

        return cutlass_w4a8_moe_mm(*args)
    # Drop the stock dispatch-only topk argument. Actual group sizes remain
    # GPU metadata; never divide them by EP or infer them from the batch.
    return load_library()(*args[:-1], variant)
