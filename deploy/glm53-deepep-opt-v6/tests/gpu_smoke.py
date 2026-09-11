"""Optional kernel/graph smoke on Hopper; no model weights or benchmark needed.

Run inside the built image with this directory bind-mounted. This file is not
run during docker build, and passing it does not validate distributed DeepEP.
"""

import torch
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.quantization.per_tensor_quant_fp8 import per_tensor_quant_fp8
from sglang.kernels.ops.moe.ep_moe_kernels import (
    silu_mul_static_tensorwise_quant_for_cutlass_moe,
    pre_reorder_for_cutlass_moe,
    cutlass_w4_run_moe_ep_preproess,
)
from sglang.srt.layers.cp.glm53_deepep import partition_decode_tokens

assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
torch.manual_seed(9)
# Poison unused capacity: the optimized kernel must not read/write the tail.
x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
x[137:] = float("nan")
count = torch.tensor([137], device="cuda", dtype=torch.int32)
scale = torch.tensor([0.03125], device="cuda")
reference_bf16 = torch.empty(137, 2048, device="cuda", dtype=torch.bfloat16)
silu_and_mul(x[:137], reference_bf16)
reference = torch.empty_like(reference_bf16, dtype=torch.float8_e4m3fn)
per_tensor_quant_fp8(reference_bf16, reference, scale, True)
actual = torch.full((1024, 2048), 7, device="cuda", dtype=torch.float8_e4m3fn)
silu_mul_static_tensorwise_quant_for_cutlass_moe(
    x, actual, scale, count, 1024, 2048, round_product=True
)
mismatch = (actual[:137].float() != reference.float()).float().mean().item()
# CUDA fast exp and Triton exp can differ at quantization thresholds.
assert mismatch <= 0.001, f"FP8 mismatch fraction {mismatch}"
torch.testing.assert_close(
    actual[:137].float(), reference.float(), rtol=0.126, atol=0.002
)
assert torch.all(actual[137:].float() == 7)

# Packing is exact, including saturated values and masked routes.
a = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16) * 100
ids = torch.randint(-1, 4, (17, 3), device="cuda")
local_ids = torch.where(ids < 0, 4, ids).int()
map_ = cutlass_w4_run_moe_ep_preproess(local_ids)
out = torch.empty(51, 128, device="cuda", dtype=torch.float8_e4m3fn)
pre_reorder_for_cutlass_moe(
    a, out, map_, local_ids, scale, 4, 3, 17, 128, clamp_fp8=True
)
ref = torch.empty_like(a, dtype=torch.float8_e4m3fn)
per_tensor_quant_fp8(a, ref, scale, True)
for row in range(17):
    for j in range(3):
        if ids[row, j].item() >= 0:
            torch.testing.assert_close(
                out[map_[row * 3 + j]].float(), ref[row].float(), rtol=0, atol=0
            )

# Live GPU count must change on replay even when the capture shape stays 32.
hidden = torch.arange(32 * 16, device="cuda").reshape(32, 16).bfloat16()
live = torch.tensor([32], device="cuda", dtype=torch.int32)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        parts = [partition_decode_tokens(hidden, live, r, 8) for r in range(8)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        parts = [partition_decode_tokens(hidden, live, r, 8) for r in range(8)]
torch.cuda.current_stream().wait_stream(stream)
for raw in (31, 1, 0, 32, 17):
    live.fill_(raw)
    graph.replay()
    assert sum(n.item() for _, n in parts) == raw
print(f"GPU kernel / local graph smoke passed; FP8 mismatch fraction={mismatch:.6f}")
