"""Exercise both real normal-W4AFP8 branches with CPU reference kernels.

Checks packing, invalid routes, rounding and combine/scaling. The mock GEMM
does not validate CUTLASS or CUDA launch behavior.
"""

import ast
import os
import unittest
from pathlib import Path

import torch


class Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


class W4Tests(unittest.TestCase):
    def test_old_and_optimized_moe_outputs_match_with_padding_and_skew(self):
        root = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang"
        path = root / "srt/layers/moe/cutlass_w4a8_moe.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if getattr(n, "name", "") == "cutlass_w4a8_moe_deepep_normal"
        )
        self.optimized = False
        self.fused_rows = []

        def quant(x, out, scale, is_static):
            out.copy_(
                (x.float() * scale.reciprocal())
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )

        def preprocess(ids, experts):
            flat = ids.flatten()
            order = flat.argsort(stable=True)
            inverse = torch.empty_like(order)
            count_invalid = (flat < 0).sum()
            inverse[order] = torch.arange(order.numel()) - count_invalid
            return flat[order][count_invalid:], inverse, None

        def ordinary_preprocess(ids):
            order = ids.flatten().argsort(stable=True)
            inverse = torch.empty_like(order, dtype=torch.int32)
            inverse[order] = torch.arange(order.numel(), dtype=torch.int32)
            return inverse

        def permute(a, out, mapping, ids, scale, topk, hidden, **kw):
            for i, dest in enumerate(mapping.tolist()):
                if dest >= 0:
                    out[dest].copy_(a[i // topk])

        def fused_permute(a, out, mapping, ids, scale, experts, topk, m, k, **kw):
            self.assertTrue(kw["clamp_fp8"])
            for i, expert in enumerate(ids.flatten().tolist()):
                if expert != experts:
                    quant(a[i // topk], out[mapping[i]], scale, True)

        def sizes(ids, offsets, ps1, ps2, amap, cmap, experts, n, k):
            counts = torch.bincount(ids.flatten().long(), minlength=experts + 1)[
                :experts
            ]
            offsets.copy_(
                torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
            )

        def mm(out, inp, weight, scale, weight_scale, offsets, problems, *args):
            # Use the exact expert packing/offsets; expand fake packed weights.
            rows = self.offsets.tolist()
            for expert in range(weight.shape[0]):
                start, end = rows[expert : expert + 2]
                w = weight[expert].float().repeat_interleave(2, dim=-1)
                out[start:end].copy_((inp[start:end].float() * scale) @ w.T)

        def silu(x, out):
            gate, up = x.float().chunk(2, -1)
            out.copy_((gate / (1 + (-gate).exp()) * up).to(x.dtype))

        def fused_silu(x, out, scale, count, expected, n, **kw):
            self.assertTrue(kw["round_product"])
            valid = int(count.item())
            self.fused_rows.append(valid)
            intermediate = torch.empty((valid, n), dtype=x.dtype)
            silu(x[:valid], intermediate)
            quant(intermediate, out[:valid], scale, True)

        def combine(c2, out, mapping, ids, weights, topk, k, routed_scale, **kw):
            self.assertEqual(routed_scale, 1.0)
            out.zero_()
            for i, dest in enumerate(mapping.tolist()):
                if dest >= 0:
                    row, j = divmod(i, topk)
                    out[row] += (c2[dest] * weights[row, j].to(c2.dtype)).to(c2.dtype)

        ns = dict(
            torch=torch,
            _is_cuda=True,
            glm53_deepep_opt_enabled=lambda: self.optimized,
            cutlass_w4_run_moe_ep_preproess=ordinary_preprocess,
            deepep_run_moe_deep_preprocess=preprocess,
            deepep_permute_triton_kernel=Kernel(permute),
            pre_reorder_for_cutlass_moe=fused_permute,
            get_cutlass_w4a8_moe_mm_data=sizes,
            cutlass_w4a8_moe_mm=mm,
            silu_and_mul=silu,
            per_tensor_quant_fp8=quant,
            silu_mul_static_tensorwise_quant_for_cutlass_moe=fused_silu,
            deepep_post_reorder_triton_kernel=Kernel(combine),
        )
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias("annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
        torch.manual_seed(72)
        m, k, n, e, topk = 19, 16, 8, 4, 3
        a = torch.randn(m, k).bfloat16()
        w1 = torch.randint(-2, 3, (e, n * 2, k // 2), dtype=torch.int8)
        w2 = torch.randint(-2, 3, (e, k, n // 2), dtype=torch.int8)
        weights = torch.rand(m, topk)
        fixtures = [
            torch.randint(-1, e, (m, topk)),
            torch.full((m, topk), -1),
            torch.zeros((m, topk), dtype=torch.int64),
        ]
        for ids in fixtures:
            self.offsets = torch.empty(e + 1, dtype=torch.int64)
            strides = [torch.empty(e, dtype=torch.int64) for _ in range(8)]
            args = [
                a,
                w1,
                w2,
                torch.ones(e),
                torch.ones(e),
                weights,
                ids,
                *strides,
                self.offsets,
                torch.empty(e, 3),
                torch.empty(e, 3),
                torch.tensor([0.25]),
                torch.tensor([0.5]),
            ]
            self.optimized = False
            old = ns["cutlass_w4a8_moe_deepep_normal"](*args)
            self.optimized = True
            new = ns["cutlass_w4a8_moe_deepep_normal"](*args)
            torch.testing.assert_close(new, old, rtol=0, atol=0)
            self.assertEqual(self.fused_rows[-1], int((ids >= 0).sum()))


if __name__ == "__main__":
    unittest.main()
