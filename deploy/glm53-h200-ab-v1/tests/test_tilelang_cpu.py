"""Check CUDA dispatch and raw layout selection without claiming GPU execution."""

import ast
import unittest
from types import SimpleNamespace as NS

from test_runtime import ROOT


def function(path, name, globals_):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_)
    return globals_[name]


class TileLangTests(unittest.TestCase):
    def test_cuda_gate_accepts_shared_raw_layout_and_rejects_mixed_layout(self):
        check = function(
            "srt/arg_groups/overrides.py", "_check_tilelang_dsa_fp8_kv", {}
        )
        check("fp8_e4m3", "tilelang", "tilelang", hip=False)
        for prefill, decode in (
            ("tilelang", "flashmla_kv"),
            ("flashmla_sparse_q8", "tilelang"),
        ):
            with self.assertRaises(ValueError):
                check("fp8_e4m3", prefill, decode, hip=False)
        check("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_dispatch_uses_device_sm_count_and_fp8_partial_combine(self):
        calls = []

        class Tensor:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.device = "cuda:0"

            def dim(self):
                return 3

            def to(self, dtype):
                calls.append(("cast", dtype))
                return Tensor(self.shape, dtype)

            def unsqueeze(self, dim):
                return self

        def pick(seq, ni, sm, blocks):
            calls.append(("geometry", seq, ni, sm, blocks))
            return 1

        def partial(*args, **kw):
            calls.append(("partial", args, kw))
            return lambda *xs: ("partial-o", "partial-lse")

        torch = NS(
            float8_e4m3fn="fp8",
            float8_e4m3fnuz="fnuz",
            cuda=NS(get_device_properties=lambda d: NS(multi_processor_count=132)),
        )
        dispatch = function(
            "kernels/ops/attention/dsa/tilelang_kernel.py",
            "tilelang_sparse_fwd",
            {
                "torch": torch,
                "_is_hip": False,
                "_is_gfx95_supported": False,
                "_pick_inner_iter": pick,
                "sparse_mla_fwd_decode_partial_fp8": partial,
                "sparse_mla_fwd_decode_combine": lambda *a, **k: lambda *x: "output",
            },
        )
        result = dispatch(
            Tensor((80, 8, 576), "bf16"),
            Tensor((4096, 1, 576), "fp8"),
            Tensor((80, 1, 2048), "int32"),
            0.1,
        )
        self.assertEqual(result, "output")
        self.assertIn(("geometry", 80, 32, 132, 1), calls)
        self.assertIn(("cast", "fp8"), calls)


if __name__ == "__main__":
    unittest.main()
