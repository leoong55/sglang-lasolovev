"""Check CUDA dispatch and raw layout selection without claiming GPU execution."""

import ast
import unittest
from types import SimpleNamespace as NS

from test_runtime import ROOT, load_method


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


class PrefixReaderTests(unittest.TestCase):
    def read_prefix(self, width, dtype, *, hip=False, wrapped=False):
        calls = []
        indices = [65, 2, 65, 127]
        buffer = NS(shape=(4096, 1, width), dtype=dtype)

        class Tensor:
            def squeeze(self, dim):
                calls.append(("squeeze", dim))
                return self

            def contiguous(self):
                return self

            def __getitem__(self, index):
                calls.append(("slice", index))
                return self

        result = Tensor()

        def get_key(layer_id):
            calls.append(("layer", layer_id))
            return buffer

        def raw_read(layer, ids, dst):
            calls.append(("raw", layer.layer_id, ids, dst))
            return result, result

        def scaled_read(kv, ids):
            calls.append(("scaled", kv.shape[-1], ids))
            if kv.shape[-1] != 656:
                raise AssertionError("raw KV reached the scaled dequantizer")
            return result

        class Wrapper:
            def __init__(self, primary):
                self.primary = primary

        backend = NS(forward_metadata=NS(page_table_1_flattened=indices))
        if wrapped:
            backend = Wrapper(backend)
        read = load_method(
            "srt/models/deepseek_common/attention_forward_methods/forward_mha.py",
            "DeepseekMHAForwardMixin",
            "_get_mla_kv_buffer_from_fp8_for_dsa",
            {
                "get_attn_backend": lambda: backend,
                "TboAttnBackend": Wrapper,
                "get_token_to_kv_pool": lambda: NS(
                    get_key_buffer=get_key, get_mla_kv_buffer=raw_read
                ),
                "filter_dcp_local_kv_indices": lambda *, kv_indices: kv_indices,
                "torch": NS(float8_e4m3fn="fp8", bfloat16="bf16"),
                "_use_aiter_gfx95": hip,
                "dequantize_k_cache_paged": scaled_read,
            },
        )
        layer = NS(attn_mha=NS(layer_id=39), kv_lora_rank=512, qk_rope_head_dim=64)
        read(layer, NS())
        return calls

    def test_raw_cuda_prefix_uses_pool_reader_with_global_pp_layer_id(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                calls = self.read_prefix(576, "fp8", wrapped=wrapped)
                self.assertIn(("layer", 39), calls)
                self.assertIn(("raw", 39, [65, 2, 65, 127], "bf16"), calls)
                self.assertFalse(any(c[0] == "scaled" for c in calls))

    def test_scaled_prefix_keeps_block_scale_dequantization(self):
        calls = self.read_prefix(656, "fp8")
        self.assertIn(("scaled", 656, [65, 2, 65, 127]), calls)
        self.assertFalse(any(c[0] == "raw" for c in calls))

    def test_rocm_fnuz_prefix_keeps_native_reader(self):
        calls = self.read_prefix(576, "fnuz", hip=True)
        self.assertIn(("raw", 39, [65, 2, 65, 127], "bf16"), calls)

    def test_unrecognized_576_byte_dtype_does_not_bypass_layout_assert(self):
        with self.assertRaises(AssertionError):
            self.read_prefix(576, "uint8")


class PPActiveCountTests(unittest.TestCase):
    def test_two_microbatches_are_deduplicated_and_finished_requests_excluded(self):
        collect = load_method(
            "srt/managers/scheduler.py", "Scheduler", "collect_inflight_reqs", {}
        )
        count = function(
            "srt/managers/scheduler_components/metrics_reporter.py",
            "_pp_active_request_count",
            {},
        )

        class Req:
            done = False

            def finished(self):
                return self.done

        requests = [Req() for _ in range(128)]
        a, b = NS(reqs=requests[:64]), NS(reqs=requests[64:])
        scheduler = NS(ps=NS(pp_size=2), running_mbs=[a, b], mbs=[a, b])
        scheduler.collect_inflight_reqs = lambda: collect(scheduler)
        self.assertEqual(count(scheduler), 128)
        requests[0].done = True
        self.assertEqual(count(scheduler), 127)
        scheduler.running_mbs[1] = None
        scheduler.mbs[1] = None
        self.assertEqual(count(scheduler), 63)


if __name__ == "__main__":
    unittest.main()
