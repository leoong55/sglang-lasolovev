"""Host checks for the live-state boundary; these are not CUDA validation."""

import ast
import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang"
spec = importlib.util.spec_from_file_location(
    "glm53_bcg_tested", ROOT / "srt/layers/cp/glm53_bcg.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class GraphTests(unittest.TestCase):
    def test_exact_concatenated_geometry_and_tail_fallback(self):
        for lengths in ([8192], [1, 8191], [4095, 4097], [256] * 32):
            self.assertEqual(helper.exact_local_rows(lengths), 1024)
            total = sum(lengths)
            for rank in range(8):
                self.assertEqual(len(range(rank, total, 8)), 1024)
        for lengths in (None, [], [8191], [8193], [0, 8192], [-1, 8193]):
            self.assertIsNone(helper.exact_local_rows(lengths))
        self.assertIsNone(helper.exact_local_rows([8192], 4))

    def test_live_attention_batch_qkv_context_and_allocator_refresh(self):
        current = SimpleNamespace(forward_batch=None)
        attn_state = SimpleNamespace(inputs=None)
        attn_state.set_attn_inputs = lambda value: setattr(attn_state, "inputs", value)
        attn_state.clear_attn_inputs = lambda: setattr(attn_state, "inputs", None)
        seen = []

        class Allocator:
            def __init__(self, *args):
                self.pointer = 0

        def run(positions, hidden, batch, allocator, *args):
            self.assertIs(attn_state.inputs.batch, batch)
            self.assertEqual(allocator.pointer, 0)
            allocator.pointer += 2
            seen.append((batch.prefix, positions.clone()))
            return hidden + batch.prefix

        attn = SimpleNamespace(_forward_impl=run, prepare_qkv_latent=lambda: None)
        modules = {
            "sglang.srt.layers.communicator": SimpleNamespace(
                AttentionInputs=lambda hidden, batch, fn: SimpleNamespace(batch=batch),
                get_attn_tp_context=lambda: attn_state,
            ),
            "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph": SimpleNamespace(
                eager_execution=nullcontext
            ),
            "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph": SimpleNamespace(
                get_tc_piecewise_forward_context=lambda: current
            ),
            "sglang.srt.utils": SimpleNamespace(BumpAllocator=Allocator),
        }
        with patch.dict(sys.modules, modules):
            for prefix in (0, 60000, 120000):
                current.forward_batch = SimpleNamespace(
                    prefix=prefix,
                    positions=torch.tensor([prefix]),
                    attn_cp_metadata=object(),
                    attn_dcp_metadata=object(),
                )
                result = helper._attention_eager(attn, torch.ones(1), None, None, None)
                self.assertEqual(result.item(), prefix + 1)
                self.assertIsNone(attn_state.inputs)
            attn._forward_impl = lambda *args: (_ for _ in ()).throw(
                RuntimeError("test")
            )
            with self.assertRaisesRegex(RuntimeError, "test"):
                helper._attention_eager(attn, torch.ones(1), None, None, None)
            self.assertIsNone(attn_state.inputs)
        self.assertEqual([v[0] for v in seen], [0, 60000, 120000])

    def test_dcp_plan_rebuilt_for_changed_prefix_and_request_slot(self):
        calls = []

        def plan(*args):
            calls.append(args)
            return object()

        runner = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(prepare_context_parallel_metadata_for_dcp=plan),
                kv_cache_dtype="packed_fp8",
                device="cpu",
            ),
            _prefill_forward_context=lambda batch: nullcontext(),
        )
        pools = SimpleNamespace(
            get_req_to_token_pool=lambda: SimpleNamespace(req_to_token="pool"),
            get_token_to_kv_pool=lambda: SimpleNamespace(
                get_kv_buffer_shape=lambda: ["packed_shape"]
            ),
        )
        kernel = object()
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.model_executor.forward_context": pools,
                "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin": SimpleNamespace(
                    create_chunked_prefix_cache_kv_indices=kernel
                ),
            },
        ):
            previous = None
            for prefix, slot in ((0, 1), (60000, 17)):
                batch = SimpleNamespace(
                    seq_lens=[prefix + 8192],
                    extend_prefix_lens=[prefix],
                    extend_prefix_lens_cpu=[prefix],
                    extend_seq_lens=[8192],
                    req_pool_indices=[slot],
                    seq_lens_sum=prefix + 8192,
                    attn_dcp_metadata=previous,
                )
                helper.prepare_dcp(runner, batch)
                self.assertIsNot(batch.attn_dcp_metadata, previous)
                previous = batch.attn_dcp_metadata
        self.assertEqual([a[2] for a in calls], [[0], [60000]])
        self.assertEqual([a[4] for a in calls], [[1], [17]])
        self.assertEqual(calls[1][7], "packed_shape")
        self.assertIs(calls[1][-1], kernel)

    def test_suspend_flag_restores_nested_and_exception_state(self):
        path = (
            ROOT
            / "srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py"
        )
        tree = ast.parse(path.read_text())
        func = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "suspend_breakable_cuda_graph"
        )
        ns = {"contextmanager": contextmanager, "_in_breakable_cuda_graph": True}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), ns)
        with self.assertRaisesRegex(ValueError, "exit"):
            with ns["suspend_breakable_cuda_graph"]():
                self.assertFalse(ns["_in_breakable_cuda_graph"])
                with ns["suspend_breakable_cuda_graph"]():
                    pass
                self.assertFalse(ns["_in_breakable_cuda_graph"])
                raise ValueError("exit")
        self.assertTrue(ns["_in_breakable_cuda_graph"])

    def test_eager_scope_restores_capture_token_and_rejects_live_capture(self):
        from contextvars import ContextVar
        path=ROOT/"srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py"
        tree=ast.parse(path.read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="eager_execution")
        current=ContextVar("test_capture",default=None)
        original=object();current.set(original)
        cuda=SimpleNamespace(is_current_stream_capturing=lambda:False)
        ns={"contextmanager":contextmanager,"_current_capture_var":current,"torch":SimpleNamespace(cuda=cuda)}
        exec(compile(ast.Module(body=[func],type_ignores=[]),str(path),"exec"),ns)
        with patch.dict(sys.modules,{"sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context":SimpleNamespace(suspend_breakable_cuda_graph=nullcontext)}):
            with self.assertRaisesRegex(ValueError,"body"):
                with ns["eager_execution"]():
                    self.assertIsNone(current.get())
                    raise ValueError("body")
            self.assertIs(current.get(),original)
            cuda.is_current_stream_capturing=lambda:True
            with self.assertRaisesRegex(RuntimeError,"active CUDA capture"):
                with ns["eager_execution"]():self.fail("entered active capture")
            self.assertIs(current.get(),original)

    def test_opt_in_support_guard_rejects_other_models_and_topologies(self):
        cfg = SimpleNamespace(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            tp_size=8,
            ep_size=8,
            dp_size=1,
            pp_size=1,
            dcp_size=4,
            kv_cache_dtype="fp8_e4m3",
            page_size=64,
            quantization="w4afp8",
            dcp_comm_backend="ag_rs",
        )
        resolved = SimpleNamespace(attn_cp_size=8, moe_a2a_backend="none")
        model = SimpleNamespace(
            num_hidden_layers=78, architectures=["GlmMoeDsaForCausalLM"]
        )
        overrides = SimpleNamespace(
            resolving_view=lambda s: cfg,
            resolved_view=lambda s: resolved,
            model_config_of=lambda s: SimpleNamespace(hf_config=model),
            attention_backends_of=lambda s: ("flashmla_sparse_q8", "flashmla_kv"),
        )
        with (
            patch.dict(sys.modules, {"sglang.srt.arg_groups.overrides": overrides}),
            patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_BCG": "1"}),
        ):
            self.assertTrue(helper.supports(None))
            for obj, key, value in (
                (cfg, "dcp_size", 1),
                (cfg, "dp_size", 4),
                (cfg, "cp_strategy", "zigzag"),
                (resolved, "moe_a2a_backend", "deepep"),
                (model, "num_hidden_layers", 45),
            ):
                before = getattr(obj, key)
                setattr(obj, key, value)
                self.assertFalse(helper.supports(None))
                setattr(obj, key, before)
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_BCG": "0"}):
            self.assertFalse(helper.supports(None))


if __name__ == "__main__":
    unittest.main()
