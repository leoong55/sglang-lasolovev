"""CPU control-flow checks; no claim about CUDA kernel correctness/performance."""

import contextlib
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from test_host_controls import drop_options, replace_value
from test_host_launch import args_for, launch

ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
spec = importlib.util.spec_from_file_location(
    "cp_fusion_test_subject", ROOT / "python/sglang/srt/layers/cp/glm53_decode_fusion.py"
)
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


class FusionTest(unittest.TestCase):
    def test_deferred_reduce_restored_even_on_error(self):
        projection = NS(use_decode_attn_tp=True, tp_size=1, reduce_results=False)
        with self.assertRaisesRegex(RuntimeError, "attention failed"):
            with fusion.defer_output_reduce(projection, True):
                self.assertFalse(projection.use_decode_attn_tp)
                raise RuntimeError("attention failed")
        self.assertTrue(projection.use_decode_attn_tp)
        with fusion.defer_output_reduce(projection, False):
            self.assertTrue(projection.use_decode_attn_tp)

    def test_fallback_performs_exactly_one_reduce_before_norm(self):
        calls = []
        x, residual = torch.tensor([[1., 3.]]), torch.tensor([[2., 4.]])
        def reduce(t):
            calls.append("reduce")
            return t * 8
        class Norm:
            weight = torch.ones(2)
            variance_epsilon = 1e-6
            def __call__(self, t, r):
                calls.append("norm")
                return t + r, r
        comm = NS(flashinfer_allreduce_residual_rmsnorm=Mock(return_value=(None, None)))
        modules = {
            "sglang.srt.distributed": NS(tensor_model_parallel_all_reduce=reduce),
            "sglang.srt.layers.flashinfer_comm_fusion": comm,
        }
        with patch.dict(sys.modules, modules):
            out, _ = fusion.finish(x, residual, Norm())
            torch.testing.assert_close(out, x * 8 + residual)
            self.assertEqual(calls, ["reduce", "norm"])
            calls.clear()
            comm.flashinfer_allreduce_residual_rmsnorm.return_value = (x, residual)
            self.assertIs(fusion.finish(x, residual, Norm())[0], x)
            self.assertEqual(calls, [])
            comm.flashinfer_allreduce_residual_rmsnorm.side_effect = RuntimeError("CUDA failed")
            with self.assertRaisesRegex(RuntimeError, "CUDA failed"):
                fusion.finish(x, residual, Norm())
            self.assertEqual(calls, [])

    def test_eligibility_rejects_prefill_wrong_peers_and_missing_workspace(self):
        group = NS(ranks=list(range(8)), world_size=8, device_group=object(), cpu_group=object())
        tp = NS(ranks=list(range(8)), world_size=8)
        parallel = NS(tp_size=8, moe_ep_size=8, attn_cp_size=8, attn_tp_size=1,
                      attn_dp_size=1, pp_size=1, nnodes=1, attn_dcp_size=4)
        comm = NS(can_use_flashinfer_allreduce=Mock(return_value=True))
        modules = {
            "sglang.srt.distributed": NS(get_moe_ep_group=lambda: group, get_tp_group=lambda: tp),
            "sglang.srt.layers.flashinfer_comm_fusion": comm,
            "sglang.srt.runtime_context": NS(
                get_parallel=lambda: parallel, get_platform=lambda: NS(is_sm90=True),
                get_exec=lambda: NS(comm=NS(flashinfer_allreduce_fusion_backend="trtllm"))),
        }
        layer = NS(dsa_enable_prefill_cp=True, self_attn=NS(o_proj=NS(
            use_decode_attn_tp=True, tp_size=1, reduce_results=False)))
        mode = Mock()
        mode.is_decode.return_value = True
        batch = NS(forward_mode=mode)
        x = torch.zeros((40, 6144), dtype=torch.bfloat16)
        with patch.dict(sys.modules, modules), patch.object(fusion, "ENABLED", True):
            self.assertTrue(fusion.eligible(layer, batch, x, x))
            self.assertFalse(comm.can_use_flashinfer_allreduce.call_args.kwargs["use_attn_tp_group"])
            mode.is_decode.return_value = False
            self.assertFalse(fusion.eligible(layer, batch, x, x))
            mode.is_decode.return_value = True
            group.ranks = list(reversed(range(8)))
            self.assertFalse(fusion.eligible(layer, batch, x, x))
            group.ranks = list(range(8))
            comm.can_use_flashinfer_allreduce.return_value = False
            self.assertFalse(fusion.eligible(layer, batch, x, x))
            comm.can_use_flashinfer_allreduce.return_value = True
            parallel.attn_dp_size = 2
            self.assertFalse(fusion.eligible(layer, batch, x, x))
            parallel.attn_dp_size = 1
            self.assertFalse(fusion.eligible(layer, batch, x[:0], x[:0]))

    def test_launcher_defaults_off_and_guards_experiment(self):
        base = drop_options(args_for(16384, [8192, 16384]), ["--speculative-"])
        def check(argv):
            with patch.object(launch, "validate"), patch.dict(os.environ, {"SGLANG_ENABLE_CP_V2": "1"}):
                return launch.check_profile(launch.configure(argv))
        with patch.dict(os.environ, {"SGLANG_GLM53_CP_DECODE_FUSION": "1"}):
            launch.configure_runtime_env(check(base))
            self.assertEqual(os.environ["SGLANG_GLM53_CP_DECODE_FUSION"], "0")
        for option in (["--glm53-cp-decode-fusion", "attention"], ["--glm53-cp-decode-fusion=attention"]):
            argv = base + option
            result = launch.runtime_argv(argv)
            self.assertEqual(result, base + ["--flashinfer-allreduce-fusion-backend", "trtllm"])
            with patch.dict(os.environ):
                launch.configure_runtime_env(check(argv))
                self.assertEqual(os.environ["SGLANG_GLM53_CP_DECODE_FUSION"], "1")
        for bad in (
            args_for(16384, [16384]),
            replace_value(base, "--cuda-graph-backend-decode", "disabled"),
            replace_value(base, "--max-running-requests", 2049),
            base + ["--flashinfer-allreduce-fusion-backend", "mnnvl"],
            base + ["--enforce-disable-flashinfer-allreduce-fusion"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                check(bad + ["--glm53-cp-decode-fusion", "attention"])


if __name__ == "__main__":
    unittest.main()
