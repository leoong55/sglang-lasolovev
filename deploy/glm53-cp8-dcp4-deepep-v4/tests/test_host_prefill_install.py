"""Startup and transactional installer checks without GPU dependencies."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import torch
from test_host_prefill_deepep import CombineInput, runtime


class TestStartup(unittest.TestCase):
    def setUp(self):
        class W4:
            pass

        self.config = NS(
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=78,
            hidden_size=6144,
            n_routed_experts=256,
            num_experts_per_tok=8,
            n_shared_experts=1,
            hidden_act="silu",
            first_k_dense_replace=3,
        )
        self.moe = NS(
            experts=NS(
                quant_method=W4(),
                should_fuse_routed_scaling_factor_in_topk=False,
                num_local_experts=32,
                moe_ep_rank=3,
            ),
            is_nextn=False,
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=3,
        )
        self.group = NS(ranks=list(range(8)), rank_in_group=3, device_group=object())
        self.parallel = NS(
            tp_size=8,
            attn_cp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            cp_strategy="interleave",
            attn_cp_group=self.group,
            attn_cp_rank=3,
        )
        self.moe_config = NS(enable_eplb=False, ep_num_redundant_experts=0)
        self.buffer = MagicMock()
        self.dispatcher = MagicMock()
        self.graph_check = MagicMock(return_value=True)
        self.modules = {
            "sglang.srt.distributed": NS(get_tp_group=lambda: self.group),
            "sglang.srt.layers": NS(deep_gemm_wrapper=NS(ENABLE_JIT_DEEPGEMM=True)),
            "sglang.srt.layers.moe.token_dispatcher.deepep": NS(
                DeepEPBuffer=self.buffer,
                DeepEPDispatcher=self.dispatcher,
                DeepEPNormalCombineInput=CombineInput,
            ),
            "sglang.srt.layers.moe.utils": NS(
                DeepEPMode=NS(NORMAL="normal"),
                get_moe_a2a_backend=lambda: NS(is_none=lambda: True),
            ),
            "sglang.srt.layers.quantization.w4afp8": NS(W4AFp8MoEMethod=W4),
            "sglang.srt.model_executor.cuda_graph_config": NS(
                Backend=NS(DISABLED="disabled"),
                Phase=NS(PREFILL="prefill"),
                check_cuda_graph_backend=self.graph_check,
            ),
            "sglang.srt.runtime_context": NS(
                get_exec=lambda: NS(moe=self.moe_config),
                get_parallel=lambda: self.parallel,
            ),
        }

    def construct(self):
        with patch.dict(sys.modules, self.modules):
            return runtime.Glm53PrefillDeepEP(self.moe, self.config)

    def test_bf16_normal_buffer_allocated_at_construction(self):
        driver = self.construct()
        self.assertEqual((driver.cp_rank, driver.cp_size), (3, 8))
        self.buffer.get_deepep_buffer.assert_called_once_with(
            self.group.device_group, 6144, 2, "normal"
        )
        self.dispatcher.return_value.set_quant_config.assert_called_once_with(
            {"normal_dispatcher_output_dtype": "bf16"}
        )
        self.assertEqual(
            self.dispatcher.call_args.kwargs["params_dtype"], torch.bfloat16
        )
        self.assertEqual(self.dispatcher.call_args.kwargs["deepep_mode"], "normal")
        self.dispatcher.return_value.dispatch.assert_not_called()

    def test_incompatible_topology_and_model_fail_before_buffer(self):
        for obj, field, value in (
            (self.config, "num_hidden_layers", 61),
            (self.parallel, "moe_ep_size", 4),
            (self.parallel, "cp_strategy", "in-seq-split"),
            (self.moe_config, "enable_eplb", True),
            (self.moe.experts, "should_fuse_routed_scaling_factor_in_topk", True),
            (self.moe.experts, "moe_ep_rank", 0),
        ):
            with (
                self.subTest(field=field),
                patch.object(obj, field, value),
                self.assertRaises(ValueError),
            ):
                self.construct()
        self.buffer.get_deepep_buffer.assert_not_called()

    def test_prefill_graphs_rejected_before_buffer(self):
        self.graph_check.return_value = False
        with self.assertRaisesRegex(ValueError, "eager prefill"):
            self.construct()
        self.buffer.get_deepep_buffer.assert_not_called()


class TestInstaller(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "installer_tested", Path(__file__).resolve().parents[1] / "install.py"
        )
        self.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.installer)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sglang"
        self.bundle = Path(self.temp.name) / "bundle"
        self.original = b"VALUE = 'original'\n"
        self.updated = b"VALUE = 'updated'\n"
        self.added = b"VALUE = 'new'\n"
        (self.root / "srt").mkdir(parents=True)
        (self.root / "srt/existing.py").write_bytes(self.original)
        rows = []
        for name, before, after in (
            ("added.py", None, self.added),
            ("existing.py", self.original, self.updated),
        ):
            path = f"python/sglang/srt/{name}"
            target = self.bundle / "overlay" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(after)
            rows.append(
                dict(
                    path=path,
                    base_sha256=self.installer.digest(before) if before else None,
                    patched_sha256=self.installer.digest(after),
                )
            )
        (self.bundle / "base-files.json").write_text(
            json.dumps(dict(base_commit="base", files=rows))
        )

    def test_new_file_install_verify_and_idempotence(self):
        for verify in (False, True, False):
            self.installer.install(self.root, self.bundle, verify_only=verify)
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), self.updated)
        self.assertEqual((self.root / "srt/added.py").read_bytes(), self.added)

    def test_mismatch_validates_all_before_any_mutation(self):
        (self.root / "srt/existing.py").write_bytes(b"other = 1\n")
        with self.assertRaisesRegex(RuntimeError, "Source mismatch"):
            self.installer.install(self.root, self.bundle)
        self.assertFalse((self.root / "srt/added.py").exists())
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), b"other = 1\n")

    def test_failed_compile_rolls_back_added_and_existing_files(self):
        with patch.object(
            self.installer.py_compile,
            "compile",
            side_effect=[None, RuntimeError("failure")],
        ):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                self.installer.install(self.root, self.bundle)
        self.assertFalse((self.root / "srt/added.py").exists())
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), self.original)

    def test_verify_only_never_installs(self):
        with self.assertRaisesRegex(RuntimeError, "Source mismatch"):
            self.installer.install(self.root, self.bundle, verify_only=True)
        self.assertFalse((self.root / "srt/added.py").exists())
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), self.original)
