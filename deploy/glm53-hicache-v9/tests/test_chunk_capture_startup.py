"""Replay the configuration and verify-input construction that failed at startup."""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_chunk_resolved_profile as profile

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def method(path, cls_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class CaptureStartupTest(unittest.TestCase):
    def setUp(self):
        profile.ResolvedProfileTest.setUp(self)
        self.model_declarations = list(self.args._resolved_overrides)
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.runtime_context": NS(get_exec=lambda: None),
            "sglang.srt.utils": NS(is_cuda=lambda: True),
        }))
        path = ROOT / "model_executor/cuda_graph_config.py"
        spec = importlib.util.spec_from_file_location("capture_graph_config_test", path)
        self.graph = importlib.util.module_from_spec(spec)
        self.enterContext(patch.dict(sys.modules, {spec.name: self.graph}))
        spec.loader.exec_module(self.graph)

    def policy(self):
        path = ROOT / "speculative/dflash_utils.py"
        tree = ast.parse(path.read_text())
        names = {"_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS"}
        nodes = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        return profile.extract(
            "speculative/dflash_utils.py", {"resolve_dflash_verify_mask_policy"}, namespace
        )["resolve_dflash_verify_mask_policy"]

    def capture_info(self, backend, *, draft=False, eagle=False):
        policy = self.policy()
        hidden = NS(NULL="null", FULL="full")
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.speculative.dflash_info": NS(DFlashVerifyInput=NS),
            "sglang.srt.speculative.eagle_info": NS(EagleVerifyInput=NS),
            "sglang.srt.speculative.dflash_utils": NS(resolve_dflash_verify_mask_policy=policy),
        }))
        runner = NS(
            model_runner=NS(
                spec_algorithm=NS(
                    is_eagle=lambda: eagle, is_standalone=lambda: False,
                    is_dflash_family=lambda: not eagle, is_ngram=lambda: False,
                ),
                is_draft_worker=draft, attn_backend=backend,
                model_config=NS(hidden_size=16), dtype="bf16", device="cpu",
            ),
            captured_req_width=8, speculative_num_steps=1, speculative_num_draft_tokens=8,
            buffers=NS(custom_mask=object()),
            _capture_ragged_verify_layout=lambda _: None,
        )
        get_info = method(
            "model_executor/runner/decode_cuda_graph_runner.py", "DecodeCudaGraphRunner",
            "get_spec_info", {
                "CaptureHiddenMode": hidden,
                "get_spec": lambda: NS(speculative_eagle_topk=1),
                "torch": NS(zeros=lambda *a, **kw: None),
            },
        )
        return runner, get_info

    def test_capture_inputs_pass_causal_validation_for_every_graph_bucket(self):
        backend = type("DeepseekSparseAttnBackend", (), {"glm53_dflash_dcp": True})()
        runner, get_info = self.capture_info(backend)
        for bs in (1, 2, 4, 8, 16, 32):
            with self.subTest(bs=bs):
                info = get_info(runner, bs * 8)
                batch = NS(spec_info=info, batch_size=bs, input_ids=range(bs * 8))
                self.dflash.validate_verify_layout(batch, 8)
                self.assertIsNone(info.custom_mask)
                self.assertEqual(info.capture_hidden_mode, "full")

    def test_mask_policy_is_scoped_and_preserves_eagle_tree_masks(self):
        policy = self.policy()
        for enabled in (False, True):
            backend = type("DeepseekSparseAttnBackend", (), {"glm53_dflash_dcp": enabled})()
            wrapped = NS(full_attn_backend=NS(full_attn_backend=backend))
            self.assertEqual(policy(wrapped), ("DeepseekSparseAttnBackend", not enabled))
        unknown = type("OtherAttentionBackend", (), {"glm53_dflash_dcp": True})()
        self.assertTrue(policy(unknown)[1])
        runner, get_info = self.capture_info(backend, eagle=True)
        self.assertIs(get_info(runner, 8).custom_mask, runner.buffers.custom_mask)
        runner, get_info = self.capture_info(backend, draft=True)
        self.assertIsNone(get_info(runner, 8).custom_mask)

    def test_custom_and_ragged_masks_remain_rejected(self):
        for changes in ({"custom_mask": object()}, {"ragged_verify_layout": object()}):
            info = NS(draft_token_num=8, custom_mask=None, ragged_verify_layout=None)
            vars(info).update(changes)
            with self.assertRaises(ValueError):
                self.dflash.validate_verify_layout(
                    NS(spec_info=info, batch_size=1, input_ids=range(8)), 8
                )

    def resolve_graph(self, *, chunk=16384, backend="breakable"):
        args = self.args
        # Each launch resolves a fresh record; a previous graph declaration
        # would otherwise become an explicit JSON input on the second parse.
        args._resolved_overrides = list(self.model_declarations)
        args.cuda_graph_config = None
        for key, value in {
            "disable_cuda_graph": False, "disable_prefill_cuda_graph": False,
            "disable_decode_cuda_graph": False, "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": backend, "cuda_graph_max_bs_decode": 32,
            "cuda_graph_max_bs_prefill": chunk, "cuda_graph_bs_decode": [1, 2, 4, 8, 16, 32],
            "cuda_graph_bs_prefill": [n for n in (8192, 16384, 32768) if n <= chunk],
            "cuda_graph_tc_compiler": None,
        }.items():
            setattr(args, key, value)
        namespace = vars(self.overrides).copy()
        namespace.update(vars(self.graph))
        namespace["declare_resolution"] = (
            lambda args, source, **fields: args._resolved_overrides.append((source, fields))
        )
        parse = profile.extract(
            "arg_groups/cuda_graph_hook.py", {"parse_cuda_graph_config"}, namespace
        )["parse_cuda_graph_config"]
        parse(args)
        return namespace

    def model_cp_rule(self, namespace):
        path = ROOT / "arg_groups/model_hook.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "handle_model_specific_adjustments")
        # Execute the DSA-specific CP graph rule in its actual model hook.
        platform = next(
            n for n in ast.walk(function)
            if isinstance(n, ast.If) and ast.unparse(n.test)
            == "not get_platform().is_npu and (not get_platform().is_xpu)"
        )
        cp_rule = next(n for n in platform.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == "cfg.enable_prefill_cp")
        namespace.update(server_args=self.args, cfg=self.overrides.resolving_view(self.args))
        exec(compile(ast.Module(body=[cp_rule], type_ignores=[]), str(path), "exec"), namespace)
        return self.overrides.resolving_view(self.args).cuda_graph_config

    def test_explicit_prefill_graph_survives_model_rule_at_8k_16k_32k(self):
        for chunk in (8192, 16384, 32768):
            with self.subTest(chunk=chunk):
                config = self.model_cp_rule(self.resolve_graph(chunk=chunk))
                self.assertEqual(config.prefill.backend, "breakable")
                self.assertEqual(config.prefill.max_bs, chunk)
                self.assertEqual(config.prefill.bs[-1], chunk)
                self.assertEqual(config.decode.backend, "full")
                self.assertEqual(config.decode.bs, [1, 2, 4, 8, 16, 32])

    def test_graph_opt_out_and_unsupported_profiles_stay_disabled(self):
        for backend in ("disabled", "full", "tc_piecewise"):
            config = self.model_cp_rule(self.resolve_graph(backend=backend))
            self.assertEqual(config.prefill.backend, "disabled")
        for key, value in (("tp_size", 4), ("dsa_prefill_backend", "tilelang")):
            old = getattr(self.args, key)
            setattr(self.args, key, value)
            config = self.model_cp_rule(self.resolve_graph())
            self.assertEqual(config.prefill.backend, "disabled")
            setattr(self.args, key, old)
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_BCG": "0"}):
            config = self.model_cp_rule(self.resolve_graph())
            self.assertEqual(config.prefill.backend, "disabled")


if __name__ == "__main__":
    unittest.main()
