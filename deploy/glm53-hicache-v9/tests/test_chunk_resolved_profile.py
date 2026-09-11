"""Exercise the real resolved views and GLM53 gates without CUDA/model loading."""

import ast
import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def extract(path, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(body) == len(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[]))
    namespace = {} if namespace is None else namespace
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "layers/cp" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResolvedProfileTest(unittest.TestCase):
    def setUp(self):
        names = {
            "ResolvedView",
            "ResolvingConfig",
            "resolving_view",
            "resolved_view",
            "_declaration_overlay",
            "attention_backends_of",
            "model_config_of",
            "record_of",
        }
        actual = extract("arg_groups/model_override_base.py", names)
        self.overrides = NS(**{name: actual[name] for name in names})
        self.bcg = load_helper("glm53_bcg")
        self.dflash = load_helper("glm53_dflash")
        self.hicache = load_helper("glm53_hicache")
        model = NS(
            hf_config=NS(num_hidden_layers=78, architectures=["GlmMoeDsaForCausalLM"]),
            is_multimodal=False,
        )
        # Raw generic backend is unset; resolution selects DSA. The DSA
        # implementation fields below match the user's 2026-09-11 startup.
        self.args = NS(
            _model_config=model,
            _resolved_overrides=[
                ("model", {"attention_backend": "dsa", "attn_cp_size": 8})
            ],
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            attn_cp_size=1,
            dsa_prefill_backend="flashmla_sparse_q8",
            dsa_decode_backend="flashmla_kv",
            speculative_algorithm="DFLASH",
            tp_size=8,
            ep_size=8,
            dp_size=1,
            pp_size=1,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            enable_cp_decode_attn_tp=True,
            dcp_size=4,
            dcp_comm_backend="ag_rs",
            kv_cache_dtype="fp8_e4m3",
            page_size=64,
            quantization="w4afp8",
            moe_a2a_backend="none",
            enable_hierarchical_cache=True,
            hicache_storage_backend=None,
            hicache_host_memory_mode="cache",
            hicache_write_policy="write_through",
            hicache_mem_layout="layer_first",
            hicache_io_backend="direct",
            enable_lmcache=False,
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            cuda_graph_config=NS(prefill=NS(backend="breakable")),
        )
        modules = {
            "sglang.srt.arg_groups.overrides": self.overrides,
            "sglang.srt.configs.model_config": NS(
                ModelConfig=object, is_deepseek_v4=lambda _: False
            ),
            "sglang.srt.layers.cp.glm53_bcg": self.bcg,
            "sglang.srt.layers.cp.glm53_dflash": self.dflash,
            "sglang.srt.layers.cp.glm53_hicache": self.hicache,
        }
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(
            patch.dict(os.environ, {
                "SGLANG_GLM53_PREFILL_BCG": "1",
                "SGLANG_GLM53_DFLASH_DCP": "1",
                "SGLANG_GLM53_HICACHE_DCP": "1",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            })
        )

    def test_logged_dsa_profile_is_recognized_by_all_three_gates(self):
        resolved = self.overrides.resolved_view(self.args)
        self.assertEqual(self.overrides.attention_backends_of(resolved), ("dsa", "dsa"))
        self.assertTrue(self.dflash.supports_dflash_dcp(self.args))
        self.assertTrue(self.bcg.supports(self.args))
        self.assertTrue(self.hicache.supports_hicache_dflash(resolved))

    def test_real_backend_hicache_guard_accepts_logged_profile(self):
        path = ROOT / "layers/attention/dsa_backend.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekSparseAttnBackend"
        )
        init = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        start = next(
            i for i, n in enumerate(init.body)
            if isinstance(n, ast.ImportFrom)
            and n.module == "sglang.srt.layers.cp.glm53_dflash"
        )
        guard = init.body[start + 2]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.unparse(guard.test), "self.dcp_enabled")
        # Stop before GPU/backend validation; execute the actual profile gate.
        body = [*init.body[start : start + 2], *guard.body[:3]]
        backend = NS(dcp_enabled=True)
        exec(
            compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
            {"self": backend, "model_runner": NS(server_args=self.args)},
        )
        self.assertTrue(backend.glm53_dflash_dcp)

    def test_actual_graph_compatibility_preserves_breakable(self):
        cp = extract(
            "layers/cp/bcg.py", {"supports_prefill_cp_bcg"}, vars(self.overrides).copy()
        )
        self.enterContext(
            patch.dict(sys.modules, {
                "sglang.srt.layers.cp.bcg": NS(
                    supports_prefill_cp_bcg=cp["supports_prefill_cp_bcg"]
                )
            })
        )
        declarations = []
        namespace = vars(self.overrides).copy()
        namespace.update(
            logger=logging.getLogger(__name__),
            declare_resolution=lambda *a, **kw: declarations.append(kw),
            with_phase=lambda config, phase, **kw: kw,
            Phase=NS(PREFILL="prefill"),
            Backend=NS(DISABLED="disabled"),
        )
        actual = extract(
            "arg_groups/cuda_graph_hook.py",
            {"disable_breakable_cudagraph_if_incompatible"},
            namespace,
        )
        actual["disable_breakable_cudagraph_if_incompatible"](self.args)
        self.assertEqual(declarations, [])
        self.args.dsa_prefill_backend = "tilelang"
        actual["disable_breakable_cudagraph_if_incompatible"](self.args)
        self.assertEqual(len(declarations), 1)

    def test_resolved_dsa_implementations_override_raw_inputs(self):
        self.args.dsa_prefill_backend = "auto"
        self.args.dsa_decode_backend = "auto"
        self.args._resolved_overrides.append(("attention", {
            "dsa_prefill_backend": "flashmla_sparse_q8",
            "dsa_decode_backend": "flashmla_kv",
        }))
        self.assertTrue(self.dflash.supports_dflash_dcp(self.args))
        self.assertTrue(self.bcg.supports(self.args))

    def test_wrong_generic_or_internal_backends_are_rejected(self):
        for field, value in (
            ("prefill_attention_backend", "fa4"),
            ("decode_attention_backend", "fa4"),
            ("dsa_prefill_backend", "tilelang"),
            ("dsa_decode_backend", "flashmla_sparse_q8"),
        ):
            with self.subTest(field=field):
                self.args._resolved_overrides.append(("test", {field: value}))
                self.assertFalse(self.dflash.supports_dflash_dcp(self.args))
                self.assertFalse(self.bcg.supports(self.args))
                self.args._resolved_overrides.pop()

    def test_other_models_topologies_and_cache_policies_stay_rejected(self):
        for field, value in (("dcp_size", 2), ("tp_size", 4), ("moe_a2a_backend", "deepep")):
            with self.subTest(field=field):
                self.args._resolved_overrides.append(("test", {field: value}))
                self.assertFalse(self.dflash.supports_dflash_dcp(self.args))
                self.assertFalse(self.bcg.supports(self.args))
                self.args._resolved_overrides.pop()
        self.args._model_config.hf_config.num_hidden_layers = 45
        self.assertFalse(self.dflash.supports_dflash_dcp(self.args))
        self.assertFalse(self.bcg.supports(self.args))
        self.args.hicache_write_policy = "write_back"
        self.assertFalse(self.hicache.supports_hicache_dflash(self.args))
        self.args.hicache_write_policy = "write_through"
        with patch.dict(os.environ, {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0"}):
            self.assertFalse(self.hicache.supports_hicache_dflash(self.args))


if __name__ == "__main__":
    unittest.main()
