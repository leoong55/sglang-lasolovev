"""Connect toggles to real resolved views, DSA gates and graph capacity."""

import ast
import logging
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_chunk_resolved_profile as profile
import test_chunk_capture_startup as capture


class RuntimeControlsTest(unittest.TestCase):
    def setUp(self):
        capture.CaptureStartupTest.setUp(self)

    resolve_graph = capture.CaptureStartupTest.resolve_graph
    model_cp_rule = capture.CaptureStartupTest.model_cp_rule

    def backend_gate(self):
        path = profile.ROOT / "layers/attention/dsa_backend.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekSparseAttnBackend")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.ImportFrom) and n.module == "sglang.srt.layers.cp.glm53_dflash")
        guard = init.body[start + 2]
        body = [*init.body[start:start + 2], *guard.body[:3]]
        backend = NS(dcp_enabled=True)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), {"self": backend, "model_runner": NS(server_args=self.args)})
        return backend

    def test_all_four_feature_combinations_pass_real_runtime_gates(self):
        namespace = vars(self.overrides).copy()
        namespace.update(logger=logging.getLogger(__name__), use_mla_backend=lambda _: True)
        validate = profile.extract("arg_groups/hicache_hook.py", {"resolve_hicache_dcp_compatibility"}, namespace)["resolve_hicache_dcp_compatibility"]
        for spec in (None, "DFLASH"):
            for cache in (False, True):
                self.args.speculative_algorithm = spec
                self.args.enable_hierarchical_cache = cache
                with patch.dict(os.environ, {
                    "SGLANG_GLM53_DFLASH_DCP": "1" if spec else "0",
                    "SGLANG_GLM53_HICACHE_DCP": "1" if cache else "0",
                }):
                    validate(self.args)
                    backend = self.backend_gate()
                    self.assertEqual(backend.glm53_dflash_dcp, spec == "DFLASH")
                    self.assertEqual(self.hicache.supports_hicache_cp_dcp(self.args), cache)
                    self.assertEqual(self.model_cp_rule(self.resolve_graph()).prefill.backend, "breakable")

    def test_no_speculation_keeps_model_and_layout_checks(self):
        self.args.speculative_algorithm = None
        for name, value in (("tp_size", 4), ("dcp_size", 2), ("hicache_write_policy", "write_back"),
                            ("hicache_storage_backend", "mooncake"), ("hicache_io_backend", "kernel")):
            old = getattr(self.args, name)
            setattr(self.args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.backend_gate()
            setattr(self.args, name, old)
        self.args._model_config.hf_config.num_hidden_layers = 45
        with self.assertRaises(ValueError):
            self.backend_gate()

    def test_other_speculative_algorithms_remain_outside_profile(self):
        for algorithm in ("EAGLE", "DSPARK", "STANDALONE"):
            self.args.speculative_algorithm = algorithm
            with self.subTest(algorithm=algorithm), self.assertRaises(ValueError):
                self.backend_gate()

    def test_graph_pool_capacity_and_verify_width_above_32(self):
        for width in (1, 8):
            for running in (40, 48, 64, 96, 128):
                requested = sorted({1, 2, 4, 8, 16, 32, running})
                runtime = NS(graph=NS(cuda_graph_config=NS(decode=NS(bs=requested))), overlap=NS(enable_two_batch_overlap=False))
                namespace = dict(
                    get_exec=lambda: runtime,
                    get_flags=lambda: NS(capture=NS(enable_torch_compile=False)),
                    get_cuda_graph_batch_size_alignment=lambda: 8,
                    get_cuda_graph_max_batch_size=lambda n: n,
                )
                choose = profile.extract("model_executor/runner/base_cuda_graph_runner.py", {"get_batch_sizes_to_capture"}, namespace)["get_batch_sizes_to_capture"]
                sizes, _ = choose(NS(req_to_token_pool=NS(size=running)), width)
                self.assertEqual(sizes[-1], running)
                self.assertTrue(all(bs * width % 8 == 0 for bs in sizes))
                if width == 8:
                    for bs in sizes:
                        info = NS(draft_token_num=8, custom_mask=None, ragged_verify_layout=None)
                        self.dflash.validate_verify_layout(NS(spec_info=info, batch_size=bs, input_ids=range(bs * 8)), 8)


if __name__ == "__main__":
    unittest.main()
