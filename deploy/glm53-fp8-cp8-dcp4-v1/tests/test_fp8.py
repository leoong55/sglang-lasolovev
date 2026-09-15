"""Check real FP8 profile resolution, native runner selection and collectives.

These tests execute source functions with CPU-only dependencies substituted.
They do not establish CUDA graph correctness or end-to-end model quality.
"""

import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import launch_fp8
from render import render

REPO = KIT.parents[1]
os.environ.setdefault("SGLANG_SOURCE_ROOT", str(REPO))
if os.environ.get("FP8_INSTALLED") == "1":
    ROOT = launch_fp8.package_root() / "srt"
else:
    ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiles = load_file(
    "resolved_fp8_test_base",
    launch_fp8.LEGACY / "tests/test_chunk_resolved_profile.py",
)
profiles.ROOT = ROOT
fp8_contract = load_file("fp8_contract_tested", ROOT / "layers/cp/glm53_fp8.py")


def checkpoint():
    return dict(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=78,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        quantization_config=dict(
            quant_method="fp8",
            weight_block_size=[128, 128],
            activation_scheme="dynamic",
        ),
    )


class FP8ResolvedTests(profiles.ResolvedProfileTest):
    def setUp(self):
        super().setUp()
        self.args.quantization = None
        self.args.moe_runner_backend = "auto"
        self.args._model_config.hf_config = NS(**checkpoint())
        self.enterContext(
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.cp.glm53_fp8": fp8_contract},
            )
        )

    def all_gates(self):
        return (
            self.bcg.supports(self.args),
            self.dflash.supports_dflash_dcp(self.args),
            self.hicache.supports_hicache_cp_dcp(self.args),
        )

    def test_autodetection_and_explicit_fp8_keep_all_features(self):
        for quant in (None, "fp8"):
            for runner in ("auto", "triton"):
                self.args.quantization = quant
                self.args.moe_runner_backend = runner
                self.assertEqual(self.all_gates(), (True, True, True))

    def test_wrong_fp8_storage_and_runner_are_rejected(self):
        quant = self.args._model_config.hf_config.quantization_config
        for key, value in (
            ("quant_method", "w4afp8"),
            ("weight_block_size", [1, 128]),
            ("activation_scheme", "static"),
        ):
            with self.subTest(key=key), patch.dict(quant, {key: value}):
                self.assertEqual(self.all_gates(), (False, False, False))
        for runner in ("humming", "deep_gemm", "cutlass"):
            self.args.moe_runner_backend = runner
            self.assertEqual(self.all_gates(), (False, False, False))

    def test_w4_contract_still_works(self):
        self.args.quantization = "w4afp8"
        self.args.moe_runner_backend = "humming"
        self.args._model_config.hf_config.quantization_config = {}
        self.assertEqual(self.all_gates(), (True, True, True))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.config_path = Path(directory) / "config.json"
        self.config_path.write_text(json.dumps(checkpoint()))
        self.container = render("validation")[0]["spec"]["template"]["spec"][
            "containers"
        ][0]
        self.argv = self.container["args"]
        self.argv[self.argv.index("--model-path") + 1] = directory
        self.enterContext(
            patch.dict(
                os.environ, {e["name"]: e["value"] for e in self.container["env"]}
            )
        )

    def prepare(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            return launch_fp8.prepare([*self.argv, *extra])

    def test_actual_manifest_defaults_survive_launcher(self):
        profile, runtime = self.prepare()
        self.assertEqual(profile.moe_runner_backend, "auto")
        self.assertFalse(profile.disable_shared_experts_fusion)
        self.assertEqual(profile.max_running_requests, 40)
        self.assertEqual(profile.cuda_graph_bs_decode, [1, 2, 4, 8, 16, 32, 40])
        self.assertNotIn("--quantization", runtime)
        self.assertNotIn("--moe-runner-backend", runtime)
        self.assertNotIn("--disable-shared-experts-fusion", runtime)
        self.assertEqual(os.environ["SGLANG_GLM53_HUMMING_EP_AWARE"], "0")
        self.assertEqual(os.environ["SGLANG_GLM53_DFLASH_DCP"], "1")
        self.assertEqual(os.environ["SGLANG_GLM53_HICACHE_DCP"], "1")
        self.assertEqual(os.environ["SGLANG_GLM53_PREFILL_BCG"], "1")
        self.assertEqual(os.environ["SGLANG_GLM53_DRAFT_CACHE_WINDOW"], "2048")

    def test_explicit_fp8_is_preserved(self):
        _, runtime = self.prepare(["--quantization", "fp8"])
        self.assertEqual(runtime[runtime.index("--quantization") + 1], "fp8")

    def test_w4_and_unsupported_runners_fail_before_loading(self):
        for extra in (
            ["--quantization", "w4afp8"],
            ["--moe-runner-backend", "humming"],
            ["--moe-runner-backend", "deep_gemm"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                self.prepare(extra)
        config = checkpoint()
        config["quantization_config"]["quant_method"] = "w4afp8"
        self.config_path.write_text(json.dumps(config))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.prepare()

    def test_manifest_uses_requested_weights_and_isolated_service(self):
        deployment, service = render("validation")
        pod = deployment["spec"]["template"]["spec"]
        model = next(v for v in pod["volumes"] if v["name"] == "model")
        self.assertEqual(model["persistentVolumeClaim"]["claimName"], "sglang-fp8-pvc")
        args = pod["containers"][0]["args"]
        self.assertEqual(args[args.index("--model-path") + 1], "/mnt/model-pvc-fp8")
        self.assertEqual(
            service["spec"]["selector"], deployment["spec"]["selector"]["matchLabels"]
        )


def method(path, class_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    code = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    exec(compile(code, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class Backend:
    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        if name.startswith("is_"):
            return lambda: self.value == name[3:]
        raise AttributeError(name)


class NativeExecutionTests(unittest.TestCase):
    def test_actual_fp8_auto_selects_triton_for_none_a2a(self):
        auto, none = Backend("auto"), Backend("none")
        namespace = {"get_moe_runner_backend": lambda: auto, "_is_hip": False}
        get_deepgemm = method(
            "layers/quantization/fp8.py",
            "Fp8MoEMethod",
            "is_deepgemm_moe_runner_backend_enabled",
            namespace,
        )
        with patch.dict(
            sys.modules,
            {"sglang.srt.layers.moe.utils": NS(get_moe_a2a_backend=lambda: none)},
        ):
            self.assertFalse(get_deepgemm(auto, none))
            namespace.update(
                MoeRunnerBackend=NS(TRITON=Backend("triton")),
                MoeRunner=lambda backend, config: NS(backend=backend),
            )
            create = method(
                "layers/quantization/fp8.py",
                "Fp8MoEMethod",
                "create_moe_runner",
                namespace,
            )
            target = NS(
                is_deepgemm_moe_runner_backend_enabled=lambda: get_deepgemm(auto, none)
            )
            create(target, NS(), NS())
            self.assertEqual(target.runner.backend.value, "triton")
            self.assertTrue(target._owns_moe_runner)

    def test_native_h200_ep8_shared_default_and_allreduce(self):
        namespace = dict(
            quant_blocks_shared_experts_fusion=lambda _: False,
            get_exec=lambda: NS(moe=NS(enforce_shared_experts_fusion=False)),
            is_sbo_enabled=lambda: False,
            is_tbo_enabled=lambda: False,
            is_deepep_class_backend=lambda: False,
            _is_cuda=True,
            _is_hip=False,
            _is_musa=False,
            torch=NS(cuda=NS(get_device_capability=lambda _: (9, 0))),
            get_parallel=lambda: NS(moe_ep_size=8),
        )
        reason = method(
            "models/deepseek_v2.py",
            "DeepseekV2ForCausalLM",
            "shared_experts_fusion_disable_reason",
            namespace,
        )
        cfg = NS(
            architectures=["GlmMoeDsaForCausalLM"],
            n_shared_experts=1,
            n_routed_experts=256,
        )
        result = reason(
            NS(fused_shared_experts_architecture="GlmMoeDsaForCausalLM"), cfg, NS()
        )
        self.assertIn("under expert parallelism", result)
        namespace = dict(
            should_skip_mlp_all_reduce=lambda: False,
            get_parallel=lambda: NS(dwdp_size=1),
            should_use_dp_reduce_scatterv=lambda: False,
            should_use_flashinfer_cutlass_moe_fp4_allgather=lambda: False,
            get_moe_a2a_backend=lambda: Backend("none"),
        )
        functions = profiles.extract(
            "layers/moe/utils.py", {"should_skip_post_experts_all_reduce"}, namespace
        )
        for is_tp in (True, False):
            self.assertFalse(
                functions["should_skip_post_experts_all_reduce"](is_tp_path=is_tp)
            )


if __name__ == "__main__":
    unittest.main()
