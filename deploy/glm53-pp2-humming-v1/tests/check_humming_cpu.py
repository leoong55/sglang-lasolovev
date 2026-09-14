"""CPU tests for backend selection, signed packing, EP metadata and controls.

CUDA-dependent SGLang imports are isolated with small fixtures. Torch tensor
operations and Humming's checkpoint schema run for real; no CUDA correctness
or throughput claim is made by this suite.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

ROOT = Path(importlib.util.find_spec("sglang").submodule_search_locations[0])


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def attrs(tensor, values):
    for key, value in values.items():
        setattr(tensor, key, value)


class QuantConfig:
    @classmethod
    def get_from_keys(cls, values, keys):
        return next(values[key] for key in keys if key in values)


class Method:
    runner = None

    def __init__(self, config=None):
        self.config = config


class FakeRunner:
    def __init__(self, backend, config):
        self.runner_core = object()
        self.fused_func = object()
        self.config = config


class Linear(torch.nn.Module):
    pass


class MoE(torch.nn.Module):
    pass


@contextlib.contextmanager
def subjects(backend="humming", a2a="none"):
    fp8, unquant = (
        type("Fp8Linear", (Method,), {}),
        type("UnquantLinear", (Method,), {}),
    )
    backend_value = NS(is_humming=lambda: backend == "humming")
    stubs = {
        "sglang.srt.environ": NS(envs=NS()),
        "sglang.srt.layers.linear": NS(LinearBase=Linear, set_weight_attrs=attrs),
        "sglang.srt.layers.moe": NS(
            MoeRunner=FakeRunner,
            MoeRunnerBackend=NS(HUMMING="humming"),
            MoeRunnerConfig=NS,
            get_moe_runner_backend=lambda: backend_value,
        ),
        "sglang.srt.layers.moe.utils": NS(
            get_moe_a2a_backend=lambda: NS(is_none=lambda: a2a == "none")
        ),
        "sglang.srt.layers.moe.fused_moe_triton": NS(
            FusedMoE=MoE, FusedMoeWeightScaleSupported=NS(GROUP=NS(value="group"))
        ),
        "sglang.srt.layers.quantization.base_config": NS(
            FusedMoEMethodBase=Method,
            QuantizationConfig=QuantConfig,
            QuantizeMethodBase=Method,
            LinearMethodBase=Method,
        ),
        "sglang.srt.layers.quantization.fp8": NS(Fp8LinearMethod=fp8),
        "sglang.srt.layers.quantization.unquant": NS(
            UnquantizedLinearMethod=unquant, UnquantizedFusedMoEMethod=Method
        ),
        "sglang.srt.layers.quantization.utils": NS(
            is_layer_skipped=lambda prefix, ignored: prefix in ignored
        ),
        "sglang.srt.utils": NS(set_weight_attrs=attrs),
        "sglang.srt.layers.parameter": NS(
            **{
                name: type(name, (), {})
                for name in (
                    "BasevLLMParameter",
                    "BlockQuantScaleParameter",
                    "ChannelQuantScaleParameter",
                    "GroupQuantScaleParameter",
                    "ModelWeightParameter",
                    "PackedvLLMParameter",
                    "PerTensorScaleParameter",
                    "RowvLLMParameter",
                )
            }
        ),
    }
    with patch.dict(sys.modules, stubs):
        w4 = load(
            "sglang.srt.layers.quantization.w4afp8",
            "srt/layers/quantization/w4afp8.py",
        )
        humming = load(
            "sglang.srt.layers.quantization.humming",
            "srt/layers/quantization/humming.py",
        )
        subject = load(
            "sglang.srt.layers.quantization.w4afp8_humming",
            "srt/layers/quantization/w4afp8_humming.py",
        )
        yield w4, humming, subject, fp8, unquant


def runner_config(**changes):
    values = dict(
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        no_combine=False,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        hidden_size=128,
        intermediate_size_per_partition=128,
        num_experts=16,
        num_local_experts=2,
    )
    values.update(changes)
    return NS(**values)


class W4HummingTest(unittest.TestCase):
    def test_opt_in_is_moe_only_and_cutlass_remains_default(self):
        for backend in ("auto", "cutlass", "humming"):
            with (
                self.subTest(backend=backend),
                subjects(backend) as (w4, _, humming, fp8, unquant),
            ):
                config = w4.W4AFp8Config(ignored_layers=["skip"])
                self.assertIsInstance(
                    config.get_quant_method(Linear(), "attention"), fp8
                )
                self.assertIsInstance(
                    config.get_quant_method(Linear(), "skip"), unquant
                )
                method = config.get_quant_method(MoE(), "experts")
                expected = (
                    humming.W4AFp8HummingMoEMethod
                    if backend == "humming"
                    else w4.W4AFp8MoEMethod
                )
                self.assertIs(type(method), expected)

    def test_signed_packing_all_byte_values_and_nonuniform_scales(self):
        with subjects() as (_, humming, _, _, _):
            schema = humming._W4AFp8CheckpointWeightSchema(128)
            packed = (
                torch.arange(256, dtype=torch.int16)
                .to(torch.uint8)
                .view(torch.int8)
                .reshape(1, 4, 64)
            )
            scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float32).reshape(
                1, 4, 1
            )
            _, converted = schema.convert_humming(
                dict(weight=packed, weight_scale_inv=scales),
                [2, 2],
                [128],
                torch.bfloat16,
                num_experts=1,
            )
            unsigned = converted["weight"].view(torch.uint8)
            biased = (
                torch.stack((unsigned & 15, unsigned >> 4), -1).flatten(-2).float() - 8
            )
            original = packed.view(torch.uint8)
            raw_nibbles = (
                torch.stack((original & 15, original >> 4), -1)
                .flatten(-2)
                .to(torch.int16)
            )
            reference = torch.where(
                raw_nibbles < 8, raw_nibbles, raw_nibbles - 16
            ).float()
            torch.testing.assert_close(biased, reference, rtol=0, atol=0)
            torch.testing.assert_close(
                converted["weight_scale"], scales.to(torch.bfloat16), rtol=0, atol=0
            )

    def test_post_load_preserves_global_count_and_releases_original_parameters(self):
        from humming.layer import HummingMethod

        with subjects() as (w4, _, subject, _, _):
            layer = MoE()
            layer.num_experts, layer.num_local_experts = 256, 64
            method = subject.W4AFp8HummingMoEMethod(w4.W4AFp8Config())
            method.create_weights(
                layer, 64, 128, 128, torch.bfloat16, weight_loader=lambda *a, **k: None
            )
            for name in ("w13", "w2"):
                getattr(layer, name + "_weight").data.fill_(-120)
                getattr(layer, name + "_weight_scale_inv").data.fill_(0.25)
            old = weakref.ref(layer.w13_weight)
            method.create_moe_runner(
                layer, runner_config(num_experts=256, num_local_experts=64)
            )
            core = method.runner.runner_core
            with (
                patch.object(
                    torch.Tensor,
                    "is_cuda",
                    new_callable=PropertyMock,
                    return_value=True,
                ),
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
                patch.object(HummingMethod, "prepare_layer_meta") as prepare,
                patch.object(HummingMethod, "transform_humming_layer") as transform,
            ):
                method.process_weights_after_loading(layer)
                method.process_weights_after_loading(layer)
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(transform.call_count, 2)
            self.assertEqual(
                [c.kwargs["num_experts"] for c in prepare.call_args_list], [64, 64]
            )
            self.assertEqual(layer.num_experts, 256)
            self.assertIsNone(old())
            self.assertIs(method.runner.runner_core, core)
            self.assertIsNone(method.runner.fused_func)
            self.assertEqual(layer.w13_weight.dtype, torch.int32)
            self.assertFalse(hasattr(layer, "w13_weight_scale_inv"))
            self.assertFalse(hasattr(layer, "w13_input_scale"))

    def test_unsupported_runner_modes_fail_before_conversion(self):
        bad = [
            dict(activation="gelu"),
            dict(is_gated=False),
            dict(apply_router_weight_on_input=True),
            dict(no_combine=True),
            dict(num_fused_shared_experts=1),
            dict(params_dtype=torch.float16),
            dict(gemm1_alpha=1.7),
        ]
        with subjects() as (w4, _, subject, _, _):
            for fields in bad:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    subject.W4AFp8HummingMoEMethod(w4.W4AFp8Config()).create_moe_runner(
                        MoE(), runner_config(**fields)
                    )
        with (
            subjects(a2a="deepep") as (w4, _, subject, _, _),
            self.assertRaisesRegex(ValueError, "none"),
        ):
            subject.W4AFp8HummingMoEMethod(w4.W4AFp8Config()).create_moe_runner(
                MoE(), runner_config()
            )


if __name__ == "__main__":
    unittest.main()
