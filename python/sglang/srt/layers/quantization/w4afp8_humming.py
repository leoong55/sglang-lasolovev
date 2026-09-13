# SPDX-License-Identifier: Apache-2.0
"""Opt-in Humming MoE for an existing signed-INT4 W4AFP8 checkpoint.

Selected by --quantization w4afp8 --moe-runner-backend humming. Dense FP8
methods, attention and KV layout are unchanged. Weight loading uses the same
parameter names and EP mapping as CUTLASS. Only after loading do we repack one
sublayer at a time, without retaining another copy of the expert weights.

Humming uses dynamic per-token FP8 activation scales, whereas the CUTLASS
path uses the checkpoint's static scales. This is a numerical/backend A/B,
not a claim of bit-identical arithmetic or validated model accuracy.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod

logger = logging.getLogger(__name__)
_reported = False


class W4AFp8HummingMoEMethod(W4AFp8MoEMethod):
    def create_moe_runner(self, layer, moe_runner_config):
        from importlib.metadata import version

        from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        from sglang.srt.layers.quantization.humming import _lazy_import_humming

        if version("humming-kernels") != "0.1.12":
            raise ValueError(
                "This W4AFP8 Humming integration requires humming-kernels==0.1.12"
            )
        _lazy_import_humming()

        if not get_moe_a2a_backend().is_none():
            raise ValueError("W4AFP8 Humming currently requires --moe-a2a-backend none")
        if (
            moe_runner_config.activation != "silu"
            or not moe_runner_config.is_gated
            or moe_runner_config.apply_router_weight_on_input
            or moe_runner_config.no_combine
            or moe_runner_config.num_fused_shared_experts
            or getattr(layer, "with_bias", False)
            or any(
                getattr(moe_runner_config, key, None) is not None
                for key in (
                    "gemm1_alpha",
                    "gemm1_beta",
                    "gemm1_clamp_limit",
                    "swiglu_limit",
                )
            )
        ):
            raise ValueError(
                "W4AFP8 Humming requires ordinary gated SiLU with separate shared "
                "experts, output router weights and combine enabled"
            )
        if moe_runner_config.params_dtype != torch.bfloat16:
            raise ValueError("W4AFP8 Humming currently requires BF16 model activations")
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.HUMMING, moe_runner_config)
        # The standard fused wrapper creates a temporary HummingRunnerCore on
        # every invocation. Keep this runner alive for torch.compile custom-op
        # IDs, its tuning cache and CUDA graph replay; the registered standard
        # pre/post-permute functions execute the same kernels.
        self.runner.fused_func = None

    def process_weights_after_loading(self, layer):
        if getattr(self, "_humming_processed", False):
            return
        if not layer.w13_weight.is_cuda or torch.cuda.get_device_capability(
            layer.w13_weight.device
        ) != (9, 0):
            raise ValueError("This W4AFP8 Humming experiment targets SM90 (Hopper)")

        from humming import dtypes
        from humming.layer import HummingMethod
        from humming.schema import HummingInputSchema

        from sglang.srt.layers.quantization.humming import (
            _W4AFp8CheckpointWeightSchema,
        )

        input_schema = HummingInputSchema(
            a_dtype=dtypes.float8e4m3, input_scale_group_size=0
        )
        checkpoint_schema = _W4AFp8CheckpointWeightSchema(self.quant_config.group_size)
        cfg = self.moe_runner_config
        if cfg.num_local_experts != layer.w13_weight.shape[0]:
            raise ValueError("Humming local expert count does not match loaded weights")
        layer.register_buffer(
            "locks",
            torch.zeros(1024, dtype=torch.int32, device=layer.w13_weight.device),
        )

        for name, n, k in (
            ("w13", 2 * cfg.intermediate_size_per_partition, cfg.hidden_size),
            ("w2", cfg.hidden_size, cfg.intermediate_size_per_partition),
        ):
            # Keep the ordinary W4 loader, including its input-scale validation.
            # No CUTLASS post-load scale interleaving has happened at this point.
            schema, tensors = checkpoint_schema.convert_humming(
                tensors={
                    "weight": getattr(layer, name + "_weight"),
                    "weight_scale_inv": getattr(layer, name + "_weight_scale_inv"),
                },
                shape_n_stacks=[n // 2, n // 2] if name == "w13" else [n],
                shape_k_stacks=[k],
                param_dtype=cfg.params_dtype,
                num_experts=cfg.num_local_experts,
            )
            for suffix in ("weight", "weight_scale_inv", "input_scale"):
                delattr(layer, name + "_" + suffix)
            for suffix, tensor in tensors.items():
                layer.register_parameter(
                    name + "_" + suffix, torch.nn.Parameter(tensor, requires_grad=False)
                )
            # Humming mutates/replaces the packed parameters during transform.
            # Drop the conversion dictionary first so it cannot pin old weights.
            del tensors, tensor
            HummingMethod.prepare_layer_meta(
                layer=layer,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema,
                weight_schema=schema,
                has_bias=False,
                num_experts=cfg.num_local_experts,
                torch_dtype=cfg.params_dtype,
                sublayer_name=name,
            )
            HummingMethod.transform_humming_layer(layer, sublayer_name=name)

        self._humming_processed = True
        global _reported
        if not _reported:
            logger.info(
                "W4AFP8 MoE: Humming; signed INT4 weights preserved, dynamic "
                "per-token FP8 activations; local/global experts=%s/%s; "
                "dense FP8 and attention unchanged",
                cfg.num_local_experts,
                cfg.num_experts,
            )
            _reported = True

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.moe_runner.humming import HummingMoeQuantInfo

        return self.runner.run(dispatch_output, HummingMoeQuantInfo(layer=layer))
