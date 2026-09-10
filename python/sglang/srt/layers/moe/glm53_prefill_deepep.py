# SPDX-License-Identifier: Apache-2.0
"""Opt-in CP-local W4AFP8 MoE prefill; the standard decode path is untouched.

This uses DeepEP's legacy normal (BF16) dispatcher and the existing CUTLASS
W4AFP8 adapter. It does not change the process-wide MoE backend or routed
weight layout. A separately loaded TP1 shared expert avoids a hidden-state
all-gather while preserving the original TP-sharded shared expert for decode.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def local_token_count(total, rank, size, physical_rows):
    """Logical rows of a rank-strided CP shard, excluding its tail padding."""
    if total < 0 or size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid interleave CP token layout")
    rows = total // size + int(rank < total % size)
    if rows > physical_rows:
        raise ValueError("CP logical rows exceed the physical hidden-state shard")
    return rows


def duplicate_shared_weights(weights, model):
    """Feed both shared-expert modules through their own original weight loaders.

    Expand checkpoint names before the usual gate/up stacking. Clone the extra
    tensor so asynchronous loaders cannot share mutable checkpoint storage.
    Fail before serving if any auxiliary weight or scale was not supplied.
    """
    marker = ".mlp.prefill_shared_experts."
    expected = set()
    for name, _ in model.named_parameters():
        if marker not in name:
            continue
        if ".gate_up_proj." in name:
            expected.add(name.replace(".gate_up_proj.", ".gate_proj."))
            expected.add(name.replace(".gate_up_proj.", ".up_proj."))
        else:
            expected.add(name)
    missing = expected.copy()
    for name, tensor in weights:
        auxiliary = name.replace(".mlp.shared_experts.", marker, 1)
        # Yield the auxiliary copy first, before any asynchronous original load.
        if auxiliary in expected:
            yield auxiliary, tensor.clone()
            missing.discard(auxiliary)
        yield name, tensor
    if missing:
        raise RuntimeError(
            "GLM53 prefill shared-expert checkpoint entries missing: "
            + ", ".join(sorted(missing))
        )


class Glm53PrefillDeepEP:
    """One dispatcher per MoE layer, with one process-wide transport buffer."""

    def __init__(self, moe, config):
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPBuffer,
            DeepEPDispatcher,
            DeepEPNormalCombineInput,
        )
        from sglang.srt.layers.moe.utils import DeepEPMode, get_moe_a2a_backend
        from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )
        from sglang.srt.runtime_context import get_exec, get_parallel

        parallel = get_parallel()
        if (
            config.architectures != ["GlmMoeDsaForCausalLM"]
            or config.num_hidden_layers != 78
            or config.hidden_size != 6144
            or config.n_routed_experts != 256
            or config.num_experts_per_tok != 8
            or config.n_shared_experts != 1
            or config.hidden_act != "silu"
            or not isinstance(moe.experts.quant_method, W4AFp8MoEMethod)
            or moe.is_nextn
            or moe.is_hash
            or moe.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "Prefill DeepEP v4 requires full GLM-5.3 W4AFP8, unfused shared expert"
            )
        if (
            parallel.tp_size != 8
            or parallel.attn_cp_size != 8
            or parallel.moe_ep_size != 8
            or parallel.moe_tp_size != 1
            or parallel.cp_strategy != "interleave"
            or not get_moe_a2a_backend().is_none()
            or get_exec().moe.enable_eplb
            or get_exec().moe.ep_num_redundant_experts != 0
            or moe.experts.should_fuse_routed_scaling_factor_in_topk
        ):
            raise ValueError(
                "Prefill DeepEP v4 requires CP8/EP8, static expert placement, backend none"
            )
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise ValueError(
                "The existing DeepEP normal combine requires DeepGEMM enabled"
            )
        if not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
            raise ValueError(
                "Prefill DeepEP v4 requires eager prefill; decode graphs remain supported"
            )
        group = get_tp_group()
        if (
            group.ranks != parallel.attn_cp_group.ranks
            or group.rank_in_group != moe.experts.moe_ep_rank
        ):
            raise ValueError("CP, EP and DeepEP rank ordering must agree")
        self.cp_rank = parallel.attn_cp_rank
        self.cp_size = parallel.attn_cp_size
        self.combine_input_type = DeepEPNormalCombineInput
        self.dispatcher = DeepEPDispatcher(
            group=group.device_group,
            router_topk=config.num_experts_per_tok,
            num_experts=config.n_routed_experts,
            num_local_experts=moe.experts.num_local_experts,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            deepep_mode=DeepEPMode.NORMAL,
            permute_fusion=True,
            async_finish=True,
        )
        # Keep checkpoint static activation scales: no FP8 transport requantize.
        self.dispatcher.set_quant_config({"normal_dispatcher_output_dtype": "bf16"})
        # Allocate during model construction, before the KV memory budget is
        # profiled. Never allocate the communication arena on the first request.
        buffer = DeepEPBuffer.get_deepep_buffer(
            group.device_group, config.hidden_size, 2, DeepEPMode.NORMAL
        )
        if moe.layer_id == config.first_k_dense_replace:
            logger.info(
                "GLM53 v4: DeepEP normal/BF16 prefill buffer initialized before KV sizing; "
                "NVLink bytes=%s RDMA bytes=%s; CP8/EP8, decode backend remains none",
                getattr(buffer, "num_nvl_bytes", "unknown"),
                getattr(buffer, "num_rdma_bytes", "unknown"),
            )
        self.log_first_forward = moe.layer_id == config.first_k_dense_replace

    def forward(self, moe, hidden_states, forward_batch):
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("Prefill DeepEP requires BF16 hidden states")
        total = forward_batch.attn_cp_metadata.total_seq_lens
        rows = local_token_count(total, self.cp_rank, self.cp_size, len(hidden_states))
        local_hidden = hidden_states[:rows].contiguous()
        if rows:
            logits = moe.gate(local_hidden, forward_batch=forward_batch)
            topk = moe.topk(local_hidden, logits)
        else:
            # Empty ranks must still enter the same dispatch/combine sequence.
            topk = moe.topk.empty_topk_output(
                hidden_states.device, layer_id=moe.layer_id
            )
        dispatched = self.dispatcher.dispatch(local_hidden, topk)
        output = moe.experts.quant_method.apply_deepep_normal(moe.experts, dispatched)
        combined = self.dispatcher.combine(
            self.combine_input_type(
                output, dispatched.topk_ids, dispatched.topk_weights
            )
        )
        # The existing normal adapter applies router weights but no routed
        # scaling factor. Scale exactly once, before adding the shared expert.
        combined = combined * moe.routed_scaling_factor
        if rows:
            combined = combined + moe.prefill_shared_experts(local_hidden)
        if combined.shape != local_hidden.shape:
            raise RuntimeError(
                "DeepEP combine did not restore the CP-local token shape"
            )
        if rows != len(hidden_states):
            padded = torch.zeros_like(hidden_states)
            padded[:rows].copy_(combined)
            combined = padded
        if self.log_first_forward:
            logger.info(
                "GLM53 v4: prefill DeepEP active on CP rank %d: %d/%d local rows, "
                "%d global rows; shared expert TP1; routed weights reused",
                self.cp_rank,
                rows,
                len(hidden_states),
                total,
            )
            self.log_first_forward = False
        return combined
