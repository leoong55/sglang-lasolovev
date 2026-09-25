# SPDX-License-Identifier: Apache-2.0
"""Opt-in attention-output AllReduce + residual RMSNorm for CP8 decode.

CP's attention-TP group has size one. Its decode output projection instead
reduces over global TP8. The EP8 workspace is usable only when its ordered
peers are exactly those TP8 peers; using the attention workspace is wrong.
Workspace allocation remains owned by BaseRunner, before graph capture.
"""

import logging
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("SGLANG_GLM53_CP_DECODE_FUSION", "0") == "1"
MAX_TOKENS = 2048
_reported = set()


def eligible(layer, batch, hidden_states, residual):
    if not ENABLED or not batch.forward_mode.is_decode():
        return False
    if not layer.dsa_enable_prefill_cp:
        return False
    from sglang.srt.distributed import get_moe_ep_group, get_tp_group
    from sglang.srt.layers.flashinfer_comm_fusion import can_use_flashinfer_allreduce
    from sglang.srt.runtime_context import get_exec, get_parallel, get_platform

    p = get_parallel()
    if (
        not get_platform().is_sm90
        or get_exec().comm.flashinfer_allreduce_fusion_backend != "trtllm"
        or (p.tp_size, p.moe_ep_size, p.attn_cp_size, p.attn_tp_size,
            p.attn_dp_size, p.pp_size, p.nnodes, p.attn_dcp_size)
        != (8, 8, 8, 1, 1, 1, 1, 4)
    ):
        return False
    projection = layer.self_attn.o_proj
    if (
        not getattr(projection, "use_decode_attn_tp", False)
        or projection.tp_size != 1
        or projection.reduce_results
        or residual is None
        or hidden_states.ndim != 2
        or not 0 < hidden_states.shape[0] <= MAX_TOKENS
        or hidden_states.shape[1] != 6144
        or hidden_states.dtype != torch.bfloat16
        or residual.dtype != hidden_states.dtype
        or residual.shape != hidden_states.shape
        or not residual.is_contiguous()
    ):
        return False
    ep, tp = get_moe_ep_group(), get_tp_group()
    if ep.ranks != tp.ranks:
        return False
    # This only checks an already initialized workspace. Never allocate or
    # rendezvous here: this function also runs during CUDA graph capture.
    ready = can_use_flashinfer_allreduce(
        hidden_states,
        use_attn_tp_group=False,
        expected_world_size=tp.world_size,
        expected_group_key=(ep.device_group, ep.cpu_group),
    )
    if ready not in _reported:
        _reported.add(ready)
        logger.info("GLM53 CP decode attention fusion: %s (EP8 workspace, TP8 peers)",
                    "eligible" if ready else "workspace unavailable; ordinary path")
    return ready


@contextmanager
def defer_output_reduce(projection, enabled):
    """Keep TP-sliced weights; defer only RowParallelLinear's output reduce.

The outer CP decode context owns weight slicing. This flag is used only by
RowParallelLinear's final all-reduce condition. Restore it even on exceptions.
"""
    if not enabled:
        yield
        return
    original = projection.use_decode_attn_tp
    assert original and projection.tp_size == 1 and not projection.reduce_results
    projection.use_decode_attn_tp = False
    try:
        yield
    finally:
        projection.use_decode_attn_tp = original


def finish(hidden_states, residual, norm):
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.flashinfer_comm_fusion import (
        flashinfer_allreduce_residual_rmsnorm,
    )

    result = flashinfer_allreduce_residual_rmsnorm(
        input_tensor=hidden_states,
        residual=residual,
        weight=norm.weight,
        eps=norm.variance_epsilon,
        max_token_num=MAX_TOKENS,
        use_attn_tp_group=False,
    )
    if result[0] is not None:
        return result
    # The generic RMSNorm helper does not perform an NVIDIA all-reduce when
    # fusion returns None. We deferred that reduction, so must restore it.
    # Never catch a CUDA/collective exception and retry a different collective.
    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    return norm(hidden_states, residual)
