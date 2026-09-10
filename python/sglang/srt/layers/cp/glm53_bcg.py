"""Opt-in exact-token BCG for full GLM-5.3 interleave CP8/DCP4.

Attention is an eager break with live request state; only fixed-row transformer
segments are captured. This module does not alter decode execution.
"""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("SGLANG_GLM53_PREFILL_BCG") == "1"


def supports(server_args) -> bool:
    if not enabled():
        return False
    from sglang.srt.arg_groups.overrides import (
        attention_backends_of,
        model_config_of,
        resolved_view,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    resolved = resolved_view(server_args)
    model = model_config_of(server_args).hf_config
    return (
        cfg.enable_prefill_cp
        and cfg.cp_strategy == "interleave"
        and cfg.tp_size == 8
        and cfg.ep_size == 8
        and cfg.dp_size == 1
        and cfg.pp_size == 1
        and cfg.dcp_size == 4
        and cfg.kv_cache_dtype == "fp8_e4m3"
        and cfg.page_size == 64
        and cfg.quantization == "w4afp8"
        and cfg.dcp_comm_backend == "ag_rs"
        and resolved.attn_cp_size == 8
        and resolved.moe_a2a_backend == "none"
        and attention_backends_of(resolved)[0] == "flashmla_sparse_q8"
        and getattr(model, "num_hidden_layers", None) == 78
        and "GlmMoeDsaForCausalLM" in getattr(model, "architectures", [])
    )


def exact_local_rows(extend_seq_lens, cp_size=8):
    # Interleave partitions the concatenated batch, not each request separately.
    # No padded reuse in v5: this also keeps MoE/hidden collectives invariant.
    if extend_seq_lens is None or cp_size != 8:
        return None
    if any(int(length) <= 0 for length in extend_seq_lens):
        return None
    total = sum(int(length) for length in extend_seq_lens)
    return 1024 if total == 8192 else None


def prepare_dcp(runner, batch):
    from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
        create_chunked_prefix_cache_kv_indices,
    )
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    model_runner = runner.model_runner
    # A fresh plan is necessary even when total query rows stay at 8192:
    # prefix lengths, request slots and physical KV pages change per replay.
    with runner._prefill_forward_context(batch):
        batch.attn_dcp_metadata = (
            model_runner.model.prepare_context_parallel_metadata_for_dcp(
                batch.seq_lens,
                batch.extend_prefix_lens,
                batch.extend_prefix_lens_cpu,
                batch.extend_seq_lens,
                batch.req_pool_indices,
                get_req_to_token_pool().req_to_token,
                batch.seq_lens_sum,
                get_token_to_kv_pool().get_kv_buffer_shape()[0],
                model_runner.kv_cache_dtype,
                model_runner.device,
                create_chunked_prefix_cache_kv_indices,
            )
        )
    if batch.attn_dcp_metadata is None:
        raise RuntimeError("GLM53 BCG requires a live DCP metadata plan")


def attention_break(
    attention, hidden_states, layer_scatter_modes, llama_4_scaling, prev_topk_indices
):
    # Lazily import runner modules: argument resolution imports this module too.
    return _get_attention_break()(
        attention,
        hidden_states,
        layer_scatter_modes,
        llama_4_scaling,
        prev_topk_indices,
    )


_attention_break_fn = None


def _get_attention_break():
    global _attention_break_fn
    if _attention_break_fn is None:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )

        _attention_break_fn = eager_on_graph(True)(_attention_eager)
    return _attention_break_fn


def _attention_eager(
    attention, hidden_states, layer_scatter_modes, llama_4_scaling, prev_topk_indices
):
    import torch
    from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        eager_execution,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )
    from sglang.srt.utils import BumpAllocator

    context = get_tc_piecewise_forward_context()
    if context is None:
        raise RuntimeError("Missing live prefill forward context at attention break")
    batch = context.forward_batch
    if batch.attn_cp_metadata is None or batch.attn_dcp_metadata is None:
        raise RuntimeError("Missing live CP/DCP metadata at attention break")
    # Python layer-communicator calls do not execute during graph replay.
    # Rebuild the lazy QKV context, never reuse the capture batch's latent cache.
    attn_context = get_attn_tp_context()
    with eager_execution():
        attn_context.set_attn_inputs(
            AttentionInputs(hidden_states, batch, attention.prepare_qkv_latent)
        )
        try:
            # Two per-tensor FP8 scale scratch values at most in MLA absorb.
            # A fresh allocator prevents a captured Python bump pointer from
            # advancing beyond its buffer on subsequent replays.
            allocator = BumpAllocator(2, torch.float32, batch.positions.device)
            return attention._forward_impl(
                batch.positions,
                hidden_states,
                batch,
                allocator,
                layer_scatter_modes,
                llama_4_scaling,
                prev_topk_indices,
            )
        finally:
            attn_context.clear_attn_inputs()
