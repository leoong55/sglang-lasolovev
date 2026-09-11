"""Narrow GLM-5.3 DFlash/CP/DCP integration contract."""
from __future__ import annotations

import os


def supports_dflash_dcp(server_args):
    from sglang.srt.arg_groups.overrides import (
        attention_backends_of, model_config_of, resolved_view, resolving_view,
    )
    cfg = resolving_view(server_args)
    resolved = resolved_view(server_args)
    model = model_config_of(server_args).hf_config
    return (
        os.environ.get("SGLANG_GLM53_DFLASH_DCP") == "1"
        and cfg.speculative_algorithm == "DFLASH"
        and cfg.tp_size == 8 and cfg.ep_size == 8 and cfg.dp_size == 1 and cfg.pp_size == 1
        and cfg.enable_prefill_cp and cfg.cp_strategy == "interleave"
        and cfg.enable_cp_decode_attn_tp and cfg.dcp_size == 4
        and cfg.dcp_comm_backend == "ag_rs"
        and cfg.kv_cache_dtype == "fp8_e4m3" and cfg.page_size == 64
        and cfg.quantization == "w4afp8"
        and resolved.attn_cp_size == 8 and resolved.moe_a2a_backend == "none"
        # Generic dispatch selects DSA; its kernels have separate selectors.
        and attention_backends_of(resolved) == ("dsa", "dsa")
        and resolved.dsa_prefill_backend == "flashmla_sparse_q8"
        and resolved.dsa_decode_backend == "flashmla_kv"
        and getattr(model, "num_hidden_layers", None) == 78
        and "GlmMoeDsaForCausalLM" in getattr(model, "architectures", [])
    )


def validate_verify_layout(batch, width):
    """Shape-only checks: no GPU-to-host synchronization during graph capture."""
    info = batch.spec_info
    if info is None or getattr(info, "draft_token_num", None) != width:
        raise ValueError("GLM53 DFlash requires a fixed-width verify block")
    if getattr(info, "custom_mask", None) is not None:
        raise ValueError("GLM53 DFlash DCP supports linear causal verification only")
    if getattr(info, "ragged_verify_layout", None) is not None:
        raise ValueError("GLM53 DFlash DCP does not support ragged verify")
    if width is None or width < 2 or len(batch.input_ids) != batch.batch_size * width:
        raise ValueError("GLM53 DFlash verify row count must equal batch_size * block_size")


def validate_flashmla_rows(query_rows, index_rows, cache_rows, split_rows):
    # TARGET_VERIFY is flattened into B*W independent single-query rows.
    # Every row has its own causal KV limit; scheduler splits must match.
    if index_rows != query_rows or cache_rows != query_rows or split_rows != query_rows + 1:
        raise ValueError(
            "GLM53 DFlash DCP FlashMLA metadata/query mismatch: "
            f"q={query_rows}, indices={index_rows}, lengths={cache_rows}, splits={split_rows}"
        )
