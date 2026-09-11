"""Opt-in CP8/EP8 DeepEP token ownership for full GLM-5.3 decode.

Attention consumes a replicated decode batch. DeepEP consumes unique source
tokens. Keep that conversion local to MoE, including the shared expert.
"""

from __future__ import annotations

import os

import torch


def enabled() -> bool:
    return os.environ.get("SGLANG_GLM53_DEEPEP_OPT") == "1"


def partition_decode_tokens(hidden_states, num_token_non_padded, cp_rank, cp_size):
    """Equal-sized contiguous shards, with GPU-resident validity for graph replay.

    Even batch < cp_size gives each rank a nonempty padded dispatch buffer.
    Never branch on the value of the device scalar: a graph captured at 32 must
    also mask a replay with 31 (or fewer) actual requests correctly.
    """
    batch, hidden = hidden_states.shape
    rows = max(1, (batch + cp_size - 1) // cp_size)
    start = cp_rank * rows
    end = min(start + rows, batch)
    if end - start == rows:
        local = hidden_states[start:end].contiguous()
    else:
        local = hidden_states.new_zeros((rows, hidden))
        if end > start:
            local[: end - start].copy_(hidden_states[start:end])
    if num_token_non_padded is None:
        local_count = torch.tensor(
            [max(0, min(rows, batch - start))],
            dtype=torch.int32,
            device=hidden_states.device,
        )
    else:
        local_count = (num_token_non_padded - start).clamp(0, rows)
    return local, local_count


def gather_decode_tokens(local_output, batch, cp_size, all_gather_into_tensor):
    output = local_output.new_empty(
        (local_output.shape[0] * cp_size, local_output.shape[1])
    )
    all_gather_into_tensor(output, local_output.contiguous())
    return output[:batch]


def forward_partitioned_decode(moe, hidden_states, forward_batch):
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    batch = hidden_states.shape[0]
    original_count = forward_batch.num_token_non_padded
    local, count = partition_decode_tokens(
        hidden_states, original_count, parallel.attn_cp_rank, parallel.attn_cp_size
    )
    # This Python scope runs during eager/capture. Captured kernels reference
    # the derived GPU scalar, which is recomputed from the live graph counter.
    forward_batch.num_token_non_padded = count
    try:
        local_output = moe._forward_deepep_impl(local, forward_batch)
    finally:
        forward_batch.num_token_non_padded = original_count
    return gather_decode_tokens(
        local_output,
        batch,
        parallel.attn_cp_size,
        # This profile has TP == CP == EP (same rank order). Use the TP
        # communicator, which is explicitly enabled during decode capture.
        get_tp_group().all_gather_into_tensor,
    )
