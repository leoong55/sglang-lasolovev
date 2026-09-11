"""Round-robin CP q-sequence split kernel for DSA prefill.

Migrated from ``sglang.srt.layers.attention.dsa.utils`` (RFC #29630, Phase 2.5).
"""

import triton
import triton.language as tl


@triton.jit
def dsa_cp_round_robin_split_q_seqs_kernel(
    in_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    # Capture metadata uses int64 lengths; eager batches may use int32.
    # A Python zero starts as int32 and cannot widen across a Triton loop.
    # Keep the carry and its operands int64 without narrowing caller lengths.
    extra_seq = tl.full((), 0, tl.int64)
    bs_idx = 0
    for bs in range(tokens):
        cur_len = tl.load(in_seqs_ptr + bs).to(tl.int64)
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        extra_seq = cur_len - cur_seq * cp_size
