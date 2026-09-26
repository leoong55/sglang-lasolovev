"""SM90 DeepGEMM 0.2.0 allocation contract (FP32, 32 padded Q heads)."""


def aligned(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def logits_stride(k_rows, *, compact):
    return aligned(k_rows if compact else k_rows + 256, 256)


def balanced_tiles(q_rows, k_rows, budget_bytes, *, compact):
    """Bound the allocated stride, including up to three padded Q rows."""
    if not q_rows:
        return ()
    row_bytes = logits_stride(k_rows, compact=compact) * 4
    blocks = budget_bytes // (4 * row_bytes)
    if blocks < 1:
        raise ValueError("DSA logits workspace cannot hold four aligned Q rows")
    q_blocks = aligned(q_rows, 4) // 4
    launches = (q_blocks + blocks - 1) // blocks
    base, extra = divmod(q_blocks, launches)
    tiles, start = [], 0
    for i in range(launches):
        end = min(q_rows, start + 4 * (base + (i < extra)))
        tiles.append((start, end))
        start = end
    return tuple(tiles)


def workspace_reservation_bytes(mib, *, is_draft=False):
    # Target and draft use the same transient allocator sequentially.
    return 0 if is_draft or mib is None else int(mib) * (1 << 20)
