"""Real CP split kernel + both host callers vs independent token ownership.

Run with TRITON_INTERPRET=1 on CPU, or without it on CUDA.
Compilation is checked separately by kernel_preflight.py (no interpreter).
"""

import importlib.util
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from host_source import ROOT, extract, stub

path = ROOT / "python/sglang/kernels/ops/attention/dsa/cp_split.py"
spec = importlib.util.spec_from_file_location("actual_cp_split", path)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)
stub("sglang.kernels.ops.attention.dsa.cp_split").dsa_cp_round_robin_split_q_seqs_kernel = kernel.dsa_cp_round_robin_split_q_seqs_kernel
strategy = extract("srt/layers/cp/interleave.py", {"shard_per_request"}, class_name="InterleaveCPStrategy")
legacy = extract(
    "srt/layers/attention/dsa/utils.py",
    {"dsa_cp_round_robin_split_q_seqs", "dsa_cp_round_robin_split_q_seqs_cpu"},
    namespace={"dsa_cp_round_robin_split_q_seqs_kernel": kernel.dsa_cp_round_robin_split_q_seqs_kernel},
)


def reference(lengths, cp, rank):
    # Count congruent positions in each absolute interval; no signed carry.
    result, indices, start = [], [], 0
    for i, length in enumerate(lengths):
        first = start + (rank - start) % cp
        count = max(0, (start + length - first + cp - 1) // cp)
        if count:
            result.append(count)
            indices.append(i)
        start += length
    return result, indices


class CPSplitTest(unittest.TestCase):
    def check_lengths(self, lengths, dtype, rank):
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        values = torch.tensor(lengths, dtype=dtype, device=device)
        expected, indices = reference(lengths, 8, rank)
        cp = SimpleNamespace(cp_size=8, cp_rank=rank)
        parallel = SimpleNamespace(attn_cp_size=8, attn_cp_rank=rank)
        with patch.object(legacy, "get_parallel", lambda: parallel, create=True):
            outputs = [strategy.shard_per_request(cp, lengths, values), legacy.dsa_cp_round_robin_split_q_seqs(lengths, values)]
        for cpu_lens, gpu_lens, cpu_ids, gpu_ids in outputs:
            self.assertEqual(cpu_lens, expected)
            self.assertEqual(cpu_ids, indices)
            self.assertEqual(gpu_lens.cpu().tolist(), expected)
            self.assertEqual(gpu_ids.cpu().tolist(), indices)
            self.assertEqual(gpu_lens.dtype, dtype)

    def test_capture_buckets_and_mixed_short_requests(self):
        rng = random.Random(53)
        cases = [[], [0], [1], [1] * 32, [0, 1, 0, 7, 8, 9, 63, 64, 65]]
        for bucket in (8192, 16384, 32768):
            cases += [[bucket], [1, 7, 9, bucket - 17], [bucket // 2, bucket // 2]]
        cases += [[rng.randrange(66) for _ in range(32)] for _ in range(4)]
        for dtype in (torch.int32, torch.int64):
            for rank in range(8):
                for lengths in cases:
                    with self.subTest(dtype=dtype, rank=rank, lengths=lengths):
                        self.check_lengths(lengths, dtype, rank)

    def test_int64_lengths_are_not_narrowed(self):
        # Allocates three lengths, not billions of token rows.
        for rank in range(8):
            self.check_lengths([2**31 + 3, 1, 2**31 + 5], torch.int64, rank)


if __name__ == "__main__":
    unittest.main()
