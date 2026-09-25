"""Host-side launch geometry and poison-padding checks; no CUDA claim."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from host_source import extract, cp_gather, cp_metadata, InterleaveCPStrategy
from test_host_graph import helper


class EPPaddingTests(unittest.TestCase):
    def test_ep_estimate_is_not_a_route_count_or_buffer_bound(self):
        fn = extract("srt/layers/moe/moe_runner/humming.py",
                     {"estimate_local_valid_shape_m"}, "HummingRunnerCore").estimate_local_valid_shape_m
        core = NS(layer=NS(_humming_standard_ep_aware=True),
                  num_experts=32, global_num_experts=256)
        for batch in (1, 8, 40, 48, 2048, 16384):
            ids = torch.full((batch, 8), -1, dtype=torch.int32)
            # Empty, uniformly distributed and entirely local routes use the
            # same host heuristic. Runtime GPU metadata remains authoritative.
            for fill in (-1, 0, 31):
                ids.fill_(fill)
                self.assertEqual(fn(core, ids), batch)
            self.assertEqual(fn(core, ids, expected_m=7), 224)
        core.layer._humming_standard_ep_aware = False
        self.assertEqual(fn(core, torch.empty(40, 8)), 320)
        core.layer = NS()  # Other Humming integrations keep their dispatch semantics.
        self.assertEqual(fn(core, torch.empty(40, 8)), 320)

    def test_admission_real_lengths_overhead_bound_and_rollback(self):
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_BCG_PADDING": "1"}):
            for lengths, expected in (([1012] * 16, 16384), ([1012] * 7, 8192),
                                      ([8191], 8192), ([8192], 8192),
                                      ([32001], 32768), ([1012], None),
                                      ([8193], None), ([1, 16191], 16384)):
                got = helper.replay_bucket(sum(lengths), lengths, [8192, 16384, 32768])
                self.assertEqual(got, expected)
            self.assertIsNone(helper.replay_bucket(8192, [8191], [8192]))
            self.assertIsNone(helper.replay_bucket(8192, [0, 8192], [8192]))
            self.assertIsNone(helper.replay_bucket(16192, [16192], [8192]))
            self.assertIsNone(helper.replay_bucket(16192, [16192], [16384], cp_size=4))
            self.assertIsNone(helper.replay_bucket(16192, [16192], [16384], max_padding_factor=1))
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_BCG_PADDING": "0"}):
            self.assertIsNone(helper.replay_bucket(16192, [1012]*16, [16384]))
            self.assertEqual(helper.replay_bucket(16384, [16384], [16384]), 16384)

    def test_real_gather_excludes_poison_padding_at_production_shapes(self):
        # Run the actual interleave gather implementation with a simulated
        # collective. This is also the gather used before KV/DSA stores.
        for total, bucket in ((16192, 16384), (7084, 8192), (8191, 8192)):
            logical = [len(range(rank, total, 8)) for rank in range(8)]
            physical = bucket // 8
            expected = torch.arange(total * 2).reshape(total, 2).float()
            chunks = []
            for rank in range(8):
                chunk = torch.zeros(physical, 2)
                chunk[:logical[rank]].copy_(expected[rank::8])
                chunks.append(chunk)
            for rank in range(8):
                strategy = InterleaveCPStrategy()
                strategy.cp_rank, strategy.cp_size = rank, 8
                meta = cp_metadata.InterleaveContextParallelMetadata(
                    total_seq_lens=total, per_rank_actual_token=[physical]*8,
                    per_rank_logical_token=logical)
                local = torch.full((physical, 2), float("nan"))
                local[:logical[rank]].copy_(expected[rank::8])
                parallel = NS(dcp_enabled=True, attn_cp_group=object())
                def gather(out, inp):
                    torch.testing.assert_close(inp, chunks[rank])
                    out.copy_(torch.cat(chunks))
                with patch.object(cp_gather, "get_parallel", return_value=parallel), \
                     patch.object(cp_gather, "attn_cp_all_gather_into_tensor", side_effect=gather):
                    got = strategy._gather_interleaved_tensor(local, NS(attn_cp_metadata=meta))
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                self.assertEqual(meta.total_seq_lens, total)


if __name__ == "__main__":
    unittest.main()
