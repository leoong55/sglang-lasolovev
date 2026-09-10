"""Byte/order/lifetime checks against independent references and the v2 path.

Collectives are CPU mocks. The tests do not validate CUDA/NCCL execution or
claim any H200 speedup. Production functions are extracted from the overlay.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from host_source import (
    InterleaveCPStrategy,
    cp_gather,
    cp_metadata,
    dcp_comm,
    dcp_metadata,
)


WIDTH = 656


def packed_rows(n, salt=0):
    return (torch.arange(n * WIDTH).reshape(n, 1, WIDTH) * 37 + salt).remainder(256).to(torch.uint8)


class TestPreparedDCPGather(unittest.TestCase):
    def test_aligned_prefixes_all_ranks_byte_exact_and_workspace_reused(self):
        # Unequal requests, empty prefixes, random physical pool placement,
        # and two layers whose packed bytes differ in every field.
        for size in (2, 4, 8):
            lens = [0, 256, 512, 0, 256]
            total = sum(lens)
            for rank in range(size):
                with self.subTest(size=size, rank=rank):
                    plan = dcp_metadata.DCPPrefillGatherPlan(total, size, rank)
                    indices = torch.randperm(total // size, generator=torch.Generator().manual_seed(rank)).to(torch.int32) + 5
                    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=size, dcp_rank=rank, dcp_group=SimpleNamespace())
                    first_ptrs = None
                    for layer in (1, 2):
                        expected = packed_rows(total, salt=layer * 17)
                        persistent = packed_rows(total // size + 10, salt=201)
                        persistent[indices.long()] = expected[rank::size]

                        def gather(out, local):
                            torch.testing.assert_close(local, expected[rank::size], rtol=0, atol=0)
                            out.copy_(torch.cat([expected[r::size] for r in range(size)]))

                        parallel.dcp_group.all_gather_into_tensor = MagicMock(side_effect=gather)
                        actual = torch.full_like(expected, 255)
                        with patch.object(dcp_comm, 'get_parallel', return_value=parallel):
                            dcp_comm._gather_packed_prefix_with_plan(persistent, indices, actual, plan)
                            # Compare to the previous generic implementation too.
                            generic = dcp_comm.all_gather_kv_cache_for_dcp(
                                persistent[indices.long()], None, torch.tensor(lens, dtype=torch.int32))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual, generic, rtol=0, atol=0)
                        ptrs = tuple(t.data_ptr() for t in next(iter(plan.workspaces.values())))
                        if first_ptrs is None:
                            first_ptrs = ptrs
                        self.assertEqual(ptrs, first_ptrs)
                        self.assertEqual(len(plan.workspaces), 1)

    def test_packed_extend_uses_plan_updates_layer_and_zeroes_tail(self):
        size, rank, total, current = 4, 2, 256, 9
        plan = dcp_metadata.DCPPrefillGatherPlan(total, size, rank)
        parallel = SimpleNamespace(dcp_size=size, dcp_rank=rank, dcp_group=SimpleNamespace())
        buffer = torch.empty((320, 1, WIDTH), dtype=torch.float8_e4m3fn)
        for layer in (2, 7):
            expected = packed_rows(total, layer * 23)
            pool = SimpleNamespace(dsa_kv_cache_store_fp8=True,
                get_key_buffer=MagicMock(return_value=expected[rank::size].contiguous().view(torch.float8_e4m3fn)))
            parallel.dcp_group.all_gather_into_tensor = lambda out, inp: out.copy_(torch.cat([expected[r::size] for r in range(size)]))
            nope = torch.full((current, 1, 528), layer, dtype=torch.uint8)
            rope = torch.full((current, 1, 128), layer + 1, dtype=torch.uint8)
            buffer.view(torch.uint8).fill_(251)
            with patch.object(dcp_comm, 'get_parallel', return_value=parallel), \
                 patch.object(dcp_comm, 'all_gather_kv_cache_for_dcp', side_effect=AssertionError('generic path used')), \
                 patch('sglang.kernels.ops.attention.dsa.quant_k_cache.quantize_k_cache_separate', return_value=(nope, rope)):
                dcp_comm.all_gather_kv_cache_for_mla_extend(
                    pool, SimpleNamespace(layer_id=layer), [0, 256], torch.arange(total // size, dtype=torch.int32),
                    total, buffer, 512, torch.empty(current, 1, 512), torch.empty(current, 1, 64), plan)
            pool.get_key_buffer.assert_called_once_with(layer)
            actual = buffer.view(torch.uint8)
            torch.testing.assert_close(actual[:total], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[total:total+current], torch.cat((nope, rope), -1), rtol=0, atol=0)
            self.assertEqual(actual[total+current:].count_nonzero().item(), 0)

    def test_zero_prefix_avoids_collective_and_bad_plan_fails_before_collective(self):
        group = SimpleNamespace(all_gather_into_tensor=MagicMock())
        parallel = SimpleNamespace(dcp_size=4, dcp_rank=0, dcp_group=group)
        with patch.object(dcp_comm, 'get_parallel', return_value=parallel):
            plan = dcp_metadata.DCPPrefillGatherPlan(0, 4, 0)
            dcp_comm._gather_packed_prefix_with_plan(packed_rows(4), torch.empty(0, dtype=torch.int32), packed_rows(0), plan)
            self.assertEqual(plan.workspaces, {})
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                dcp_comm._gather_packed_prefix_with_plan(packed_rows(4), torch.tensor([0, 1]), packed_rows(4), dcp_metadata.DCPPrefillGatherPlan(4, 4, 0))
        group.all_gather_into_tensor.assert_not_called()

    def test_workspaces_separate_by_stream_and_batch(self):
        expected = packed_rows(8)
        group = SimpleNamespace(all_gather_into_tensor=lambda out, inp: out.copy_(torch.cat([expected[r::4] for r in range(4)])))
        parallel = SimpleNamespace(dcp_size=4, dcp_rank=0, dcp_group=group)
        plans = [dcp_metadata.DCPPrefillGatherPlan(8, 4, 0) for _ in range(2)]
        # Exercise CUDA stream-key logic with CPU allocations, not a CUDA test.
        with patch.object(dcp_comm, 'get_parallel', return_value=parallel), \
             patch.object(torch.Tensor, 'is_cuda', new=property(lambda self: True)):
            for plan in plans:
                for stream in (11, 22, 11):
                    with patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=stream)):
                        out = torch.empty_like(expected)
                        dcp_comm._gather_packed_prefix_with_plan(expected[::4], torch.tensor([0, 1]), out, plan)
                        torch.testing.assert_close(out, expected, rtol=0, atol=0)
                self.assertEqual(len(plan.workspaces), 2)
        all_ptrs = [t.data_ptr() for p in plans for pair in p.workspaces.values() for t in pair]
        self.assertEqual(len(set(all_ptrs)), len(all_ptrs))


class TestPreparedCPGather(unittest.TestCase):
    def run_case(self, total, physical, shape, dtype, padded_input=False):
        size = 8
        logical = [total // size + (r < total % size) for r in range(size)]
        for rank in range(size):
            with self.subTest(total=total, physical=physical, rank=rank, dtype=dtype):
                strategy = InterleaveCPStrategy()
                strategy.cp_size, strategy.cp_rank = size, rank
                meta = cp_metadata.InterleaveContextParallelMetadata(total_seq_lens=total,
                    per_rank_actual_token=[physical] * size, per_rank_logical_token=logical)
                batch = SimpleNamespace(attn_cp_metadata=meta)
                parallel = SimpleNamespace(dcp_enabled=True, attn_cp_group=object())
                retained = []
                first_ptrs = None
                for layer in (1, 2):
                    # Noncontiguous local input with optional poisoned physical padding.
                    expected = (torch.arange(total * shape * 2).reshape(total, shape * 2) + layer).to(dtype)[:, ::2]
                    chunks = []
                    for r in range(size):
                        padded = torch.zeros((physical, shape), dtype=dtype)
                        padded[:logical[r]].copy_(expected[r::size])
                        chunks.append(padded)
                    local = expected[rank::size]
                    if padded_input:
                        local = torch.full((physical, shape), 77, dtype=dtype)
                        local[:logical[rank]].copy_(expected[rank::size])

                    def gather(out, inp):
                        torch.testing.assert_close(inp, chunks[rank], rtol=0, atol=0)
                        out.copy_(torch.cat(chunks))

                    with patch.object(cp_gather, 'get_parallel', return_value=parallel), \
                         patch.object(cp_gather, 'attn_cp_all_gather_into_tensor', side_effect=gather):
                        actual = strategy._gather_interleaved_tensor(local, batch)
                        parallel.dcp_enabled = False
                        reference = strategy._gather_interleaved_tensor(local, batch)
                        parallel.dcp_enabled = True
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    workspace = next(iter(meta.gather_workspaces.values()))
                    ptrs = tuple(t.data_ptr() for t in workspace)
                    if first_ptrs is None:
                        first_ptrs = ptrs
                    self.assertEqual(ptrs, first_ptrs)
                    self.assertNotIn(actual.data_ptr(), ptrs)
                    retained.append((actual, actual.clone()))
                for actual, snapshot in retained:
                    torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)

    def test_even_and_uneven_order_padding_and_output_lifetime(self):
        for total, physical in ((32, 4), (19, 3), (19, 8), (1, 1)):
            for dtype in (torch.bfloat16, torch.uint8):
                self.run_case(total, physical, shape=7, dtype=dtype, padded_input=False)

    def test_physical_input_padding_is_never_transmitted_as_tokens(self):
        self.run_case(19, 8, shape=3, dtype=torch.bfloat16, padded_input=True)

    def test_shape_stream_and_logical_length_changes_do_not_share_storage(self):
        strategy = InterleaveCPStrategy()
        strategy.cp_size, strategy.cp_rank = 8, 0
        meta = cp_metadata.InterleaveContextParallelMetadata(total_seq_lens=16,
            per_rank_actual_token=[2]*8, per_rank_logical_token=[2]*8)
        parallel = SimpleNamespace(dcp_enabled=True, attn_cp_group=object())
        with patch.object(cp_gather, 'get_parallel', return_value=parallel), \
             patch.object(cp_gather, 'attn_cp_all_gather_into_tensor', side_effect=lambda out, inp: out.copy_(torch.cat([inp]*8))), \
             patch.object(torch.Tensor, 'is_cuda', new=property(lambda self: True)):
            for stream, width, logical in ((11, 576, 2), (11, 128, 2), (22, 576, 2), (11, 576, 1)):
                meta.per_rank_logical_token = [logical]*8
                meta.total_seq_lens = logical*8
                with patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=stream)):
                    result = strategy._gather_interleaved_tensor(torch.full((logical, width), 3), SimpleNamespace(attn_cp_metadata=meta))
                    torch.testing.assert_close(result, torch.full((logical*8, width), 3))
        self.assertEqual(len(meta.gather_workspaces), 4)
        pointers = [t.data_ptr() for pair in meta.gather_workspaces.values() for t in pair]
        self.assertEqual(len(set(pointers)), len(pointers))


if __name__ == '__main__':
    unittest.main()
