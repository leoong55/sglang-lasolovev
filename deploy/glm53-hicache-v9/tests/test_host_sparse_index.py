"""Exercise real HiCache packing/copy methods with CPU byte-transfer substitutes.

These tests verify logical layer ordering and DCP page relocation. They do not
validate CUDA DMA or establish a GPU performance improvement.
"""

import logging
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from test_host_hicache import direct_copy, extract


def host_class(copy_fn=direct_copy):
    names = {
        "_is_device_layer_sharded", "_device_owned_layer_range",
        "_effective_host_layer_num", "_is_device_layer_owned",
        "_host_layer_index", "_owned_device_layer_ids",
        "logical_size", "logical_page_size",
    }
    methods = extract("mem_cache/pool_host/base.py", names, {}, "HostKVCache")
    base = type("HostBase", (), {
        **vars(methods), "clear": lambda self: None, "dcp_size": 1,
    })
    namespace = dict(
        torch=torch, threading=threading, HostKVCache=base,
        logger=logging.getLogger(__name__),
        _is_cuda=False, _is_hip=False, _is_npu=False, _is_xpu=False, _is_mps=False,
        get_allocator_from_storage=lambda _: None,
        host_memory_budget_bytes=lambda: 10**9,
        DSATokenToKVPool=NS(index_k_with_scale_buffer_dtype=torch.uint8),
        ALLOC_MEMORY_FUNCS={
            "cpu": lambda shape, **kwargs: torch.empty(shape, dtype=kwargs["dtype"]),
        },
        transfer_kv_direct=copy_fn,
    )
    return extract(
        "mem_cache/pool_host/dsa.py", {"DSAIndexerPoolHost"}, namespace,
    ).DSAIndexerPoolHost


def device_pool(skip_layers, *, start_layer=0, rank=0):
    stride = 64 * 132
    pool = NS(
        page_size=64, store_dtype=torch.uint8, start_layer=start_layer,
        end_layer=start_layer + len(skip_layers), layer_num=len(skip_layers),
        index_head_dim=128, quant_block_size=128, device="cpu",
        layer_shard_enabled=False, skip_topk_layers=skip_layers,
        dcp_size=4, dcp_rank=rank,
    )
    pool.index_k_with_scale_buffer = [
        torch.randint(0, 256, (0 if skip else 40, stride), dtype=torch.uint8)
        for skip in skip_layers
    ]
    return pool


def make_host(pool, drafts=(), copy_fn=direct_copy):
    anchor = NS(logical_size=1024, mtp_draft_device_pools=drafts)
    return host_class(copy_fn)(pool, anchor, "layer_first", pin_memory=False)


class SparseIndexerTransferTests(unittest.TestCase):
    def test_relocated_round_trip_keeps_all_dcp_owner_pages(self):
        # Include skipped first, middle and last logical layers, and a PP-local
        # offset. Index pages stay replicated for every DCP owner.
        mask = [True, False, False, True, False, True]
        old = torch.cat([torch.arange(256, 512), torch.arange(1024, 1280)])
        new = torch.cat([torch.arange(1536, 1792), torch.arange(512, 768)])
        host_ids = torch.cat([torch.arange(0, 256), torch.arange(768, 1024)])
        for rank in range(4):
            with self.subTest(rank=rank):
                calls = []

                def copy(**kwargs):
                    self.assertTrue(all(x.shape[0] > 0 for x in kwargs["src_layers"]))
                    self.assertTrue(all(x.shape[0] > 0 for x in kwargs["dst_layers"]))
                    calls.append((len(kwargs["src_layers"]), len(kwargs["src_indices"])))
                    direct_copy(**kwargs)

                pool = device_pool(mask, start_layer=39, rank=rank)
                host = make_host(pool, copy_fn=copy)
                self.assertEqual(host.target_host_layer_mapping, {1: 0, 2: 1, 4: 2})
                self.assertEqual(host.layer_num, 3)
                expected = {
                    layer: pool.index_k_with_scale_buffer[layer][old[::64] // 64].clone()
                    for layer in (1, 2, 4)
                }
                host.backup_from_device_all_layer(pool, host_ids, old, "direct")
                self.assertEqual(calls, [(3, 8)])  # Every physical page, on every rank.
                for buffer in pool.index_k_with_scale_buffer:
                    buffer.zero_()
                for layer in range(len(mask)):
                    host.load_to_device_per_layer(pool, host_ids, new, layer, "direct")
                self.assertEqual(calls, [(3, 8), (1, 8), (1, 8), (1, 8)])
                for layer, wanted in expected.items():
                    torch.testing.assert_close(
                        pool.index_k_with_scale_buffer[layer][new[::64] // 64], wanted,
                    )
                for layer in (0, 3, 5):
                    self.assertEqual(pool.index_k_with_scale_buffer[layer].numel(), 0)

    def test_78_logical_layers_allocate_21_host_and_device_index_buffers(self):
        producers = {0, 1, 2, *range(6, 78, 4)}
        pool = device_pool([layer not in producers for layer in range(78)])
        host = make_host(pool)
        self.assertEqual(len(producers), 21)
        self.assertEqual(host.layer_num, 21)
        self.assertEqual(host.size_per_token, 21 * 132)
        self.assertEqual(host.get_size_per_token(), 21 * 132)
        self.assertEqual(len(host.index_k_device_ptrs), 21)
        self.assertEqual(len(host.index_k_data_refs), 21)
        self.assertEqual(host.index_k_with_scale_buffer.nbytes, 21 * 17 * 64 * 132)
        self.assertEqual(sum(x.nbytes for x in pool.index_k_with_scale_buffer), 21 * 40 * 64 * 132)
        self.assertEqual(set(host.target_host_layer_mapping), producers)

    def test_packed_mtp_tail_uses_logical_target_offset(self):
        pool = device_pool([False, True, False, True, True])
        draft = device_pool([False])
        host = make_host(pool, drafts=(draft,))
        self.assertEqual(host.layer_num, 3)
        old, new, host_ids = torch.arange(256, 512), torch.arange(1024, 1280), torch.arange(768, 1024)
        expected = draft.index_k_with_scale_buffer[0][4:8].clone()
        host.backup_from_device_all_layer(pool, host_ids, old, "direct")
        draft.index_k_with_scale_buffer[0].zero_()
        # Controller schedules this draft at depth 0 but passes logical ID 5,
        # after all five target layers. The physical packed host slot is 2.
        host.load_to_device_per_layer(draft, host_ids, new, 5, "direct", is_draft=True)
        torch.testing.assert_close(draft.index_k_with_scale_buffer[0][16:20], expected)
        with self.assertRaisesRegex(ValueError, "Invalid packed MTP"):
            host.load_to_device_per_layer(draft, host_ids, new, 2, "direct", is_draft=True)

    def test_zero_producer_pool_does_not_submit_empty_dma(self):
        pool = device_pool([True, True])
        host = make_host(pool, copy_fn=lambda **_: self.fail("empty DMA submitted"))
        ids = torch.arange(256)
        host.backup_from_device_all_layer(pool, ids, ids, "direct")
        for layer in range(2):
            host.load_to_device_per_layer(pool, ids, ids, layer, "direct")
        self.assertEqual(host.index_k_with_scale_buffer.nbytes, 0)

    def test_completion_events_remain_logical_and_follow_real_transfers(self):
        events = []
        pool = device_pool([True, False, True, False, True])
        host = make_host(pool, copy_fn=lambda **_: events.append("index-copy"))

        class Event:
            def record(self):
                pass

            def wait(self, *_):
                pass

        engine_class = extract("mem_cache/l2_transfer.py", {"L2TransferEngine"}, dict(
            device_module=NS(Event=Event, stream=lambda _: nullcontext()),
            make_timing_event_pair=lambda: (Event(), Event(), False),
            TransferCompletion=lambda *args: args,
        )).L2TransferEngine
        engine = object.__new__(engine_class)
        engine.host_to_device_stream = None
        engine.io_backend = "direct"
        ids = torch.arange(256)
        main = NS(load_to_device_per_layer=lambda *args, **_: events.append(f"main-{args[3]}"))
        transfers = [
            NS(host_pool=owner, device_pool=pool, host_indices=ids, device_indices=ids,
               layer_mapper=lambda layer: layer, is_draft=False)
            for owner in (main, host)
        ]
        engine.submit_host_to_device(
            transfers, layer_num=5, on_layer_done=lambda layer: events.append(f"done-{layer}"),
        )
        self.assertEqual(events, [
            "main-0", "done-0", "main-1", "index-copy", "done-1",
            "main-2", "done-2", "main-3", "index-copy", "done-3", "main-4", "done-4",
        ])


class SparseIndexerGuardTests(unittest.TestCase):
    def setUp(self):
        self.memory = NS(
            enable_hisparse=False, enable_hierarchical_cache=True,
            hicache_io_backend="direct", hicache_mem_layout="layer_first",
            hicache_write_policy="write_through", hicache_host_memory_mode="cache",
            hicache_storage_backend=None,
        )
        self.parallel = NS(attn_cp_size=8, attn_dcp_size=4, enable_dsa_cache_layer_split=False)
        self.disagg = NS(disaggregation_mode="null")
        self.enabled = True
        namespace = dict(
            get_memory=lambda: self.memory, get_parallel=lambda: self.parallel,
            get_disagg=lambda: self.disagg,
            envs=NS(SGLANG_GLM53_HICACHE_INDEX_ELISION=NS(get=lambda: self.enabled)),
        )
        self.should_elide = extract(
            "mem_cache/kv_cache_configurator.py", {"_should_elide_dsa_index_k"}, namespace,
        )._should_elide_dsa_index_k

    def test_opt_in_and_excluded_paths(self):
        self.assertTrue(self.should_elide(is_draft_worker=False))
        self.assertFalse(self.should_elide(is_draft_worker=True))
        self.enabled = False
        self.assertFalse(self.should_elide(is_draft_worker=False))
        self.enabled = True
        for owner, name, value in [
            (self.memory, "enable_hisparse", True),
            (self.memory, "hicache_io_backend", "kernel"),
            (self.memory, "hicache_mem_layout", "page_first"),
            (self.memory, "hicache_write_policy", "write_back"),
            (self.memory, "hicache_host_memory_mode", "swap"),
            (self.memory, "hicache_storage_backend", "file"),
            (self.parallel, "attn_cp_size", 4),
            (self.parallel, "attn_dcp_size", 8),
            (self.parallel, "enable_dsa_cache_layer_split", True),
            (self.disagg, "disaggregation_mode", "prefill"),
        ]:
            with self.subTest(name=name, value=value):
                original = getattr(owner, name)
                setattr(owner, name, value)
                self.assertFalse(self.should_elide(is_draft_worker=False))
                setattr(owner, name, original)

    def test_existing_non_hicache_elision_does_not_require_opt_in(self):
        self.memory.enable_hierarchical_cache = False
        self.enabled = False
        self.assertTrue(self.should_elide(is_draft_worker=False))
        self.assertFalse(self.should_elide(is_draft_worker=True))

    def test_solver_factory_and_host_use_the_same_producer_count(self):
        producers = {0, 1, 2, *range(6, 78, 4)}
        skips_topk = lambda config, layer: layer not in producers

        class CPUPool:
            quant_block_size = 128
            index_k_with_scale_buffer_dtype = torch.uint8

            def __new__(cls, size, *, skip_topk_layers=None, layer_num, **kwargs):
                # Only allocation is substituted. The real factory must pass
                # its skip mask, and the real solver must budget the same mask.
                return device_pool(skip_topk_layers or [False] * layer_num)

        namespace = dict(
            torch=torch, get_memory=lambda: self.memory, get_parallel=lambda: self.parallel,
            get_schedule=lambda: NS(page_size=64),
            get_exec=lambda: NS(features=NS(enable_memory_saver=False)),
            DSATokenToKVPool=CPUPool, dsa_layer_skips_topk=skips_topk,
            get_dsa_index_head_dim=lambda config: 128,
            calculate_mla_kv_cache_dim=lambda **kwargs: 656,
            _should_elide_dsa_index_k=self.should_elide,
        )
        build = extract(
            "mem_cache/kv_cache_configurator.py", {"_build_dsa_kv_pool"},
            namespace.copy(), "KVCacheConfigurator",
        )._build_dsa_kv_pool
        size = extract(
            "model_executor/pool_configurator.py", {"_compute_dsa_indexer_cell_size"},
            namespace.copy(), "DefaultPoolConfigurator",
        )._compute_dsa_indexer_cell_size
        kvc = NS(
            is_draft_worker=False, kv_cache_dtype=torch.float8_e4m3fn, device="cpu",
            layer_info=NS(start_layer=0, end_layer=78, num_effective_layers=78),
            model_config=NS(hf_config=NS(), kv_lora_rank=512, qk_rope_head_dim=64),
        )
        modules = {
            "sglang.srt.layers.cp.utils": NS(
                get_glm_dsa_cp_layer_shard_info=lambda _: (None, 1),
                get_layer_shard_range=None,
            ),
            "sglang.srt.mem_cache.kv_cache_configurator": NS(
                _should_elide_dsa_index_k=self.should_elide,
            ),
        }
        with patch.dict("sys.modules", modules):
            for enabled, expected_layers in ((False, 78), (True, 21)):
                with self.subTest(enabled=enabled):
                    self.enabled = enabled
                    pool = build(kvc, max_total_num_tokens=2048)
                    host = make_host(pool)
                    indexer_cell_bytes = size(None, kvc=kvc, num_layers=78)
                    stored_device_layers = sum(
                        buffer.shape[0] > 0 for buffer in pool.index_k_with_scale_buffer
                    )
                    self.assertEqual(stored_device_layers, expected_layers)
                    self.assertEqual(host.layer_num, expected_layers)
                    # Solver counts bytes per rank-local DCP slot: each slot
                    # corresponds to four virtual tokens in the indexer pool.
                    self.assertEqual(indexer_cell_bytes, host.size_per_token * 4)
                    self.assertEqual(indexer_cell_bytes, expected_layers * 132 * 4)


if __name__ == "__main__":
    unittest.main()
