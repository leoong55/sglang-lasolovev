"""Execute bounded mapping/CPU backing and L2 callbacks with CPU tensors.

Only MHA allocation and CUDA transfers are substituted. The actual ring,
versioning, filtering, host callbacks and selector staging code are executed.
"""

import ast
import logging
import gc
import os
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def extract(path, names, ns):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


class CPUMHAPool:
    def __init__(self, size, *, page_size, dtype, head_num, head_dim, v_head_dim, layer_num, device, **kw):
        self.size, self.page_size, self.dtype, self.device = size, page_size, dtype, device
        self.head_num, self.head_dim, self.layer_num = head_num, head_dim, layer_num
        self.layer_transfer_counter = None
        self.k_buffer = [torch.empty(size + page_size, head_num, head_dim, dtype=dtype) for _ in range(layer_num)]
        self.v_buffer = [torch.empty(size + page_size, head_num, v_head_dim, dtype=dtype) for _ in range(layer_num)]


class BoundedDraftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        ns = extract("layers/cp/glm53_draft_layout.py", {"bounded_draft_geometry"}, {})
        cls.geometry = staticmethod(ns["bounded_draft_geometry"])
        cls.ns = extract("mem_cache/glm53_bounded_draft.py", {"GLM53BoundedDraftPool", "bounded_draft_host_pool_class"}, {
            "torch": torch, "threading": threading, "logger": logging.getLogger(__name__),
            "MHATokenToKVPool": CPUMHAPool, "bounded_draft_geometry": cls.geometry,
            "weakref": weakref,
            "envs": NS(SGLANG_GLM53_BOUNDED_DRAFT_FASTPATH=NS(get=lambda: False)),
        })

    def pool(self, logical=16384, requests=2):
        return self.ns["GLM53BoundedDraftPool"](
            logical, max_requests=requests, page_size=256, dtype=torch.bfloat16,
            head_num=1, head_dim=128, v_head_dim=128, layer_num=6, device="cpu",
        )

    def data(self, n, salt=0):
        return ((torch.arange(n) + salt) % 113).to(torch.bfloat16)[:, None, None, None, None].expand(n, 6, 2, 1, 128).contiguous()

    def commit(self, pool, ids, req, pos, data):
        pool.commit_context(virtual=ids, requests=torch.full_like(ids, req), positions=pos, payload=data)

    def test_fixed_budget_matches_tensor_bytes_and_capacity_above_32(self):
        pool = self.pool()
        buffers = [*pool.k_buffer, *pool.v_buffer, pool.logical_versions, pool.slot_virtual, pool.slot_version]
        actual = sum(t.numel() * t.element_size() for t in buffers)
        self.assertEqual(actual, pool.fixed_gpu_bytes + 4 * pool.logical_size)
        for reqs in (32, 33, 48, 64, 96):
            rows, stride, fixed = self.geometry(reqs)
            self.assertEqual(rows, (reqs + 1) * stride)
            self.assertLess(reqs * stride + 2048 + 255 + 8, rows)
            self.assertEqual(fixed, (rows + 256) * 3084 + 1024)

    def test_actual_builder_selects_bounded_pool_and_solver_reserves_its_bytes(self):
        from test_chunk_dflash_pool import DraftPoolTest, extract as extract_methods
        fixture = DraftPoolTest()
        kvc = fixture.fixture()
        kvc.device = "cpu"
        kvc.kv_cache_dtype = torch.bfloat16
        kvc.model_config.hf_config = NS(sliding_window=2048, layer_types=["sliding_attention"] * 6)
        kvc._build_mha_kv_pool.__func__.__globals__["get_schedule"] = lambda: NS(max_running_requests=2, page_size=64, prefill_only_disable_kv_cache=False)
        mod = types.ModuleType("sglang.srt.mem_cache.glm53_bounded_draft")
        mod.GLM53BoundedDraftPool = self.ns["GLM53BoundedDraftPool"]
        with patch.dict(os.environ, {"SGLANG_GLM53_DRAFT_CACHE_WINDOW": "2048"}), patch.dict(sys.modules, {mod.__name__: mod}):
            pool = kvc._build_mha_kv_pool(max_total_num_tokens=16384, mha_pool_class=object)
            self.assertEqual(pool.size, 3 * 2560)
            self.assertEqual(pool.logical_size, 16384)
            cost_fn = extract_methods("model_executor/pool_configurator.py", {"_dflash_draft_cell_size"},
                {"get_parallel": lambda: fixture.parallel})["_dflash_draft_cell_size"]
            target = NS(is_draft_worker=False, spec_algorithm=kvc.spec_algorithm, spec_aux_config=kvc.spec_aux_config)
            self.assertEqual(cost_fn(target), 16)
            self.assertEqual(cost_fn(kvc), 0)
            layout = types.ModuleType("sglang.srt.layers.cp.glm53_draft_layout")
            layout.bounded_draft_geometry = self.geometry
            utils = types.ModuleType("sglang.srt.speculative.dflash_utils")
            utils.scale_kv_cell_size_per_token_for_dflash = extract_methods("speculative/dflash_utils.py",
                {"scale_kv_cell_size_per_token_for_dflash"}, {})["scale_kv_cell_size_per_token_for_dflash"]
            methods = extract_methods("model_executor/pool_configurator.py", {"__init__", "calculate_pool_sizes"},
                {"MemoryPoolConfig": lambda **kw: NS(**kw), "torch": torch, "mambaish_config": lambda _: None,
                 "get_parallel": lambda: fixture.parallel, "get_schedule": lambda: NS(max_running_requests=2, max_total_tokens=None),
                 "_dflash_draft_cell_size": cost_fn}, "DefaultPoolConfigurator")
            cls = type("Solver", (), {**methods, "_compute_cell_size": lambda *args: 1000})
            target.kv_cache_dtype_str = "fp8_e4m3"
            target.model_config = NS(context_len=131072)
            target.layer_info = NS(num_effective_layers=78)
            target.spec_algorithm.is_eagle = lambda: False
            target.spec_algorithm.is_standalone = lambda: False
            target.spec_aux_config.dflash_draft_num_layers = 6
            with patch.dict(sys.modules, {layout.__name__: layout, utils.__name__: utils}):
                solver = cls(target)
            self.assertEqual(solver._bounded_fixed_bytes, pool.fixed_gpu_bytes)
            self.assertEqual(solver._cell_size, 1016)
            available = pool.fixed_gpu_bytes + 4096 * 1016
            self.assertEqual(solver.calculate_pool_sizes(available, 64).max_total_num_tokens, 4096)
            kvc.model_config.hf_config.sliding_window = 4096
            with self.assertRaisesRegex(ValueError, "window2048"):
                kvc._build_mha_kv_pool(max_total_num_tokens=16384, mha_pool_class=object)

    def test_long_prefill_wrap_then_decode_retains_exact_context(self):
        pool = self.pool()
        ids, positions = torch.arange(9000), torch.arange(9000)
        data = self.data(9000)
        self.commit(pool, ids, 1, positions, data)
        self.assertTrue(torch.equal(pool.backing[:9000], data))
        target = torch.zeros(3, 10000, dtype=torch.int32)
        draft = torch.zeros_like(target)
        target[1, :9000] = ids.int()
        visible, scratch = pool.prepare_window(target_table=target, draft_table=draft,
            request_ids=torch.tensor([1]), prefix_lens=torch.tensor([9000]), block_size=8)
        start = (9000 - 2048) // 256 * 256
        for layer in range(6):
            slots = draft[1, :visible.item()].long()
            self.assertTrue(torch.equal(pool.k_buffer[layer][slots], data[start:, layer, 0]))
        self.assertTrue(bool((pool.slot_virtual[scratch] == -1).all()))
        new_ids = torch.arange(9000, 9005)
        self.commit(pool, new_ids, 1, new_ids, self.data(5, 77))
        target[1, 9000:9005] = new_ids.int()
        pool.prepare_window(target_table=target, draft_table=draft,
            request_ids=torch.tensor([1]), prefix_lens=torch.tensor([9005]), block_size=8)
        self.assertTrue(bool(pool.backing_valid[9000:9005].all()))
        self.assertFalse(bool(pool.backing_valid[9005:9008].any()))

    def test_shared_prefix_branch_and_reused_virtual_ids_refresh_ring(self):
        pool = self.pool()
        ids = torch.arange(4096)
        first = self.data(4096)
        self.commit(pool, ids, 1, ids, first)
        target = torch.zeros(3, 5000, dtype=torch.int32)
        target[1, :4096] = ids.int(); target[2, :4096] = ids.int()
        draft = torch.zeros_like(target)
        pool.prepare_window(target_table=target, draft_table=draft,
            request_ids=torch.tensor([1, 2]), prefix_lens=torch.tensor([4096, 4096]), block_size=8)
        self.assertTrue(torch.equal(pool.k_buffer[0][draft[2, :2048].long()], first[2048:, 0, 0]))
        replacement = self.data(4096, 19)
        self.commit(pool, ids, 1, ids, replacement)
        pool.prepare_window(target_table=target, draft_table=draft,
            request_ids=torch.tensor([2]), prefix_lens=torch.tensor([4096]), block_size=8)
        self.assertTrue(torch.equal(pool.k_buffer[0][draft[2, :2048].long()], replacement[2048:, 0, 0]))

    def test_l2_relocation_restores_all_layers_before_window_reuse(self):
        pool = self.pool()
        ids = torch.arange(4096); data = self.data(4096)
        self.commit(pool, ids, 1, ids, data)
        stub = types.ModuleType("sglang.srt.mem_cache.pool_host.mha")
        stub.MHATokenToKVPoolHost = object
        with patch.dict(sys.modules, {stub.__name__: stub}):
            host = self.ns["bounded_draft_host_pool_class"]()()
        host.layout = "layer_first"
        host.k_buffer = torch.empty(6, 8192, 1, 128, dtype=torch.bfloat16)
        host.v_buffer = torch.empty_like(host.k_buffer)
        host_ids = ids + 256
        host.backup_from_device_all_layer(pool, host_ids, ids, "direct")
        pool.clear_bounded_cache()
        relocated = ids + 8192
        for layer in range(6):
            host.load_to_device_per_layer(pool, host_ids, relocated, layer, "direct", is_draft=True)
            self.assertEqual(bool(pool.backing_valid[relocated].all()), layer == 5)
        self.assertTrue(torch.equal(pool.backing[relocated], data))
        target = torch.zeros(3, 5000, dtype=torch.int32); target[2, :4096] = relocated.int()
        draft = torch.zeros_like(target)
        pool.prepare_window(target_table=target, draft_table=draft,
            request_ids=torch.tensor([2]), prefix_lens=torch.tensor([4096]), block_size=8)
        for layer in range(6):
            self.assertTrue(torch.equal(pool.v_buffer[layer][draft[2, :2048].long()], data[2048:, layer, 1]))

    def test_missing_context_rejected_and_empty_commit_is_noop(self):
        pool = self.pool()
        ids = torch.empty(0, dtype=torch.long)
        self.commit(pool, ids, 1, ids, self.data(0))
        table = torch.zeros(3, 4096, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "unmaterialized"):
            pool.prepare_window(target_table=table, draft_table=table.clone(),
                request_ids=torch.tensor([1]), prefix_lens=torch.tensor([2048]), block_size=8)

    def test_last_allocator_page_and_restore_barrier(self):
        pool = self.pool()
        # The allocator reserves page zero; its final page extends past size.
        ids = torch.arange(pool.logical_size, pool.logical_size + 256)
        self.commit(pool, ids, 2, torch.arange(256), self.data(256))
        waited = []
        pool.layer_transfer_counter = NS(consumer_index=0, wait_until=waited.append)
        target = torch.zeros(3, 4096, dtype=torch.int32)
        target[2, :256] = ids.int()
        draft = torch.zeros_like(target)
        visible, scratch = pool.prepare_window(target_table=target, draft_table=draft,
            request_ids=torch.tensor([2]), prefix_lens=torch.tensor([256]), block_size=8)
        self.assertEqual(waited, [5])
        self.assertEqual(visible.item(), 256)
        self.assertEqual(scratch.numel(), 8)
        self.assertTrue(torch.equal(pool.k_buffer[0][draft[2, :256].long()], self.data(256)[:, 0, 0]))

    def test_worker_projection_commits_only_verified_prefix_rows(self):
        from test_chunk_dflash_pool import extract as extract_methods
        method = extract_methods("speculative/dflash_worker_v2.py",
            {"_append_target_hidden_to_draft_kv_by_loc"}, {"torch": torch}, "DFlashWorkerV2")
        worker = type("Worker", (), method)()
        worker.use_bounded_draft_cache = True
        worker.model_runner = NS(device=torch.device("cpu"))
        pool = self.pool()
        worker.draft_model_runner = NS(token_to_kv_pool=pool)
        layers = []
        for i in range(6):
            attn = NS(kv_proj_only=lambda h: (h, h + 3),
                      apply_k_norm=lambda k: k * 2,
                      apply_k_rope=lambda p, k: k + p[:, None])
            layers.append(NS(index=i, self_attn=attn))
        worker.draft_model = NS(layers=layers, project_target_hidden=lambda h: h + 1,
            prepare_context_hidden_for_kv=lambda layer, h: h + layer.index)
        locs = torch.arange(256, 272).reshape(2, 8)
        positions = torch.arange(8).repeat(2)
        hidden = torch.arange(16, dtype=torch.bfloat16)[:, None].expand(16, 128)
        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden, cache_loc=locs.flatten(), positions=positions,
            cache_loc_2d=locs, commit_lens=torch.tensor([1, 8]),
            context_request_ids=torch.tensor([1] * 8 + [2] * 8))
        keep = torch.tensor([0] + list(range(8, 16)))
        self.assertTrue(bool(pool.backing_valid[locs.flatten()[keep]].all()))
        self.assertFalse(bool(pool.backing_valid[257:264].any()))
        for layer in range(6):
            expected_k = ((hidden + 1 + layer) * 2 + positions[:, None])[keep, None, :]
            expected_v = (hidden + 1 + layer + 3)[keep, None, :]
            self.assertTrue(torch.equal(pool.backing[locs.flatten()[keep], layer, 0], expected_k))
            self.assertTrue(torch.equal(pool.backing[locs.flatten()[keep], layer, 1], expected_v))

    def test_selector_graph_buffers_do_not_resize_on_33_or_48_requests(self):
        ns = extract("speculative/dflash_worker_v2.py", {"_SelectorDraftSampler"}, {
            "torch": torch,
            "resolve_greedy_mask": lambda **kw: torch.zeros(kw["bs"], dtype=torch.bool),
        })
        sampler = ns["_SelectorDraftSampler"](
            draft_model=NS(candidate_selector=NS(top_k=16)), block_size=8, max_bs=32, device="cpu")
        pointers = [x.data_ptr() for x in (sampler.temperatures, sampler.greedy_mask, sampler.out)]
        for bs in (32, 33, 48, 64, 16, 32):
            sampler.stage_sampling_params(bs=bs, sampling_info=NS(temperatures=torch.full((bs, 1), .3)))
            self.assertEqual(sampler.temperatures.shape, (32,))
            self.assertEqual(pointers, [x.data_ptr() for x in (sampler.temperatures, sampler.greedy_mask, sampler.out)])
        self.assertTrue(torch.allclose(sampler.temperatures, torch.full((32,), .3)))

    def owner(self, slot, prefix):
        # Real Req instances and Tensor prefixes support weak references.
        class Request:
            pass
        owner = Request()
        owner.kv = NS(req_pool_idx=slot, cache_protected_len=len(prefix))
        owner.prefix_indices = prefix.clone()
        owner.is_retracted = False
        owner.retracted_stain = False
        return owner

    def prepare_owned(self, pool, table, draft, owners, lengths):
        return pool.prepare_window(
            target_table=table, draft_table=draft,
            request_ids=torch.tensor([o.kv.req_pool_idx for o in owners]),
            prefix_lens=torch.tensor(lengths), block_size=8,
            request_owners=owners,
        )

    def test_continuous_decode_reorders_batches_and_matches_every_ring_row(self):
        pool = self.pool(logical=24000)
        pool.fastpath_enabled = True
        target = torch.zeros(3, 12000, dtype=torch.int32)
        draft = torch.zeros_like(target)
        owners = []
        lengths = {1: 4096, 2: 4096}
        for slot, base in ((1, 0), (2, 12000)):
            ids = base + torch.arange(4096)
            self.commit(pool, ids, slot, torch.arange(4096), self.data(4096, slot))
            target[slot, :4096] = ids.int()
            owners.append(self.owner(slot, ids))
        # More than one ring rotation; accepted lengths vary and batch order
        # changes, so validation cannot rely on fixed rows or a fixed accept_len.
        for iteration in range(560):
            current = owners if iteration % 2 else owners[::-1]
            lens = [lengths[o.kv.req_pool_idx] for o in current]
            visible, _ = self.prepare_owned(pool, target, draft, current, lens)
            for owner, end, size in zip(current, lens, visible.tolist()):
                slot = owner.kv.req_pool_idx
                ids = target[slot, end - size:end].long()
                physical = draft[slot, :size].long()
                self.assertTrue(torch.equal(pool.k_buffer[0][physical], pool.backing[ids, 0, 0]))
                self.assertTrue(torch.equal(pool.v_buffer[5][physical], pool.backing[ids, 5, 1]))
            all_ids, all_pos, all_req, all_data = [], [], [], []
            for owner in current:
                slot = owner.kv.req_pool_idx
                count = (1, 7, 8)[(iteration + slot) % 3]
                pos = torch.arange(lengths[slot], lengths[slot] + count)
                ids = pos + (0 if slot == 1 else 12000)
                target[slot, pos] = ids.int()
                all_ids.append(ids); all_pos.append(pos)
                all_req.append(torch.full_like(ids, slot))
                all_data.append(self.data(count, iteration))
                lengths[slot] += count
            pool.commit_context(virtual=torch.cat(all_ids), positions=torch.cat(all_pos),
                requests=torch.cat(all_req), payload=torch.cat(all_data), is_decode=True)
        self.assertEqual(pool.window_checks, 1)
        self.assertEqual(pool.window_reuses, 559)

    def test_continuity_rejects_repoint_slot_reuse_retraction_and_prepare_only(self):
        pool = self.pool()
        pool.fastpath_enabled = True
        ids = torch.arange(4096)
        self.commit(pool, ids, 1, ids, self.data(4096))
        table = torch.zeros(3, 5000, dtype=torch.int32)
        table[1, :4096] = ids.int(); table[2, :4096] = ids.int()
        draft = torch.zeros_like(table)
        owner = self.owner(1, ids)
        self.prepare_owned(pool, table, draft, [owner], [4096])
        self.prepare_owned(pool, table, draft, [owner], [4096])
        self.assertEqual(pool.window_checks, 2)  # no accepted commit in between
        changes = (
            lambda: setattr(owner, "prefix_indices", owner.prefix_indices.clone()),
            lambda: setattr(owner.kv, "cache_protected_len", 3840),
            lambda: setattr(owner.kv, "req_pool_idx", 2),
            lambda: setattr(owner, "retracted_stain", True),
            lambda: setattr(owner, "is_retracted", True),
        )
        for change in changes:
            pool._continuity_ready = True
            change()
            before = pool.window_checks
            self.prepare_owned(pool, table, draft, [owner], [4096])
            self.assertEqual(pool.window_checks, before + 1)
        # A new request in the same physical slot does not inherit ownership.
        replacement = self.owner(2, ids)
        pool._continuity_ready = True
        before = pool.window_checks
        self.prepare_owned(pool, table, draft, [replacement], [4096])
        self.assertEqual(pool.window_checks, before + 1)

    def test_late_prefill_radix_repoint_reloads_different_cached_values(self):
        pool = self.pool()
        pool.fastpath_enabled = True
        ids = torch.arange(4096)
        replacement = ids + 8192
        self.commit(pool, ids, 1, ids, self.data(4096))
        self.commit(pool, replacement, 2, ids, self.data(4096, 23))
        table = torch.zeros(3, 5000, dtype=torch.int32)
        table[1, :4096] = ids.int()
        draft = torch.zeros_like(table)
        owner = self.owner(1, ids)
        self.prepare_owned(pool, table, draft, [owner], [4096])
        # Overlap may execute first decode before processing the prior prefill
        # result. That result can repoint the radix prefix without a new forward.
        pool._continuity_ready = True
        table[1, :4096] = replacement.int()
        owner.prefix_indices = replacement.clone()
        self.prepare_owned(pool, table, draft, [owner], [4096])
        self.assertEqual(pool.window_checks, 2)
        self.assertTrue(torch.equal(pool.k_buffer[0][draft[1, :2048].long()],
            self.data(4096, 23)[2048:, 0, 0]))

    def test_prefill_restore_clear_and_disabled_flag_force_validation(self):
        pool = self.pool()
        pool.fastpath_enabled = True
        ids = torch.arange(4096)
        self.commit(pool, ids, 1, ids, self.data(4096))
        table = torch.zeros(3, 5000, dtype=torch.int32); table[1, :4096] = ids.int()
        draft = torch.zeros_like(table)
        owner = self.owner(1, ids)
        self.prepare_owned(pool, table, draft, [owner], [4096])
        pool._continuity_ready = True
        pool.restore_epoch += 1
        self.prepare_owned(pool, table, draft, [owner], [4096])
        self.assertEqual(pool.window_checks, 2)
        self.commit(pool, ids, 1, ids, self.data(4096, 7))
        self.prepare_owned(pool, table, draft, [owner], [4096])
        self.assertEqual(pool.window_checks, 3)
        pool._continuity_ready = True
        pool.fastpath_enabled = False
        self.prepare_owned(pool, table, draft, [owner], [4096])
        self.assertEqual(pool.window_checks, 4)
        pool.clear_bounded_cache()
        self.assertFalse(pool._window_owners)
        self.assertFalse(pool._continuity_ready)

    def test_owner_snapshots_do_not_retain_finished_requests_or_prefixes(self):
        pool = self.pool()
        owner = self.owner(1, torch.arange(4096))
        req_ref, prefix_ref = weakref.ref(owner), weakref.ref(owner.prefix_indices)
        pool._remember_window_owners([owner], 1)
        del owner
        gc.collect()
        self.assertIsNone(req_ref())
        self.assertIsNone(prefix_ref())

    def test_host_length_bound_uses_logical_page_and_is_safe_across_page_wrap(self):
        pool = self.pool()
        true_lens = torch.arange(1, 10000)
        visible = true_lens - torch.clamp(true_lens - 2048, min=0) // 256 * 256
        out = torch.empty_like(true_lens)
        for extra in (0, 1, 7, 8, 256):
            host = true_lens + extra
            pool.fill_seq_lens_cpu_bound(prefix_lens_cpu=host, reserved_lens_cpu=None,
                visible_lens=None, out=out)
            self.assertTrue(bool((out >= visible).all()))
            self.assertLessEqual(out.max().item(), 2303)
            pool.fill_seq_lens_cpu_bound(prefix_lens_cpu=None, reserved_lens_cpu=host,
                visible_lens=None, out=out)
            self.assertTrue(bool((out >= visible).all()))
        pool.fill_seq_lens_cpu_bound(prefix_lens_cpu=None, reserved_lens_cpu=None,
            visible_lens=visible, out=out)
        self.assertTrue(torch.equal(out, visible))


if __name__ == "__main__":
    unittest.main()
