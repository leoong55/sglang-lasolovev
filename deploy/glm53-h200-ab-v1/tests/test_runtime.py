"""Execute real scheduler methods with CPU fakes at allocator/worker boundaries."""
import ast
import enum
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

if os.environ.get('H200_INSTALLED') == '1':
    ROOT = Path(next(iter(importlib.util.find_spec('sglang').submodule_search_locations)))
else:
    ROOT = Path(__file__).resolve().parents[3]/'python/sglang'


class Result(enum.Enum):
    CONTINUE = 0
    NO_TOKEN = 1
    OTHER = 2


class ReachedShort(Exception):
    pass


def load_method(file, cls, method, globals_):
    tree = ast.parse((ROOT/file).read_text())
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT/file), 'exec'), globals_)
    return globals_[method]


def flags(park=False, skip=False):
    return NS(SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL=NS(get=lambda: park),
              SGLANG_ENABLE_H200_SKIP_NOT_FITTING=NS(get=lambda: skip))


class ChunkTests(unittest.TestCase):
    def run_chunk(self, *, enabled=True, total=0, chunk=1024, running=True, prefill_only=False, swa=False):
        req = NS(prefix_indices=[], full_untruncated_fill_ids=list(range(4096)),
                 sampling_params=NS(max_new_tokens=128), retracted_stain=False)
        req.set_extend_range = lambda start, end: setattr(req, 'extend_range', NS(length=end-start))
        req.extend_range = NS(length=0)
        batch = NS(is_empty=lambda: not running, is_prefill_only=prefill_only)
        adder = NS(dllm_config=None, rem_chunk_tokens=chunk, rem_total_tokens=total,
                   is_hybrid_swa=swa, rem_swa_tokens=0, page_size=64,
                   running_batch=batch, prefill_delayer_single_pass=None, can_run_list=[],
                   _update_prefill_budget=lambda *a, **kw: None,
                   _mamba_gap_budget_for_req=lambda r: 0)
        method = load_method('srt/managers/schedule_policy.py', 'PrefillAdder', 'add_chunked_req',
                             {'envs': flags(park=enabled), 'CLIP_MAX_NEW_TOKENS': 4096})
        result = method(adder, req)
        return req, adder, result

    def test_memory_pressure_parks_without_charging_or_advancing(self):
        req, adder, result = self.run_chunk()
        self.assertIs(result, req)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(req.extend_range.length, 0)

    def test_disabled_preserves_native_escape(self):
        req, adder, _ = self.run_chunk(enabled=False)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.extend_range.length, 1024)

    def test_lone_chunk_does_not_deadlock(self):
        req, adder, _ = self.run_chunk(running=False)
        self.assertEqual(adder.can_run_list, [req])

    def test_prefill_only_batch_cannot_release_decode_memory(self):
        req, adder, _ = self.run_chunk(prefill_only=True)
        self.assertEqual(adder.can_run_list, [req])

    def test_chunk_budget_zero_is_not_memory_pressure(self):
        req, adder, _ = self.run_chunk(total=100000, chunk=0)
        self.assertEqual(adder.can_run_list, [req])

    def test_positive_memory_admits_chunk(self):
        req, adder, _ = self.run_chunk(total=100000)
        self.assertEqual(req.extend_range.length, 1024)

    def test_native_swa_parking_still_works(self):
        req, adder, result = self.run_chunk(enabled=False, swa=True)
        self.assertIs(result, req)
        self.assertEqual(adder.can_run_list, [])


class QueueTests(unittest.TestCase):
    def run_queue(self, *, enabled, result=Result.NO_TOKEN, budget=Result.CONTINUE):
        touched = []
        reqs = [NS(rid=rid, beam_group=None, kv=NS(mamba_cow_src_index=9,
                    mamba_needs_clear=True, holds_mamba=False),
                    init_next_round_input=lambda tree: None) for rid in ('large', 'short')]
        adder = NS(can_run_list=[], budget_state=lambda: budget)
        def add(req, **kwargs):
            touched.append(req.rid)
            if req.rid == 'short':
                raise ReachedShort
            return result
        adder.add_one_req = add
        batch = NS(reqs=[], batch_is_full=False, is_empty=lambda: True)
        scheduler = NS(grammar_manager=NS(has_waiting_grammars=lambda: False),
            enable_hierarchical_cache=False, enable_priority_preemption=False,
            is_hybrid_swa=False, chunked_req=None, waiting_queue=reqs,
            min_free_slots_delayer=None, get_num_allocatable_reqs=lambda *a, **k: 100,
            policy=NS(calc_priority=lambda *a: None), chunked_prefill_size=4096,
            tp_worker=NS(model_runner=NS(attn_backend=NS())), page_size=64,
            tree_cache=NS(), token_to_kv_pool_allocator=NS(),
            new_token_ratio_tracker=NS(current=1), max_prefill_tokens=16384,
            is_mixed_chunk=False, priority_scheduling_preemption_threshold=0,
            max_prefill_bs=80, max_running_requests=80, dllm_config=None,
            enable_lora=False, req_to_token_pool=NS(),
            disaggregation_mode='null', enable_hicache_storage=False,
            truncation_align_size=None)
        method = load_method('srt/managers/scheduler.py', 'Scheduler', '_get_new_batch_prefill_raw', {
            'envs': flags(skip=enabled), 'get_memory': lambda: NS(enable_flexkv=False),
            'get_schedule': lambda: NS(prefill_max_requests=None),
            'TEST_RETRACT': False, 'PrefillAdder': lambda *a, **k: adder,
            'DisaggregationMode': NS(PREFILL='prefill'), 'AddReqResult': Result})
        reached = False
        try:
            method(scheduler, None, batch)
        except ReachedShort:
            reached = True
        self.assertIsNone(reqs[0].kv.mamba_cow_src_index)
        self.assertFalse(reqs[0].kv.mamba_needs_clear)
        return reached, touched, batch

    def test_rejected_head_does_not_block_fitting_request(self):
        reached, touched, batch = self.run_queue(enabled=True)
        self.assertTrue(reached)
        self.assertEqual(touched, ['large', 'short'])
        self.assertFalse(batch.batch_is_full)

    def test_switch_off_preserves_head_block(self):
        reached, touched, batch = self.run_queue(enabled=False)
        self.assertFalse(reached)
        self.assertEqual(touched, ['large'])
        self.assertTrue(batch.batch_is_full)

    def test_global_memory_exhaustion_stops_scan(self):
        self.assertFalse(self.run_queue(enabled=True, budget=Result.NO_TOKEN)[0])

    def test_chunk_budget_exhaustion_stops_scan(self):
        self.assertFalse(self.run_queue(enabled=True, budget=Result.OTHER)[0])

    def test_other_rejection_is_not_bypassed(self):
        self.assertFalse(self.run_queue(enabled=True, result=Result.OTHER)[0])


if __name__ == '__main__':
    unittest.main()
