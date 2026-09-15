"""Execute real scheduler methods with CPU fakes at allocator/worker boundaries."""

import ast
import enum
import importlib.util
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

if os.environ.get("H200_INSTALLED") == "1":
    ROOT = Path(
        next(iter(importlib.util.find_spec("sglang").submodule_search_locations))
    )
else:
    ROOT = Path(__file__).resolve().parents[3] / "python/sglang"


class Result(enum.Enum):
    CONTINUE = 0
    NO_TOKEN = 1
    OTHER = 2
    SKIP = 3


class ReachedShort(Exception):
    pass


def load_method(file, cls, method, globals_):
    tree = ast.parse((ROOT / file).read_text())
    node = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == cls
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == method
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / file), "exec"), globals_)
    return globals_[method]


def flags(park=False, skip=False, admit=False):
    return NS(
        SGLANG_ENABLE_H200_ADMIT_FULL_NEED=NS(get=lambda: admit),
        SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL=NS(get=lambda: park),
        SGLANG_ENABLE_H200_SKIP_NOT_FITTING=NS(get=lambda: skip),
    )


class ChunkTests(unittest.TestCase):
    def run_chunk(
        self,
        *,
        enabled=True,
        total=0,
        chunk=1024,
        running=True,
        prefill_only=False,
        swa=False
    ):
        req = NS(
            prefix_indices=[],
            full_untruncated_fill_ids=list(range(4096)),
            sampling_params=NS(max_new_tokens=128),
            retracted_stain=False,
        )
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", NS(length=end - start)
        )
        req.extend_range = NS(length=0)
        batch = NS(is_empty=lambda: not running, is_prefill_only=prefill_only)
        adder = NS(
            dllm_config=None,
            rem_chunk_tokens=chunk,
            rem_total_tokens=total,
            is_hybrid_swa=swa,
            rem_swa_tokens=0,
            page_size=64,
            running_batch=batch,
            prefill_delayer_single_pass=None,
            can_run_list=[],
            _update_prefill_budget=lambda *a, **kw: None,
            _mamba_gap_budget_for_req=lambda r: 0,
        )
        method = load_method(
            "srt/managers/schedule_policy.py",
            "PrefillAdder",
            "add_chunked_req",
            {"envs": flags(park=enabled), "CLIP_MAX_NEW_TOKENS": 4096},
        )
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
        reqs = [
            NS(
                rid=rid,
                beam_group=None,
                kv=NS(mamba_cow_src_index=9, mamba_needs_clear=True, holds_mamba=False),
                init_next_round_input=lambda tree: None,
            )
            for rid in ("large", "short")
        ]
        adder = NS(can_run_list=[], budget_state=lambda: budget)

        def add(req, **kwargs):
            touched.append(req.rid)
            if req.rid == "short":
                raise ReachedShort
            return result

        adder.add_one_req = add
        batch = NS(reqs=[], batch_is_full=False, is_empty=lambda: True)
        scheduler = NS(
            grammar_manager=NS(has_waiting_grammars=lambda: False),
            enable_hierarchical_cache=False,
            enable_priority_preemption=False,
            is_hybrid_swa=False,
            chunked_req=None,
            waiting_queue=reqs,
            min_free_slots_delayer=None,
            get_num_allocatable_reqs=lambda *a, **k: 100,
            policy=NS(calc_priority=lambda *a: None),
            chunked_prefill_size=4096,
            tp_worker=NS(model_runner=NS(attn_backend=NS())),
            page_size=64,
            tree_cache=NS(),
            token_to_kv_pool_allocator=NS(),
            new_token_ratio_tracker=NS(current=1),
            max_prefill_tokens=16384,
            is_mixed_chunk=False,
            priority_scheduling_preemption_threshold=0,
            max_prefill_bs=80,
            max_running_requests=80,
            dllm_config=None,
            enable_lora=False,
            req_to_token_pool=NS(),
            disaggregation_mode="null",
            enable_hicache_storage=False,
            truncation_align_size=None,
        )
        method = load_method(
            "srt/managers/scheduler.py",
            "Scheduler",
            "_get_new_batch_prefill_raw",
            {
                "envs": flags(skip=enabled),
                "get_memory": lambda: NS(enable_flexkv=False),
                "get_schedule": lambda: NS(prefill_max_requests=None),
                "TEST_RETRACT": False,
                "PrefillAdder": lambda *a, **k: adder,
                "DisaggregationMode": NS(PREFILL="prefill"),
                "AddReqResult": Result,
            },
        )
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
        self.assertEqual(touched, ["large", "short"])
        self.assertFalse(batch.batch_is_full)

    def test_switch_off_preserves_head_block(self):
        reached, touched, batch = self.run_queue(enabled=False)
        self.assertFalse(reached)
        self.assertEqual(touched, ["large"])
        self.assertTrue(batch.batch_is_full)

    def test_global_memory_exhaustion_stops_scan(self):
        self.assertFalse(self.run_queue(enabled=True, budget=Result.NO_TOKEN)[0])

    def test_chunk_budget_exhaustion_stops_scan(self):
        self.assertFalse(self.run_queue(enabled=True, budget=Result.OTHER)[0])

    def test_other_rejection_is_not_bypassed(self):
        self.assertFalse(self.run_queue(enabled=True, result=Result.OTHER)[0])

    def test_short_budget_rejection_can_scan_to_shorter_request(self):
        self.assertTrue(self.run_queue(enabled=True, result=Result.SKIP)[0])


class AdmissionTests(unittest.TestCase):
    def adder(self, free=100000):
        ns = {
            "AddReqResult": Result,
            "CLIP_MAX_NEW_TOKENS": 4096,
            "IGNORE_EOS_RESERVE_TOKENS": 0,
        }
        obj = NS(
            full_need_reqs=[],
            can_run_list=[],
            page_size=64,
            short_bypass_limit=512,
            rem_short_bypass=512,
            dllm_config=None,
            rem_chunk_tokens=0,
            rem_input_tokens=4096,
            token_to_kv_pool_allocator=NS(available_size=lambda: free),
            tree_cache=NS(evictable_size=lambda: 0, disable=False),
        )
        for name in (
            "full_need_fits",
            "_short_bypass_fits",
            "ceil_paged_tokens",
            "_update_prefill_budget",
            "budget_state",
            "add_one_req",
            "add_one_req_ignore_eos",
        ):
            method = load_method(
                "srt/managers/schedule_policy.py", "PrefillAdder", name, ns
            )
            setattr(obj, name, method.__get__(obj))
        return obj

    def req(self, prompt=1024, output=512, allocated=0, prefix=0, finished=False):
        req = NS(
            origin_input_ids=list(range(prompt)),
            output_ids=[],
            full_untruncated_fill_ids=list(range(prompt)),
            prefix_indices=list(range(prefix)),
            kv=NS(kv_allocated_len=allocated, cache_protected_len=0),
            sampling_params=NS(max_new_tokens=output, ignore_eos=False),
            finished=lambda: finished,
            host_hit_length=0,
            storage_hit_length=0,
            last_node=None,
            retracted_stain=False,
            needs_host_load_back=lambda: False,
        )
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", NS(length=end - start)
        )
        return req

    def test_all_microbatches_and_pending_extend_are_reserved_once(self):
        a = self.adder(free=1407)
        r1 = self.req(allocated=1024)  # 512 + page
        r2 = self.req(allocated=1280)  # 256 + page
        candidate = self.req(prompt=256, output=192)  # 448 + page
        a.full_need_reqs = [r1, r2, r1]
        a.can_run_list = [r2]
        self.assertFalse(a.full_need_fits(candidate))
        a.token_to_kv_pool_allocator.available_size = lambda: 1408
        self.assertTrue(a.full_need_fits(candidate))  # 576 + 320 + 512

    def test_incomplete_prefill_is_reserved(self):
        a = self.adder(free=1600)
        a.full_need_reqs = [self.req(prompt=2048, output=512, allocated=512)]
        self.assertFalse(a.full_need_fits(self.req(prompt=64, output=64)))

    def test_finished_request_does_not_hold_future_debt(self):
        a = self.adder(free=192)
        a.full_need_reqs = [self.req(prompt=9999, finished=True)]
        self.assertTrue(a.full_need_fits(self.req(prompt=64, output=64)))

    def test_locked_prefix_and_allocated_kv_are_not_double_charged(self):
        a = self.adder(free=256)
        self.assertTrue(
            a.full_need_fits(self.req(prompt=1024, output=192, prefix=1024))
        )

    def test_page_slack_and_unallocated_pending_requests_are_charged(self):
        a = self.adder(free=255)
        r = self.req(prompt=64, output=1)
        a.can_run_list = [r]
        self.assertFalse(a.full_need_fits(self.req(prompt=1, output=1)))

    def test_short_bypass_does_not_repeat_the_full_allowance(self):
        a = self.adder()
        a.rem_chunk_tokens = -448
        a.rem_short_bypass = 64
        self.assertFalse(a._short_bypass_fits(512))
        self.assertTrue(a._short_bypass_fits(64))
        self.assertFalse(a._short_bypass_fits(65))

    def configure_admission(self, a):
        a.rem_total_tokens = a.cur_rem_tokens = 100000
        a.rem_total_token_offset = a.cur_rem_token_offset = 0
        a.is_hybrid_swa = a.is_hybrid_ssm_cache = False
        a.rem_mamba_slots = None
        a.dsa_prefill_cp_in_seq_split = False
        a.prefill_max_requests = None
        a.prefill_delayer_single_pass = None
        a.req_states = None
        a.running_batch = None
        a.new_token_ratio = 1
        a._mamba_gap_budget_for_req = lambda r: 0
        a._check_prefill_tile_budget = lambda n: None
        a._lock_node = lambda n: nullcontext()
        a._req_inc_lock_ref = lambda r: None
        for name in (
            "log_hit_tokens",
            "log_input_tokens",
            "reprocessed_log_hit_tokens",
            "reprocessed_log_input_tokens",
            "log_device_hit_tokens",
            "log_host_hit_tokens",
            "log_storage_hit_tokens",
        ):
            setattr(a, name, 0)

    def test_two_admission_paths_share_a_bounded_extra_budget(self):
        for ignore_eos in (False, True):
            with self.subTest(ignore_eos=ignore_eos):
                a = self.adder()
                self.configure_admission(a)
                a.tree_cache.disable = ignore_eos
                for n in (448, 512, 64):
                    r = self.req(prompt=n, output=128)
                    r.sampling_params.ignore_eos = ignore_eos
                    result = a.add_one_req(
                        r, has_chunked_req=True, truncation_align_size=None
                    )
                    if n == 512:
                        self.assertEqual(result, Result.SKIP)
                        self.assertNotIn(r, a.can_run_list)
                self.assertEqual(
                    sum(r.extend_range.length for r in a.can_run_list), 512
                )
                self.assertEqual(a.rem_short_bypass, 0)
                self.assertEqual(a.budget_state(), Result.OTHER)


if __name__ == "__main__":
    unittest.main()
