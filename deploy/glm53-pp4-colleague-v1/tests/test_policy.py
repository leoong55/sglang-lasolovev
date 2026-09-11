import ast
import importlib.util
import unittest
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "python/sglang/srt/managers"
spec = importlib.util.spec_from_file_location(
    "pp_full_need", POLICY / "pp_full_need.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def req(prompt=75000, output=1000, held=0, prefix=0, finished=False):
    r = NS(
        origin_input_ids=range(prompt),
        output_ids=[],
        prefix_indices=range(prefix),
        kv=NS(kv_allocated_len=held),
        sampling_params=NS(max_new_tokens=output),
        finished=lambda: finished,
    )
    return r


class BudgetTests(unittest.TestCase):
    def budget(self, live=(), free=10**9, short=512):
        return m.FullNeedBudget(lambda: list(live), lambda: free, 64, 40, short)

    def test_duplicate_across_microbatches_is_charged_once(self):
        a, b = req(), req()
        cost = m.future_tokens(a, 64) + m.future_tokens(b, 64)
        self.assertTrue(self.budget([a, a], cost).can_admit(b, [a]))
        self.assertFalse(self.budget([a, a], cost - 1).can_admit(b, [a]))

    def test_other_microbatch_unfinished_prefill_is_reserved(self):
        a = req(held=4096, prefix=4096)
        self.assertFalse(self.budget([a], 80000).can_admit(req(), []))

    def test_global_40_includes_chunked_and_excludes_finished(self):
        live = [req() for _ in range(40)]
        self.assertFalse(self.budget(live).can_admit(req(), []))
        self.assertTrue(self.budget(live).can_admit(live[0], [live[0]]))
        live[0].finished = lambda: True
        self.assertTrue(self.budget(live).can_admit(req(), []))

    def test_device_prefix_counts_host_prefix_does_not(self):
        a = req(prefix=60000)
        self.assertEqual(m.future_tokens(a, 64), 16064)
        a.host_hit_length = 60000
        a.prefix_indices = []
        self.assertEqual(m.future_tokens(a, 64), 76096)

    def test_max_new_is_not_clipped_or_scaled(self):
        a = req(prompt=75000, output=20000, held=75000)
        self.assertEqual(m.future_tokens(a, 64), 20096)

    def test_chunk_uses_own_reservation_and_parks_under_pressure(self):
        a, b = req(held=4096), req(held=75000)
        other = m.future_tokens(b, 64)
        self.assertEqual(
            self.budget([a, b], other + 4160).chunk_cap(a, [a], 4096), 4096
        )
        self.assertEqual(self.budget([a, b], other + 127).chunk_cap(a, [], 4096), 0)

    def test_short_bypass_is_a_total_batch_quota(self):
        budget = self.budget()
        self.assertTrue(budget.short_fits(448, 0))
        self.assertFalse(budget.short_fits(512, -448))
        self.assertTrue(budget.short_fits(64, -448))
        self.assertFalse(budget.short_fits(64, -512))
        self.assertFalse(budget.short_fits(1024, 0))
        self.assertFalse(self.budget(short=0).short_fits(64, 0))

    def test_fit_scan_bounds_successful_overtakes(self):
        a = req()
        for _ in range(8):
            scan = m.FitScan()
            self.assertTrue(scan.skip(a))
            scan.admitted()
        self.assertFalse(m.FitScan().skip(a))
        scan = m.FitScan(scan_limit=2)
        self.assertTrue(scan.skip(req()))
        self.assertTrue(scan.skip(req()))
        self.assertFalse(scan.skip(req()))


# Execute the real patched PrefillAdder methods with fake pools, without CUDA
# imports. This catches wiring/lock regressions, not just helper arithmetic.
source = ast.parse((POLICY / "schedule_policy.py").read_text())
selected = [
    n
    for n in source.body
    if isinstance(n, ast.ClassDef) and n.name in ("AddReqResult", "PrefillAdder")
]
future = ast.parse("from __future__ import annotations").body
ns = dict(Enum=Enum, auto=auto, contextmanager=contextmanager, CLIP_MAX_NEW_TOKENS=4096)
exec(
    compile(
        ast.Module(body=future + selected, type_ignores=[]),
        "schedule_policy.py",
        "exec",
    ),
    ns,
)
Adder = ns["PrefillAdder"]
Result = ns["AddReqResult"]


class IntegrationTests(unittest.TestCase):
    def adder(self, budget):
        a = Adder.__new__(Adder)
        a.full_need_budget = budget
        a.page_size = 64
        a.dllm_config = None
        a.is_hybrid_swa = a.is_all_swa = a.is_hybrid_ssm_cache = False
        a.rem_chunk_tokens = 4096
        a.rem_input_tokens = 65536
        a.rem_total_token_offset = a.cur_rem_token_offset = 0
        a.rem_mamba_slots = None
        a.can_run_list = []
        a.new_chunked_req = None
        a.prefill_delayer_single_pass = None
        a.dsa_prefill_cp_in_seq_split = False
        a.prefill_max_requests = None
        a._mamba_slot_cost = 0
        a.token_to_kv_pool_allocator = NS(available_size=lambda: 1000000)
        a.tree_cache = NS(disable=False, evictable_size=lambda: 0)
        return a

    def test_admission_check_is_inside_balanced_prefix_lock(self):
        events = []

        class Budget:
            def can_admit(self, r, selected):
                self_outer.assertEqual(events, ["lock"])
                return False

        self_outer = self
        a = self.adder(Budget())
        a.tree_cache.inc_lock_ref = lambda node: events.append("lock")
        a.tree_cache.dec_lock_ref = lambda node: events.append("unlock")
        a.tree_cache.is_tree_cache = lambda: False
        r = req(prefix=60000)
        r.sampling_params.ignore_eos = False
        r.full_untruncated_fill_ids = r.origin_input_ids
        r.host_hit_length = 0
        r.last_node = object()
        self.assertEqual(a.add_one_req(r, False, None), Result.NO_TOKEN)
        self.assertEqual(events, ["lock", "unlock"])
        self.assertEqual(a.can_run_list, [])

    def test_park_keeps_request_without_mutating_range(self):
        r = req()
        a = self.adder(NS(chunk_cap=lambda *args: 0))
        self.assertIs(a.add_chunked_req(r), r)
        self.assertEqual(a.can_run_list, [])

    def test_parked_chunk_prevents_second_long_prefill(self):
        a = self.adder(NS(short_fits=lambda *args: False))
        r = req()
        r.sampling_params.ignore_eos = False
        r.full_untruncated_fill_ids = r.origin_input_ids
        r.host_hit_length = 0
        self.assertEqual(a.add_one_req(r, True, None), Result.NO_TOKEN)
        self.assertIsNone(a.new_chunked_req)
        self.assertEqual(a.can_run_list, [])

    def test_budget_state_honors_total_short_quota(self):
        a = self.adder(NS(short_tokens=512))
        a.rem_chunk_tokens = -448
        self.assertEqual(a.budget_state(), Result.CONTINUE)
        a.rem_chunk_tokens = -512
        self.assertEqual(a.budget_state(), Result.OTHER)
        a.full_need_budget = None
        a.rem_chunk_tokens = 0
        self.assertEqual(a.budget_state(), Result.OTHER)



class SchedulerChunkAccountingTests(unittest.TestCase):
    def test_parked_chunk_is_not_counted_as_submitted_work(self):
        tree = ast.parse((POLICY / "scheduler.py").read_text())
        # Execute the actual scheduler's batch-construction block with a fake
        # batch factory. This exercises the in-flight counter and the metadata
        # consumed by PP result processing, without importing CUDA services.
        body = next(
            n.body
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and any(
                isinstance(s, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "batch_chunked_req"
                    for t in s.targets
                )
                for s in n.body
            )
        )
        start = next(
            i
            for i, s in enumerate(body)
            if isinstance(s, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "batch_chunked_req"
                for t in s.targets
            )
        )
        end = next(
            i
            for i, s in enumerate(body)
            if isinstance(s, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "contains_last_prefill_chunk"
                for t in s.targets
            )
        )
        code = compile(
            ast.Module(body=body[start : end + 1], type_ignores=[]),
            "scheduler_batch_construction",
            "exec",
        )

        class Chunk:
            inflight_middle_chunks = 0

        for selected_chunk, policy_on, expected_count in [
            (False, True, 0),
            (True, True, 1),
            (False, False, 1),
        ]:
            chunk = Chunk()
            selected = [chunk] if selected_chunk else [object()]
            scheduler = NS(
                chunked_req=chunk,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                tree_cache=None,
                model_config=None,
                enable_overlap=False,
                spec_algorithm=None,
            )
            frame = dict(
                self=scheduler,
                full_need_budget=object() if policy_on else None,
                can_run_set=set(selected),
                can_run_list=selected,
                set_time_batch=lambda *args: None,
                ScheduleBatch=NS(init_new=lambda *args, **kwargs: NS(**kwargs)),
            )
            exec(code, frame)
            self.assertEqual(chunk.inflight_middle_chunks, expected_count)
            self.assertIs(scheduler.chunked_req, chunk)
            self.assertEqual(
                frame["new_batch"].chunked_req is chunk, bool(expected_count)
            )
            self.assertEqual(
                frame["new_batch"].contains_last_prefill_chunk, not bool(expected_count)
            )


if __name__ == "__main__":
    unittest.main()
