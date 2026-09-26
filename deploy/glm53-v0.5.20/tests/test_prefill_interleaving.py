import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("interleaving", ROOT / "python/sglang/srt/managers/prefill_interleaving.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Candidate, Interleaver = module.Candidate, module.PrefillInterleaver


class InterleavingTest(unittest.TestCase):
    def setUp(self):
        self.state = Interleaver()
        self.long = object()

    def plan(self, sizes, **kw):
        self.candidates = [Candidate(object(), n, n + 512) for n in sizes]
        options = dict(remaining=131072, budget=16384, page=256, slots=39, kv_budget=1_000_000)
        options.update(kw)
        return self.state.plan(self.long, self.candidates, **options)

    def test_short_requests_finish_next_to_long_without_second_partial(self):
        for size in (4096, 5120, 6144, 7168, 10240, 15360):
            with self.subTest(size=size):
                p = self.plan([size])
                self.assertEqual(p.limit + size, 16384)
                self.assertEqual(p.selected, (self.candidates[0].request,))
                self.assertGreaterEqual(p.limit, 256)
        self.assertIsNone(self.plan([16384, 131072]).limit)

    def test_borrow_is_repaid_before_another_large_waiter(self):
        first = self.plan([15360])
        self.assertTrue(first.borrowed)
        self.state.settle(self.long, actual_tokens=first.limit, target=first.target, finished=False, contended=True)
        self.assertEqual(self.state.debt, 7168)
        second = self.plan([15360, 1024])
        self.assertFalse(second.borrowed)
        self.assertEqual(second.limit, 15360)
        self.state.settle(self.long, actual_tokens=second.limit, target=second.target, finished=False, contended=True)
        self.assertEqual(self.state.debt, 0)

    def test_memory_and_slots_do_not_reserve_unadmittable_waiters(self):
        self.assertIsNone(self.plan([1024], slots=0).limit)
        self.assertIsNone(self.plan([1024], kv_budget=16000).limit)
        p = self.plan([131072, 4096, 4096], slots=1)
        self.assertEqual(p.selected, (self.candidates[1].request,))
        self.assertEqual(p.limit, 12288)

    def test_page_rounding_and_scan_limit(self):
        self.assertEqual(self.plan([257]).limit, 15872)
        self.assertIsNone(self.plan([131072] * 128 + [1024]).limit)

    def test_last_chunk_new_request_and_cancellation_reset_debt(self):
        p = self.plan([10240], remaining=4096)
        self.assertFalse(p.borrowed)
        self.state.debt = 2048
        self.state.settle(self.long, actual_tokens=4096, target=p.target, finished=True, contended=True)
        self.assertIsNone(self.state.request)
        self.assertEqual(self.state.debt, 0)
        self.state.debt = 2048
        self.long = object()
        self.plan([])
        self.assertEqual(self.state.debt, 0)

    def test_continuous_short_stream_preserves_long_progress(self):
        remaining = 131072
        allocations = []
        for _ in range(32):
            if not remaining:
                break
            p = self.plan([15360, 1024], remaining=remaining)
            tokens = min(remaining, p.limit or 16384)
            remaining -= tokens
            allocations.append(tokens)
            self.state.settle(self.long, actual_tokens=tokens, target=p.target, finished=not remaining, contended=bool(p.selected))
        self.assertEqual(remaining, 0)
        for i in range(0, len(allocations) - 2, 2):
            self.assertGreaterEqual(sum(allocations[i:i+2]), 16384)


if __name__ == "__main__":
    unittest.main()
