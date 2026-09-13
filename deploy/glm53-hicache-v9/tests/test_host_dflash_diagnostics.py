"""Completion and bounded-queue tests; event timing is simulated, not GPU data."""
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"])
spec = importlib.util.spec_from_file_location("diag", ROOT / "python/sglang/srt/layers/cp/glm53_dflash_diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class Event:
    clock = 0
    created = []

    def __init__(self, enable_timing=False):
        self.ready = False
        self.created.append(self)

    def record(self, stream):
        type(self).clock += 2
        self.timestamp = self.clock

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class DiagnosticsTest(unittest.TestCase):
    def probe(self, steps=4, rank=0):
        Event.clock, Event.created = 0, []
        api = NS(cuda=NS(Event=Event, current_stream=lambda _: None), int32=torch.int32,
                 empty=lambda n, **kw: torch.empty(n, dtype=torch.int32))
        with patch.dict(os.environ, {"SGLANG_GLM53_DFLASH_PROFILE_STEPS": str(steps),
                                     "SGLANG_GLM53_DFLASH_PROFILE_MIN_BS": "2",
                                     "SGLANG_GLM53_DFLASH_GRAPH_POLICY": "warn"}):
            return diag.DFlashDiagnostics("cpu", rank, tensor_api=api)

    def sample(self, probe):
        probe.start(2)
        for phase in diag.PHASES:
            probe.mark(phase)
        probe.finish(torch.tensor([1, 4]))

    def test_incomplete_copies_are_not_read_and_queue_is_bounded(self):
        probe = self.probe()
        for _ in range(7):
            self.sample(probe)
        self.assertEqual(probe.started, 4)
        self.assertEqual(len(probe.pending), 4)
        self.assertEqual(probe.report()["samples"], 0)
        for sample in probe.pending:
            sample["done"].ready = True
        probe.poll()
        result = probe.report()["by_batch_size"][2]
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["mean_committed_tokens_per_request"], 2.5)
        self.assertEqual(result["stream_elapsed_ms_per_step"], 10)
        self.assertEqual(result["stream_elapsed_ms_per_output_token"], 4)
        self.assertEqual(len(probe.pending), 0)

    def test_disabled_small_batch_and_other_rank_allocate_no_events(self):
        for steps, rank, bs in ((0, 0, 80), (32, 1, 80), (32, 0, 1)):
            probe = self.probe(steps, rank)
            probe.start(bs)
            probe.mark("prepare")
            probe.finish(torch.ones(bs, dtype=torch.int32))
            self.assertEqual(Event.created, [])

    def test_require_checks_both_graphs_and_selector(self):
        probe = self.probe(0)
        probe.policy = "require"
        probe.graph_status(bs=80, width=8, draft=True, verify=True, sampler=True)
        for draft, verify, sampler in ((False, True, True), (True, False, True), (True, True, False)):
            with self.assertRaisesRegex(RuntimeError, "required graphs"):
                probe.graph_status(bs=80, width=8, draft=draft, verify=verify, sampler=sampler)
        probe.policy = "warn"
        probe.graph_status(bs=80, width=8, draft=False, verify=False, sampler=False)


if __name__ == "__main__":
    unittest.main()
