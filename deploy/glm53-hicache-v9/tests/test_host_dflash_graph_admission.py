"""Run the real DFlash batch constructor, FB metadata and graph admission on CPU.

Only imports/model execution are substituted. No CUDA capture or attention
kernel result is claimed by this test. The scheduler's negative vote must stay
authoritative, and request units must not be mistaken for verify-token units.
"""

import ast
import dataclasses
import os
import sys
import unittest
from enum import IntEnum, auto
from functools import total_ordering
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from host_source import extract

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang"
fb_module = extract(
    "srt/model_executor/forward_batch_info.py",
    {"ForwardBatch", "ForwardMode", "CaptureHiddenMode"},
    namespace={"dataclass": dataclasses.dataclass,
               "ForwardBatchDeepSeekMHAMixin": object, "IntEnum": IntEnum,
               "auto": auto, "total_ordering": total_ordering,
               "is_pin_memory_available": lambda device: False},
)
scale_module = extract("srt/speculative/spec_info.py", {"spec_scale_global_num_tokens"})
admission_module = extract(
    "srt/model_executor/runner/decode_cuda_graph_runner.py", {"can_run_graph"},
    class_name="DecodeCudaGraphRunner",
)


def make_worker_batch(bs, width, *, vote=True, sync=True):
    """Execute the constructor statements reached by forward_batch_generation."""
    path = ROOT / "srt/speculative/dflash_worker_v2.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "DFlashWorkerV2")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "forward_batch_generation")
    start = next(i for i, n in enumerate(method.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "forward_batch")
    end = next(i for i in range(start + 1, len(method.body))
               if isinstance(method.body[i], ast.If))
    batch = NS(req_pool_indices=torch.arange(bs),
               can_run_decode_cuda_graph=vote,
               global_num_tokens=[bs] if sync else None,
               global_num_tokens_for_logprob=[bs] if sync else None)
    ns = dict(
        ForwardBatch=fb_module.ForwardBatch, ForwardMode=fb_module.ForwardMode,
        CaptureHiddenMode=fb_module.CaptureHiddenMode,
        SpeculativeAlgorithm=NS(DFLASH="DFLASH"),
        self=NS(device="cpu", _draft_block_spec_info=NS(
            num_tokens_per_req=width, num_tokens_for_logprob_per_req=width)),
        batch=batch, bs=bs, block_ids=torch.arange(bs * width).view(bs, width),
        draft_seq_lens=torch.ones(bs, dtype=torch.int32),
        draft_out_cache_loc=torch.arange(bs * width), draft_seq_lens_sum=bs,
        seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
        positions=torch.arange(bs * width), input_embeds=torch.ones(bs * width, 16),
    )
    with patch.dict(sys.modules, {"sglang.srt.speculative.spec_info": scale_module}):
        exec(compile(ast.Module(body=method.body[start:end], type_ignores=[]),
                     str(path), "exec"), ns)
    return ns["forward_batch"], batch


def runner(width=8, *, exact=False, gather=False):
    buckets = [1, 2, 4, 8, 16, 32, 40, 48, 64, 80]
    return NS(
        can_run_graph=admission_module.can_run_graph,
        ragged_verify_mode=False, captured_req_width=width,
        require_mlp_tp_gather=gather, require_mlp_sync=True,
        _max_dp_batch_size=lambda batch: max(batch.original_global_num_tokens_cpu),
        _make_graph_key=lambda bs, **kw: bs, _resolve_lora_variant=lambda batch: None,
        enable_pdmux=False, disable_padding=exact, max_bs=80, capture_bs=buckets,
        backend=NS(can_run=lambda batch, key: key in buckets),
        is_encoder_decoder=False, enable_two_batch_overlap=False,
        model_runner=NS(spec_algorithm=NS(is_ngram=lambda: False)),
    )


class DraftGraphAdmissionTest(unittest.TestCase):
    def admits(self, r, fb):
        return r.can_run_graph(r, fb)

    def test_worker_preserves_vote_and_scales_request_counts_once(self):
        for width in (2, 4, 8):
            for bs in (1, 2, 8, 32, 40, 80):
                with self.subTest(bs=bs, width=width):
                    fb, scheduled = make_worker_batch(bs, width)
                    self.assertTrue(fb.can_run_decode_cuda_graph)
                    self.assertEqual(fb.original_global_num_tokens_cpu, [bs])
                    self.assertEqual(fb.global_num_tokens_cpu, [bs * width])
                    self.assertEqual(fb.global_num_tokens_gpu.tolist(), [bs * width])
                    self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [bs * width])
                    self.assertEqual(scheduled.global_num_tokens, [bs])
                    self.assertIs(fb.req_pool_indices, scheduled.req_pool_indices)
                    self.assertTrue(self.admits(runner(width), fb))
                    # With a gathered group, bucket selection needs requests,
                    # not the width-scaled token count (which would exceed 80).
                    self.assertTrue(self.admits(runner(width, gather=True), fb))

    def test_legacy_defaults_reject_every_bucket(self):
        for bs in (1, 8, 32, 40, 80):
            fb, _ = make_worker_batch(bs, 8)
            legacy = dataclasses.replace(
                fb, can_run_decode_cuda_graph=False,
                original_global_num_tokens_cpu=None, global_num_tokens_cpu=None,
                global_num_tokens_gpu=None,
            )
            self.assertFalse(self.admits(runner(), legacy))
            self.assertTrue(self.admits(runner(), fb))

    def test_negative_scheduler_vote_is_not_overridden(self):
        for bs in (1, 8, 32, 40, 80):
            fb, _ = make_worker_batch(bs, 8, vote=False)
            self.assertFalse(fb.can_run_decode_cuda_graph)
            self.assertFalse(self.admits(runner(), fb))

    def test_capacity_width_and_exact_bucket_gates_remain(self):
        above, _ = make_worker_batch(81, 8)
        self.assertFalse(self.admits(runner(), above))
        mismatch, _ = make_worker_batch(40, 4)
        self.assertFalse(self.admits(runner(8), mismatch))
        missing, _ = make_worker_batch(33, 8)
        self.assertFalse(self.admits(runner(exact=True), missing))
        self.assertTrue(self.admits(runner(), missing))

    def test_plain_tp_without_sync_metadata_keeps_scheduler_vote(self):
        fb, _ = make_worker_batch(8, 8, sync=False)
        self.assertIsNone(fb.global_num_tokens_cpu)
        self.assertTrue(fb.can_run_decode_cuda_graph)
        r = runner()
        r.require_mlp_sync = False
        self.assertTrue(self.admits(r, fb))


if __name__ == "__main__":
    unittest.main()
