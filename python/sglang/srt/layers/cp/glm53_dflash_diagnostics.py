"""Bounded, opt-in DFlash stage timings and explicit graph coverage.

CUDA events measure elapsed time on the worker stream, including launch gaps
and waits; they are NOT a sum of kernel execution times. No synchronize/item
is used to collect a sample. Pending events and pinned host buffers are bounded.
"""

import json
import logging
import os
import time
from collections import deque

logger = logging.getLogger(__name__)
PHASES = ("prepare", "draft", "verify", "accept", "cache")


def graph_inputs(runner, batch, *, replayed):
    """Host-only evidence for an admission decision; never read CUDA tensors.

    The runner remains the authority for admission. These fields explain the
    common rejection gates without changing them or invoking another forward.
    """
    if runner is None:
        return {"runner_present": False, "reasons": ["not_captured"]}
    bs = batch.batch_size
    width = getattr(batch.spec_info, "num_tokens_per_req", 0)
    captured_width = getattr(runner, "captured_req_width", None)
    capture_bs = list(getattr(runner, "capture_bs", ()))
    raw_counts = getattr(batch, "original_global_num_tokens_cpu", None)
    admission_bs = (
        max(raw_counts) if getattr(runner, "require_mlp_tp_gather", False) and raw_counts
        else bs
    )
    reasons = []
    if not replayed:
        if getattr(runner, "require_mlp_sync", False) and not batch.can_run_decode_cuda_graph:
            reasons.append("scheduler_vote_false")
        if width and captured_width is not None and width != captured_width:
            reasons.append("request_width_mismatch")
        if not capture_bs or admission_bs > max(capture_bs):
            reasons.append("above_capture_capacity")
        elif getattr(runner, "disable_padding", False) and admission_bs not in capture_bs:
            reasons.append("exact_bucket_missing")
        if not reasons:
            reasons.append("other_admission_check")
    replay_bs = getattr(runner, "bs", None) if replayed else None
    return dict(
        runner_present=True, reasons=reasons, requests=bs,
        captured_width=captured_width, requested_width=width,
        capture_requests=capture_bs, replay_requests=replay_bs,
        replay_rows=(replay_bs * captured_width
                     if replay_bs is not None and captured_width is not None else None),
        require_mlp_sync=getattr(runner, "require_mlp_sync", False),
        scheduler_vote=bool(batch.can_run_decode_cuda_graph),
        raw_request_counts=raw_counts,
        scaled_token_counts=getattr(batch, "global_num_tokens_cpu", None),
    )


class DFlashDiagnostics:
    def __init__(self, device, rank, *, tensor_api=None):
        import torch

        self.torch = tensor_api or torch
        self.device, self.rank = device, rank
        self.policy = os.environ.get("SGLANG_GLM53_DFLASH_GRAPH_POLICY", "warn")
        self.limit = int(os.environ.get("SGLANG_GLM53_DFLASH_PROFILE_STEPS", "0"))
        self.min_bs = int(os.environ.get("SGLANG_GLM53_DFLASH_PROFILE_MIN_BS", "1"))
        if self.policy not in ("warn", "require") or self.limit < 0 or self.min_bs < 1:
            raise ValueError("Invalid GLM53 DFlash diagnostic settings")
        # One rank is enough for a compact first diagnosis; not a cross-rank
        # maximum. A full distributed profiler is still needed for imbalance.
        if rank != 0:
            self.limit = 0
        self.pending = deque()
        self.active = None
        self.started = 0
        self.finished = 0
        self.coverage = {}
        self.stats = {}
        self.reported = False

    def start(self, bs):
        self.poll()
        self.active = None
        if bs < self.min_bs or self.started >= self.limit or len(self.pending) >= 4:
            return
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(self.torch.cuda.current_stream(self.device))
        self.active = dict(bs=bs, events=[event], cpu=[time.perf_counter()])
        self.started += 1

    def mark(self, phase):
        if self.active is None:
            return
        if phase != PHASES[len(self.active["events"]) - 1]:
            raise RuntimeError("DFLASH diagnostic stages are out of order")
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(self.torch.cuda.current_stream(self.device))
        self.active["events"].append(event)
        self.active["cpu"].append(time.perf_counter())

    def graph_status(self, *, bs, width, draft, verify, sampler,
                     draft_runner=None, draft_batch=None, verify_runner=None):
        key = (bs, width, bool(draft), bool(verify), bool(sampler))
        self.coverage[key] = self.coverage.get(key, 0) + 1
        if self.rank == 0 and self.coverage[key] == 1:
            log = logger.info if draft and verify and sampler else logger.warning
            log(
                "GLM53 DFLASH execution: requests=%d, width=%d, target_rows=%d, "
                "draft_graph=%s, verify_graph=%s, sampler_in_graph=%s",
                bs, width, bs * width, draft, verify, sampler,
            )
            if draft_batch is not None:
                verify_bs = getattr(verify_runner, "bs", None) if verify else None
                details = dict(
                    draft=graph_inputs(draft_runner, draft_batch, replayed=draft),
                    verify_replay_requests=verify_bs,
                    verify_replay_rows=verify_bs * width if verify_bs is not None else None,
                    logical_verify_rows=bs * width,
                )
                log("GLM53 DFLASH graph inputs: %s", json.dumps(details, sort_keys=True))
        if self.policy == "require" and not (draft and verify and sampler):
            raise RuntimeError(
                "GLM53 DFLASH required graphs were not used (draft, verify, sampler); "
                "inspect capture memory and request buckets, or explicitly select "
                "--glm53-dflash-graph-policy warn for an eager diagnostic run"
            )

    def finish(self, commit_lens):
        if self.active is None:
            return
        sample, self.active = self.active, None
        if len(sample["events"]) != len(PHASES) + 1:
            raise RuntimeError("DFLASH diagnostic sample is incomplete")
        host = self.torch.empty(
            sample["bs"], dtype=self.torch.int32, device="cpu", pin_memory=True
        )
        host.copy_(commit_lens.detach(), non_blocking=True)
        done = self.torch.cuda.Event()
        done.record(self.torch.cuda.current_stream(self.device))
        sample.update(host=host, done=done)
        self.pending.append(sample)
        self.poll()

    def poll(self):
        while self.pending and self.pending[0]["done"].query():
            sample = self.pending.popleft()
            bs = sample["bs"]
            stats = self.stats.setdefault(bs, dict(
                samples=0, committed=0, stream_ms=[0.0] * len(PHASES),
                host_ms=[0.0] * len(PHASES),
            ))
            stats["samples"] += 1
            # Host tensor is read only AFTER the copy completion event.
            stats["committed"] += int(sample["host"].sum())
            for i in range(len(PHASES)):
                stats["stream_ms"][i] += sample["events"][i].elapsed_time(sample["events"][i + 1])
                stats["host_ms"][i] += 1000 * (sample["cpu"][i + 1] - sample["cpu"][i])
            self.finished += 1
        if self.limit and self.finished == self.limit and not self.reported:
            self.reported = True
            logger.info("GLM53 DFLASH profile: %s", json.dumps(self.report(), sort_keys=True))

    def report(self):
        batches = {}
        for bs, stats in sorted(self.stats.items()):
            count = stats["samples"]
            committed = stats["committed"]
            batches[bs] = dict(
                samples=count,
                mean_committed_tokens_per_request=committed / (count * bs),
                stream_elapsed_ms_per_step=sum(stats["stream_ms"]) / count,
                stream_elapsed_ms_per_output_token=(sum(stats["stream_ms"]) * bs / committed) if committed else None,
                stream_elapsed_ms={name: stats["stream_ms"][i] / count for i, name in enumerate(PHASES)},
                host_elapsed_ms={name: stats["host_ms"][i] / count for i, name in enumerate(PHASES)},
            )
        return dict(rank=self.rank, samples=self.finished, by_batch_size=batches,
                    timing="worker-stream elapsed includes launch gaps/waits; host times overlap GPU execution")
