"""Opt-in prefill diagnostics; bounded labels and nonblocking device timing.

Enable with SGLANG_GLM53_PREFILL_DIAGNOSTICS=1. Only TP0/PP0 exports
Prometheus observations. CUDA ranges remain visible on every rank in traces.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import time
from collections import deque

logger = logging.getLogger(__name__)
REASONS = frozenset({"token_budget", "kv_capacity", "request_slots", "delayer",
                     "batch_full", "partial_prefill", "host_cache", "other"})
STAGES = frozenset({"dcp_gather", "dcp_reduce", "kv_convert", "indexer", "topk", "attention",
                    "moe", "draft_append", "hicache_consensus", "warmup"})
_pending = deque(maxlen=256)
_rank = 0
_export = False
_metrics = None


def enabled():
    return os.environ.get("SGLANG_GLM53_PREFILL_DIAGNOSTICS", "0") == "1"


def configure(*, rank, export):
    global _rank, _export
    _rank, _export = rank, export


def _get_metrics():
    global _metrics
    if _metrics is None:
        from prometheus_client import Counter, Histogram

        _metrics = (
            Counter("sglang:glm53_prefill_admission_rejected", "Admission rejections", ["reason"]),
            Histogram("sglang:glm53_prefill_stage_seconds", "Inclusive stage duration",
                      ["stage", "clock"], buckets=(.0001, .0005, .001, .005, .01, .05, .1, .5, 1, 5, 30)),
            Histogram("sglang:glm53_prefill_batch_tokens", "Uncached batch tokens",
                      buckets=(256, 1024, 4096, 8192, 12288, 16384, 32768)),
        )
    return _metrics


def record(kind, **fields):
    if not enabled():
        return
    logger.info("glm53_prefill %s", json.dumps(dict(kind=kind, rank=_rank, **fields), sort_keys=True))
    if _export and kind == "admission_reject":
        _get_metrics()[0].labels(fields["reason"]).inc()
    elif _export and kind == "batch":
        _get_metrics()[2].observe(fields["new_tokens"])


def reject(reason, **fields):
    if reason not in REASONS:
        raise ValueError(f"Unbounded admission reason: {reason}")
    record("admission_reject", reason=reason, **fields)


def _observe(stage, clock, seconds):
    if _export:
        _get_metrics()[1].labels(stage, clock).observe(seconds)


def poll():
    """Read completed event pairs only; never wait/synchronize the device."""
    while _pending and _pending[0][2].query():
        name, start, end = _pending.popleft()
        _observe(name, "cuda", start.elapsed_time(end) / 1000)


@contextlib.contextmanager
def stage(name, *, cuda=True):
    if not enabled():
        yield
        return
    if name not in STAGES:
        raise ValueError(f"Unbounded stage: {name}")
    import torch

    # Events/CPU timestamps must not be baked into a replayed graph.
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if capturing:
        yield
        return
    poll()
    pair = None
    if cuda and torch.cuda.is_available() and len(_pending) < _pending.maxlen:
        pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        pair[0].record()
    started = time.perf_counter()
    try:
        with torch.profiler.record_function("glm53.prefill." + name):
            yield
    finally:
        _observe(name, "cpu", time.perf_counter() - started)
        if pair is not None:
            pair[1].record()
            _pending.append((name, *pair))


def traced(name, *, cuda=True):
    def decorate(fn):
        if not enabled():
            return fn
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name, cuda=cuda):
                return fn(*args, **kwargs)
        return wrapped
    return decorate
