"""Read-only PP activity and DPA logical-forward observations.

Installed in each Python worker using an opt-in, one-module import hook. The
stock scheduler is never edited. One daemon thread on PP0/TP0 samples the union
of live request IDs in every microbatch. A failed observation is evidence loss,
not a server failure. DPA observes completed logical forwards through the same
import hook, without tensor access or added collectives. Queued, finished, and
retracted requests are excluded.
"""

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
import weakref

PREFIX = "GLM53_PP_ACTIVITY "
DPA_PREFIX = "GLM53_DPA_STEP "
DPA_SAMPLE_EVERY = 15
TARGET = "sglang.srt.managers.scheduler"


def dpa_sync_conditions(scheduler, skip_all_gather):
    """Fail closed outside the audited, collectively synchronized DPA loop."""
    args, ps = scheduler.server_args, scheduler.ps
    return {
        "pp1": args.pp_size == ps.pp_size == 1,
        "dpa": args.enable_dp_attention is True
        and args.dp_size == ps.dp_size
        and ps.dp_size in (2, 4, 8),
        "no_cp": args.attn_cp_size == args.dcp_size == 1
        and args.enable_prefill_cp is False,
        "no_dwdp": args.dwdp_size == 1,
        "no_elastic": args.elastic_ep_backend is None,
        "no_speculation": args.speculative_algorithm is None
        and scheduler.spec_algorithm.is_none(),
        "no_two_batch_overlap": args.enable_two_batch_overlap is False,
        "no_disaggregation": args.disaggregation_mode == "null",
        "moe_a2a_none": args.moe_a2a_backend == "none",
        "mlp_sync_required": scheduler.require_mlp_sync is True,
        "scheduler_all_gather_enabled": skip_all_gather is False,
    }


def dpa_snapshot(scheduler, batch, conditions):
    """The surviving requests of one completed logical forward, not a timer.

    DPA's existing metadata collective supplies idle batches to empty groups.
    Thus each group advances forward_iter together; overlapping CPU result
    processing preserves that number in ScheduleBatch.copy(). No new collective
    or tensor access is introduced here.
    """
    ps = scheduler.ps
    step = batch.forward_iter
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("Missing logical forward iteration")
    valid = bool(conditions) and all(value is True for value in conditions.values())
    queued = {req.rid for req in tuple(scheduler.waiting_queue)}
    requests = {}
    if valid:
        for req in tuple(batch.reqs):
            if (
                req.finished_reason is None
                and not req.is_retracted
                and req.rid not in queued
            ):
                if req.rid in requests:
                    raise ValueError("Duplicate request in logical batch")
                requests[req.rid] = len(req.output_ids)
    return {
        "schema_version": 1,
        "timestamp": time.time(),
        "forward_iter": step,
        "sample_every_steps": DPA_SAMPLE_EVERY,
        "dp_rank": ps.dp_rank,
        "dp_size": ps.dp_size,
        "pp_rank": ps.pp_rank,
        "attn_tp_rank": ps.attn_tp_rank,
        "attn_cp_rank": ps.attn_cp_rank,
        "valid": valid,
        "sync_conditions": conditions,
        "forward_mode": batch.forward_mode.name,
        "count": len(requests),
        "rids": sorted(requests),
        "output_lengths": requests,
        "scope": "surviving_requests_after_same_collective_forward",
    }


def _emit_dpa(scheduler, batch):
    if not (
        scheduler.ps.pp_rank
        == scheduler.ps.attn_tp_rank
        == scheduler.ps.attn_cp_rank
        == 0
    ):
        return
    step = getattr(batch, "forward_iter", None)
    if isinstance(step, int) and not isinstance(step, bool) and step % DPA_SAMPLE_EVERY:
        return
    try:
        # This module is already imported by the stock scheduler; the helper
        # reads the same environment override as the actual collective path.
        from sglang.srt.managers.scheduler_components.dp_attn import (
            should_skip_scheduler_all_gather,
        )

        conditions = dpa_sync_conditions(
            scheduler, should_skip_scheduler_all_gather(scheduler.ps.dp_size)
        )
        record = dpa_snapshot(scheduler, batch, conditions)
    except Exception as exc:
        record = {
            "schema_version": 1,
            "timestamp": time.time(),
            "valid": False,
            "forward_iter": step,
            "dp_rank": scheduler.ps.dp_rank,
            "error": type(exc).__name__,
        }
    try:
        os.write(1, (DPA_PREFIX + json.dumps(record, sort_keys=True) + "\n").encode())
    except OSError:
        pass


def snapshot(scheduler):
    if not hasattr(scheduler, "running_mbs") or not hasattr(scheduler, "mbs"):
        raise RuntimeError("PP loop has not initialized")
    queued = {req.rid for req in tuple(scheduler.waiting_queue)}

    def live(req):
        return (
            req is not None
            and req.finished_reason is None
            and not req.is_retracted
            and req.rid not in queued
        )

    requests = {}

    def remember(req):
        if live(req):
            old = requests.get(req.rid)
            if old is None or len(req.output_ids) > len(old.output_ids):
                requests[req.rid] = req
            return True
        return False

    def ids(batches):
        found = set()
        for batch in batches:
            if batch is not None:
                for req in tuple(batch.reqs):
                    if remember(req):
                        found.add(req.rid)
        return found

    running = ids(tuple(scheduler.running_mbs) + (scheduler.running_batch,))
    inflight = ids(tuple(scheduler.mbs))
    chunks = tuple(getattr(scheduler, "chunked_req_mbs", ())) + (scheduler.chunked_req,)
    chunked_ids = {req.rid for req in chunks if remember(req)}
    all_ids = running | inflight | chunked_ids
    return {
        "timestamp": time.time(),
        "rids": sorted(all_ids),
        "count": len(all_ids),
        "stage": 0,
        "tp": 0,
        "valid": True,
        "running_count": len(running),
        "inflight_count": len(inflight),
        "chunked_rids": sorted(chunked_ids),
        "queued_count": len(queued),
        "output_lengths": {
            rid: len(requests[rid].output_ids) for rid in sorted(all_ids)
        },
        "scope": "union_of_live_pp_microbatches",
    }


def _sample(reference):
    while True:
        time.sleep(1)
        scheduler = reference()
        if scheduler is None:
            return
        try:
            record = snapshot(scheduler)
        except Exception as exc:
            record = {
                "timestamp": time.time(),
                "stage": 0,
                "tp": 0,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            # One write prevents the observer from fragmenting its own JSON line.
            os.write(1, (PREFIX + json.dumps(record, sort_keys=True) + "\n").encode())
        except OSError:
            return
        finally:
            del scheduler


def wrap_scheduler(cls):
    if getattr(cls, "_glm53_activity_observer", False):
        return
    original = cls.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.ps.pp_size > 1 and self.ps.pp_rank == 0 and self.ps.tp_rank == 0:
            threading.Thread(
                target=_sample,
                args=(weakref.ref(self),),
                daemon=True,
                name="glm53-pp-observer",
            ).start()

    cls.__init__ = initialize
    if os.environ.get("GLM53_DPA_OBSERVER") == "1":
        original_result = cls.process_batch_result

        @functools.wraps(original_result)
        def process_result(self, batch, result):
            returned = original_result(self, batch, result)
            # Read after stock result handling so finished/retracted requests
            # and overlap's extra terminal decode are not counted as live.
            try:
                if self.ps.pp_size == 1 and self.ps.dp_size > 1:
                    _emit_dpa(self, batch)
            except Exception:
                # Observation must not affect the scheduler's return or errors.
                pass
            return returned

        cls.process_batch_result = process_result
    cls._glm53_activity_observer = True


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        wrap_scheduler(module.Scheduler)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if not any(
        os.environ.get(key) == "1"
        for key in ("GLM53_PP_OBSERVER", "GLM53_DPA_OBSERVER")
    ):
        return
    if TARGET in sys.modules:
        wrap_scheduler(sys.modules[TARGET].Scheduler)
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
