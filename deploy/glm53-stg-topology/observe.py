"""Read-only PP activity sampling; no tensor access or scheduler decisions.

Installed in each Python worker using an opt-in, one-module import hook. The
stock scheduler is never edited. One daemon thread on PP0/TP0 samples the union
of live request IDs in every microbatch. A failed observation is evidence loss,
not a server failure. Queued, finished, and retracted requests are excluded.
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
TARGET = "sglang.srt.managers.scheduler"


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
    if os.environ.get("GLM53_PP_OBSERVER") != "1":
        return
    if TARGET in sys.modules:
        wrap_scheduler(sys.modules[TARGET].Scheduler)
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
