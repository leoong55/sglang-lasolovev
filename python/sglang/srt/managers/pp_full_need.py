"""Opt-in full-attention PP admission policy for the GLM-5.3 experiment.

Adapted from H200.zip scheduler proposals 4--7. No CUDA imports: the budget
can be checked on CPU. This is a conservative reservation, not a throughput
prediction. Prefixes must be locked while checking admission.
"""

import os


def enabled():
    return os.environ.get("SGLANG_PP_FULL_NEED", "0") == "1"


def unique_live(requests):
    return list({id(r): r for r in requests if r is not None and not r.finished()}.values())


def future_tokens(req, page_size):
    # kv_allocated_len includes this request's device prefix. A newly matched
    # prefix is usable only while its radix lock is held. Host hits still need
    # device allocation, so they must NOT be subtracted here.
    held = max(req.kv.kv_allocated_len, len(req.prefix_indices))
    target = len(req.origin_input_ids) + req.sampling_params.max_new_tokens
    remaining = max(target - held, 0)
    return ((remaining + page_size - 1) // page_size + 1) * page_size


class FullNeedBudget:
    def __init__(self, inflight, free_tokens, page_size, max_running, short_tokens=0):
        self.inflight = inflight
        self.free_tokens = free_tokens
        self.page_size = page_size
        self.max_running = max_running
        self.short_tokens = short_tokens

    def live(self, selected=()):
        return unique_live([*self.inflight(), *selected])

    def can_admit(self, candidate, selected):
        live = unique_live([*self.live(selected), candidate])
        return len(live) <= self.max_running and sum(
            future_tokens(r, self.page_size) for r in live
        ) <= self.free_tokens()

    def chunk_cap(self, candidate, selected, requested):
        # Do not charge the chunk's own entire future against its next step.
        # Other PP microbatches keep their full reservation, including any
        # unfinished prefill. Deduplication also covers mixed decode batches.
        debt = sum(
            future_tokens(r, self.page_size)
            for r in self.live(selected)
            if r is not candidate
        )
        cap = max(0, self.free_tokens() - debt - self.page_size)
        return min(requested, cap // self.page_size * self.page_size)

    def short_fits(self, tokens, remaining):
        return (
            self.short_tokens > 0
            and tokens <= self.short_tokens
            and tokens <= remaining + self.short_tokens
        )


class FitScan:
    """Bound scanning and how often a large queued request may be overtaken.

    After eight successful overtakes the queue drains until that request fits.
    The limit is in admissions, not scheduler iterations (which vary by PP).
    """

    def __init__(self, scan_limit=32, overtake_limit=8):
        self.scan_limit = scan_limit
        self.overtake_limit = overtake_limit
        self.skipped = []
        self.admissions = 0

    def skip(self, req):
        if (
            len(self.skipped) >= self.scan_limit
            or getattr(req, "_pp_fit_overtakes", 0) >= self.overtake_limit
        ):
            return False
        self.skipped.append(req)
        return True

    def admitted(self):
        self.admissions += 1
        for req in self.skipped:
            req._pp_fit_overtakes = getattr(req, "_pp_fit_overtakes", 0) + 1
