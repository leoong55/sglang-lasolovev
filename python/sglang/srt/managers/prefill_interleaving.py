"""Pure admission planning for one continuation plus complete waiting prefills.

This module never allocates KV, loads host cache or mutates a request. Actual
PrefillAdder admission remains authoritative, including a host-load miss.
"""
from dataclasses import dataclass
from typing import Any


def page_ceil(n, page):
    return (n + page - 1) // page * page


@dataclass(frozen=True)
class Candidate:
    request: Any
    work: int
    memory: int


@dataclass(frozen=True)
class Plan:
    limit: int | None
    selected: tuple
    target: int
    borrowed: bool = False


class PrefillInterleaver:
    def __init__(self):
        self.request = None
        self.debt = 0

    def reset(self):
        self.request = None
        self.debt = 0

    def plan(self, request, candidates, *, remaining, budget, page, slots,
             kv_budget, minimum=None, adaptive=True, shortest_first=False):
        if self.request is not request:
            self.request, self.debt = request, 0
        half = page_ceil((budget + 1) // 2, page)
        hard_min = minimum or page
        target = min(remaining, max(hard_min, half + self.debt)) if adaptive else min(
            remaining, minimum or (page if shortest_first else half))
        target = min(budget, page_ceil(target, page))
        if slots <= 0 or budget < hard_min + page:
            return Plan(None, (), half)
        reserved, memory = 0, 0
        selected = []
        floor = target
        borrowed = False
        for c in candidates[:128]:
            if len(selected) >= slots:
                break
            work = max(1, c.work)
            charge = page_ceil(work, page)
            if shortest_first and work >= remaining:
                break
            next_floor = floor
            borrow = False
            if reserved + charge > budget - floor:
                if adaptive and not self.debt and not selected and charge <= budget - hard_min:
                    next_floor = budget - charge
                    borrow = next_floor < target
                else:
                    if shortest_first:
                        break
                    continue
            next_reserved = reserved + charge
            continuation_charge = min(page_ceil(remaining, page), budget - next_reserved)
            # Include continuation's alignment reserve and the complete waiter's
            # KV reservation, not just this pass's compute tokens.
            if memory + c.memory + continuation_charge + page >= kv_budget:
                continue
            selected.append(c.request)
            reserved, memory = next_reserved, memory + c.memory
            floor, borrowed = next_floor, borrowed or borrow
        if not selected:
            return Plan(None, (), half)
        return Plan((budget - reserved) // page * page, tuple(selected), half, borrowed)

    def settle(self, request, *, actual_tokens, target, finished, contended):
        if finished or self.request is not request:
            self.reset()
        elif contended:
            self.debt = max(0, self.debt + target - actual_tokens)
        else:
            # No eligible competition: consume the full chunk and reset credit.
            self.debt = 0
