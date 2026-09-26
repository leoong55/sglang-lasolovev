"""Execute actual admission methods without importing Linux/CUDA dependencies.

Only the surrounding allocator/cache services are faked. Method bodies are
compiled directly from the runtime source, not reimplemented in the test.
"""
import ast
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def method(file, cls, name, namespace):
    tree = ast.parse((ROOT / file).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    source = "from __future__ import annotations\n" + ast.unparse(fn)
    exec(compile(source, str(file), "exec"), namespace)
    return namespace[name]


class Result(Enum):
    CONTINUE = auto()
    OTHER = auto()
    NO_TOKEN = auto()


@dataclass
class Admission:
    prefix_len: int
    extend_len: int
    max_new_tokens: int
    is_chunked: bool


class AdmissionTest(unittest.TestCase):
    def test_host_miss_cannot_create_second_partial(self):
        fn = method("python/sglang/srt/managers/schedule_policy.py", "PrefillAdder", "_select_prefill_admission",
                    dict(AddReqResult=Result, _PrefillAdmission=Admission, CLIP_MAX_NEW_TOKENS=4096))
        adder = SimpleNamespace(rem_total_tokens=1_000_000, rem_chunk_tokens=8192,
            ceil_paged_tokens=lambda n: (n + 255) // 256 * 256, exact_chunk_fill=False,
            is_hybrid_swa=False, dllm_config=None, prefill_interleaving=True,
            new_chunked_req=None, _check_prefill_tile_budget=lambda n: None)
        req = SimpleNamespace(prefix_indices=[], full_untruncated_fill_ids=range(16384),
                              sampling_params=SimpleNamespace(max_new_tokens=128))
        args = dict(total_tokens=17000, swa_host_hit_length=0, truncation_align_size=None, has_chunked_req=True)
        self.assertFalse(fn(adder, req, host_hit_length=8192, **args).is_chunked)
        diag = SimpleNamespace(reject=Mock())
        with patch.dict("sys.modules", {"sglang.srt.observability.glm53_prefill": diag}):
            self.assertEqual(fn(adder, req, host_hit_length=0, **args), Result.OTHER)
        diag.reject.assert_called_once_with("partial_prefill")

    def test_stale_full_flag_reaches_current_capacity_check(self):
        fn = method("python/sglang/srt/managers/scheduler.py", "Scheduler", "_get_new_batch_prefill_raw", {})
        capacity = Mock(return_value=0)
        scheduler = SimpleNamespace(tp_rank=0, pp_rank=0, grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
            enable_priority_preemption=False, is_hybrid_swa=False, prefill_interleaving=True,
            waiting_queue=[object()], chunked_req=None, min_free_slots_delayer=None,
            get_num_allocatable_reqs=capacity)
        batch = SimpleNamespace(batch_is_full=True, reqs=[object()])
        diag = SimpleNamespace(configure=Mock(), enabled=lambda: False, reject=Mock())
        with patch.dict("sys.modules", {"sglang.srt.observability": SimpleNamespace(glm53_prefill=diag)}):
            result = fn(scheduler, prefill_delayer_single_pass=None, running_batch=batch)
        capacity.assert_called_once()
        self.assertEqual(result, (None, batch))
        diag.reject.assert_called_once_with("request_slots")


if __name__ == "__main__":
    unittest.main()
