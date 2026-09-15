"""CPU-only regression tests executing AST-extracted installed production methods.
GPU/model imports are intentionally not needed; transport and tokenizer are mocks.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import dataclasses
import importlib.util
import itertools
import logging
import os
import sys
import types
import unittest
from contextlib import aclosing, nullcontext
from pathlib import Path
from unittest.mock import Mock

remaining = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args, remaining = parser.parse_known_args()
    source_path = args.source
else:
    source_path = None
if source_path is None and os.environ.get("GLM53_CANCEL_TEST_INSTALLED") == "1":
    spec = importlib.util.find_spec("sglang")
    source_path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "srt/managers/tokenizer_manager.py"
    )
if source_path is None:
    kit_root = Path(__file__).resolve().parents[1]
    source_root = Path(os.environ.get("SGLANG_SOURCE_ROOT", kit_root.parents[1]))
    source_path = source_root / "python/sglang/srt/managers/tokenizer_manager.py"
    if not source_path.exists():
        source_path = (
            kit_root / "overlay/python/sglang/srt/managers/tokenizer_manager.py"
        )
source = source_path.read_text()
tree = ast.parse(source)
N = types.SimpleNamespace
counter = itertools.count()


class GenerateReqInput:
    def __init__(self, rid="r", children=None, parallel_sample_num=1):
        self.rid = rid
        self.is_single = children is None
        self.children = children
        self.batch_size = len(children) if children is not None else 1
        self.parallel_sample_num = parallel_sample_num
        self.max_thinking_tokens = None
        self.routed_dp_rank = None
        self.return_prompt_token_ids = False
        self.stream = False
        self.lora_path = None
        self.return_logprob = False

    def normalize_batch_and_arguments(self):
        pass

    def __getitem__(self, i):
        return self.children[i]

    def regenerate_rid(self):
        self.rid = f"generated-{next(counter)}"
        return self.rid


class TokenizedGenerateReqInput:
    def __init__(self, rid):
        self.rid = rid
        self.mm_inputs = None
        self.time_stats = Mock()
        self.input_ids = [1, 2]
        self.sampling_params = N(max_new_tokens=4096)

    def wrap_pickle_fields(self):
        pass


class AbortReq:
    def __init__(self, rid="", abort_all=False):
        self.rid = rid
        self.abort_all = abort_all
        self.abort_message = None
        self.finished_reason = None
        self.weight_versions = None


class AsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def wait_for(self, predicate):
        assert predicate()


method_names = {
    "generate_request",
    "_send_one_request",
    "_send_batch_request",
    "_discard_pending_req_states",
    "_cancel_or_discard_req_states",
    "abort_request",
    "_handle_batch_request",
    "_collect_batch_responses",
    "_stream_batch_responses",
    "_handle_abort_req",
}
manager_node = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenizerManager"
)
state_node = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReqState"
)
methods = [
    n
    for n in manager_node.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in method_names
]
if {n.name for n in methods} != method_names:
    raise SystemExit("Missing patched production methods")
extracted = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        state_node,
        ast.ClassDef(
            name="Manager", bases=[], keywords=[], body=methods, decorator_list=[]
        ),
    ],
    type_ignores=[],
)
ast.fix_missing_locations(extracted)
env = dict(
    globals(),
    logger=logging.getLogger("cancel-tests"),
    get_serving=lambda: N(tokenizer_worker_num=1),
    get_disagg=lambda: N(language_only=False),
    get_bool_env_var=lambda _: False,
    wrap_shm_features=lambda x: x,
    set_time_batch=lambda *a: None,
    BatchTokenizedGenerateReqInput=lambda **kw: N(**kw),
    BatchTokenizedEmbeddingReqInput=lambda **kw: N(**kw),
    is_health_check_generate_req=lambda _: False,
)
exec(compile(extracted, str(source_path), "exec"), env)
Manager, ReqState = env["Manager"], env["ReqState"]


def fixture():
    m = Manager()
    m.rid_to_state = {}
    m.enable_metrics = True
    m.metrics_collector = N(labels={}, observe_one_aborted_request=Mock())
    m._dispatch_to_scheduler = Mock()
    m.cuda_vmm_feature_transport = N(
        prepare_for_dispatch=Mock(return_value=[]), cancel_for_dispatch=Mock()
    )
    m.auto_create_handle_loop = Mock()
    m._set_default_priority = Mock()
    m.request_logger = N(log_received_request=Mock())
    m.tokenizer = None
    m.is_pause_cond = AsyncContext()
    m.is_pause = False
    m.model_update_lock = N(reader_lock=AsyncContext())
    m.config_value = lambda _: "default"

    def init(obj, request=None):
        for item in ([obj] if obj.is_single else obj.children):
            if item.rid in m.rid_to_state:
                raise ValueError("duplicate rid")
            m.rid_to_state[item.rid] = ReqState(
                [], False, asyncio.Event(), item, Mock()
            )

    m._init_req_state = init

    async def validate(obj):
        pass

    async def tokenize(obj):
        return TokenizedGenerateReqInput(obj.rid)

    async def wait(obj, request=None):
        yield {"meta_info": {"id": obj.rid}}
        await asyncio.Event().wait()

    m._validate_and_resolve_lora = validate
    m._tokenize_one_request = tokenize
    m._wait_one_response = wait
    m._should_use_batch_tokenization = lambda *a: False
    return m


def aborted(m):
    return [
        call.args[0].rid
        for call in m._dispatch_to_scheduler.call_args_list
        if isinstance(call.args[0], AbortReq)
    ]


def add_state(m, rid, dispatched=False, finished=False):
    obj = GenerateReqInput(rid)
    m._init_req_state(obj)
    state = m.rid_to_state[rid]
    state.dispatched_to_scheduler = dispatched
    state.finished = finished
    return obj, state


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults(self):
        m = fixture()
        _, s = add_state(m, "a")
        self.assertFalse(s.dispatched_to_scheduler)
        self.assertFalse(s.abort_requested)

    async def test_single_dispatch_and_transport_failure(self):
        m = fixture()
        _, s = add_state(m, "a")
        m._send_one_request(TokenizedGenerateReqInput("a"))
        self.assertTrue(s.dispatched_to_scheduler)
        _, t = add_state(m, "b")
        m._dispatch_to_scheduler.side_effect = RuntimeError("transport failed")
        with self.assertRaises(RuntimeError):
            m._send_one_request(TokenizedGenerateReqInput("b"))
        self.assertFalse(t.dispatched_to_scheduler)
        m.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once()

    async def test_batch_dispatch_and_failure(self):
        m = fixture()
        for rid in ["a", "b"]:
            add_state(m, rid)
        m._send_batch_request([TokenizedGenerateReqInput(r) for r in ["a", "b"]])
        self.assertTrue(all(s.dispatched_to_scheduler for s in m.rid_to_state.values()))
        m = fixture()
        for rid in ["a", "b"]:
            add_state(m, rid)
        m._dispatch_to_scheduler.side_effect = RuntimeError("transport failed")
        with self.assertRaises(RuntimeError):
            m._send_batch_request([TokenizedGenerateReqInput(r) for r in ["a", "b"]])
        self.assertFalse(
            any(s.dispatched_to_scheduler for s in m.rid_to_state.values())
        )

    async def test_validation_failure_no_abort(self):
        m = fixture()

        async def reject(obj):
            raise ValueError("over context")

        m._tokenize_one_request = reject
        with self.assertRaisesRegex(ValueError, "over context"):
            await anext(m.generate_request(GenerateReqInput()))
        self.assertEqual(m.rid_to_state, {})
        self.assertEqual(aborted(m), [])

    async def test_task_cancellation_after_dispatch(self):
        m = fixture()
        entered = asyncio.Event()

        async def waiting(obj, request=None):
            entered.set()
            await asyncio.Event().wait()
            yield {}

        m._wait_one_response = waiting
        gen = m.generate_request(GenerateReqInput())
        task = asyncio.create_task(anext(gen))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(aborted(m), ["r"])
        self.assertTrue(m.rid_to_state["r"].abort_requested)
        # Late background cleanup must neither lose state nor dispatch twice.
        m.abort_request("r")
        self.assertEqual(aborted(m), ["r"])
        m.metrics_collector.observe_one_aborted_request.assert_called_once()

    async def test_generator_close_after_first_chunk(self):
        m = fixture()
        gen = m.generate_request(GenerateReqInput())
        await anext(gen)
        await gen.aclose()
        self.assertEqual(aborted(m), ["r"])
        self.assertIn("r", m.rid_to_state)

    async def test_normal_completion(self):
        m = fixture()

        async def finished(obj, request=None):
            m.rid_to_state.pop(obj.rid)
            yield {"finished": True}

        m._wait_one_response = finished
        self.assertEqual(
            [r async for r in m.generate_request(GenerateReqInput())],
            [{"finished": True}],
        )
        self.assertEqual(aborted(m), [])
        self.assertEqual(m.rid_to_state, {})

    async def test_late_close_does_not_abort_reused_rid(self):
        m = fixture()
        gen = m.generate_request(GenerateReqInput())
        await anext(gen)
        m.rid_to_state.pop("r")
        _, replacement = add_state(m, "r", dispatched=True)
        await gen.aclose()
        self.assertEqual(aborted(m), [])
        self.assertIs(m.rid_to_state["r"], replacement)

    async def test_partial_batch_failure(self):
        m = fixture()

        async def tokenize(obj):
            if obj.rid == "b":
                raise ValueError("second request invalid")
            return TokenizedGenerateReqInput(obj.rid)

        m._tokenize_one_request = tokenize
        obj = GenerateReqInput(
            ["a", "b"], [GenerateReqInput("a"), GenerateReqInput("b")]
        )
        with self.assertRaisesRegex(ValueError, "second request invalid"):
            await anext(m.generate_request(obj))
        self.assertEqual(aborted(m), ["a"])
        self.assertEqual(set(m.rid_to_state), {"a"})

    async def test_parallel_sampling_prefix_cancellation(self):
        m = fixture()
        entered = asyncio.Event()

        async def wait(obj, request=None):
            entered.set()
            await asyncio.Event().wait()
            yield {}

        m._wait_one_response = wait
        obj = GenerateReqInput(["a"], [GenerateReqInput("a")], parallel_sample_num=2)
        task = asyncio.create_task(anext(m.generate_request(obj)))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(aborted(m)), 1)
        self.assertTrue(aborted(m)[0].startswith("generated-"))
        self.assertNotIn("a", m.rid_to_state)

    async def test_parallel_sampling_children_cancellation(self):
        m = fixture()
        entered = asyncio.Event()

        async def wait(obj, request=None):
            m.rid_to_state.pop(obj.rid)
            yield {}

        m._wait_one_response = wait

        async def collect(gens):
            entered.set()
            await asyncio.Event().wait()

        m._collect_batch_responses = collect
        obj = GenerateReqInput(["a"], [GenerateReqInput("a")], parallel_sample_num=2)
        task = asyncio.create_task(anext(m.generate_request(obj)))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(aborted(m)), 2)
        self.assertTrue(all(r.startswith("generated-") for r in aborted(m)))
        self.assertEqual(set(m.rid_to_state), set(aborted(m)))

    async def test_parallel_sampling_stream_close_cancels_children_now(self):
        m = fixture()
        children_started = []
        children_closed = []

        async def wait(obj, request=None):
            # Prefix warmup finishes before sampled children are dispatched.
            if len(m.rid_to_state) == 2 and "a" in m.rid_to_state:
                m.rid_to_state.pop(obj.rid)
                yield {}
                return
            children_started.append(obj.rid)
            try:
                yield {"meta_info": {"id": obj.rid}}
                await asyncio.Event().wait()
            finally:
                children_closed.append(obj.rid)

        m._wait_one_response = wait
        obj = GenerateReqInput(["a"], [GenerateReqInput("a")], parallel_sample_num=2)
        obj.stream = True
        gen = m.generate_request(obj)
        await anext(gen)
        await gen.aclose()
        self.assertEqual(len(aborted(m)), 2)
        self.assertCountEqual(children_closed, children_started)

    async def test_cancel_during_tokenization_does_not_send_abort(self):
        m = fixture()
        entered = asyncio.Event()

        async def tokenize(obj):
            entered.set()
            await asyncio.Event().wait()

        m._tokenize_one_request = tokenize
        task = asyncio.create_task(anext(m.generate_request(GenerateReqInput())))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(m.rid_to_state, {})
        self.assertEqual(aborted(m), [])

    async def test_failure_after_successful_dispatch_still_aborts(self):
        m = fixture()

        async def tokenize(obj):
            tokenized = TokenizedGenerateReqInput(obj.rid)
            tokenized.time_stats.set_api_server_dispatch_finish_time.side_effect = (
                RuntimeError("post-dispatch stats failed")
            )
            return tokenized

        m._tokenize_one_request = tokenize
        with self.assertRaisesRegex(RuntimeError, "post-dispatch stats failed"):
            await anext(m.generate_request(GenerateReqInput()))
        self.assertEqual(aborted(m), ["r"])
        self.assertTrue(m.rid_to_state["r"].dispatched_to_scheduler)

    async def test_abort_failure_retries_and_other_members_continue(self):
        m = fixture()
        for r in ["a", "b"]:
            add_state(m, r, dispatched=True)

        def dispatch(req):
            if req.rid == "a":
                raise RuntimeError("transient")

        m._dispatch_to_scheduler.side_effect = dispatch
        with self.assertLogs("cancel-tests", level="ERROR"):
            m._cancel_or_discard_req_states(["a", "b"])
        self.assertFalse(m.rid_to_state["a"].abort_requested)
        self.assertTrue(m.rid_to_state["b"].abort_requested)
        m._dispatch_to_scheduler.side_effect = None
        m.abort_request("a")
        self.assertTrue(m.rid_to_state["a"].abort_requested)

    async def test_finished_missing_empty_and_abort_all(self):
        m = fixture()
        add_state(m, "done", dispatched=True, finished=True)
        m._cancel_or_discard_req_states(["done", "missing"])
        m.abort_request("missing")
        with self.assertLogs("cancel-tests", level="WARNING"):
            m.abort_request("")
        self.assertEqual(aborted(m), [])
        m.abort_request(abort_all=True)
        self.assertTrue(m._dispatch_to_scheduler.call_args.args[0].abort_all)

    async def test_scheduler_echo_cleans_state_and_late_echo_is_safe(self):
        m = fixture()
        _, state = add_state(m, "a", dispatched=True)
        m._cancel_or_discard_req_states(["a"])
        m._handle_abort_req(AbortReq("a"))
        self.assertNotIn("a", m.rid_to_state)
        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertEqual(
            state.out_list[0]["meta_info"]["finish_reason"]["type"], "abort"
        )
        m._handle_abort_req(AbortReq("a"))
        self.assertEqual(aborted(m), ["a"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
