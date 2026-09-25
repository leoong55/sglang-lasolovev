"""Exercise real chat generators and Starlette ASGI disconnects; no model/GPU.

Only request payloads, tokenizer and scheduler transport are fixtures. Both ASGI
disconnect protocols run through the actual response implementation.
"""

from __future__ import annotations

import ast
import asyncio
import json
import time
import unittest
from http import HTTPStatus
from types import SimpleNamespace as N

import anyio
import test_host_cancel_lifecycle as lifecycle
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

path = lifecycle.source_path.parent.parent / "entrypoints/openai/serving_chat.py"
tree = ast.parse(path.read_text())
response_node = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "_ClosingStreamingResponse"
)
chat_node = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingChat"
)
methods = [
    n
    for n in chat_node.body
    if isinstance(n, ast.AsyncFunctionDef)
    and n.name in {"_handle_streaming_request", "_generate_chat_stream"}
]
assert len(methods) == 2
extracted = ast.Module(
    body=[
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
        response_node,
        ast.ClassDef(
            name="Chat", bases=[], keywords=[], body=methods, decorator_list=[]
        ),
    ],
    type_ignores=[],
)
env = dict(
    globals(),
    should_include_usage=lambda *a: (False, False),
    build_sse_content=lambda **kw: "data: " + json.dumps(kw) + "\n\n",
)
exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), env)


def fixture(samples=1, finish=False):
    tm = lifecycle.fixture()
    tm.server_args = N(stream_response_default_include_usage=False)
    if finish:

        async def wait(obj, request=None):
            tm.rid_to_state.pop(obj.rid)
            yield {"meta_info": {"id": obj.rid}}

        tm._wait_one_response = wait
    elif samples > 1:

        async def wait(obj, request=None):
            if "r" in tm.rid_to_state and len(tm.rid_to_state) == 2:
                tm.rid_to_state.pop(obj.rid)
                yield {}
                return
            yield {"meta_info": {"id": obj.rid}}
            await asyncio.Event().wait()

        tm._wait_one_response = wait
    obj = lifecycle.GenerateReqInput("r")
    if samples > 1:
        obj = lifecycle.GenerateReqInput(["r"], [obj], parallel_sample_num=samples)
    obj.stream = True
    chat = env["Chat"]()
    chat.tokenizer_manager = tm
    chat._reported_prompt_tokens = lambda meta: 2
    chat.create_error_response = lambda msg: N(error=msg, status_code=400)

    async def content(**kwargs):
        yield 'data: {"content":"token"}\n\n'

    chat._generate_stream_content = content
    request_fields = {
        n.attr: False
        for method in methods
        for n in ast.walk(method)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "request"
    }
    request_fields.update(model="test", n=samples, stream_options=None)
    return tm, obj, chat, N(**request_fields)


class ChatDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_still_returns_http_400_before_stream(self):
        tm, obj, chat, request = fixture()

        async def reject(obj):
            raise ValueError("over context")

        tm._tokenize_one_request = reject
        response = await chat._handle_streaming_request(obj, request, None)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(tm.rid_to_state)
        self.assertEqual(lifecycle.aborted(tm), [])

    async def test_normal_stream_completion_does_not_abort(self):
        tm, obj, chat, request = fixture(finish=True)
        response = await chat._handle_streaming_request(obj, request, None)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        self.assertTrue(any("[DONE]" in str(m.get("body", "")) for m in messages))
        self.assertEqual(lifecycle.aborted(tm), [])

    async def test_asgi24_send_failure_closes_even_primed_unsent_body(self):
        for samples in (1, 2):
            for fail_on in ("headers", "body"):
                with self.subTest(samples=samples, fail_on=fail_on):
                    tm, obj, chat, request = fixture(samples)
                    response = await chat._handle_streaming_request(obj, request, None)

                    async def receive():
                        await asyncio.Event().wait()

                    async def send(message):
                        if (
                            fail_on == "headers"
                            or message["type"] == "http.response.body"
                        ):
                            raise OSError("peer disconnected")

                    with self.assertRaises(ClientDisconnect):
                        await response(
                            {"type": "http", "asgi": {"spec_version": "2.4"}},
                            receive,
                            send,
                        )
                    self.assertEqual(len(lifecycle.aborted(tm)), samples)
                    self.assertTrue(
                        all(s.abort_requested for s in tm.rid_to_state.values())
                    )

    async def test_asgi23_receive_disconnect_during_send_closes_all_samples(self):
        for samples in (1, 2):
            with self.subTest(samples=samples):
                tm, obj, chat, request = fixture(samples)
                response = await chat._handle_streaming_request(obj, request, None)
                disconnect = asyncio.Event()

                async def receive():
                    await disconnect.wait()
                    return {"type": "http.disconnect"}

                async def send(message):
                    if message["type"] == "http.response.body":
                        disconnect.set()
                        await asyncio.Event().wait()

                await asyncio.wait_for(
                    response(
                        {"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send
                    ),
                    timeout=2,
                )
                self.assertEqual(len(lifecycle.aborted(tm)), samples)

    async def test_task_cancelled_while_sending_closes_request(self):
        tm, obj, chat, request = fixture(samples=2)
        response = await chat._handle_streaming_request(obj, request, None)
        sending = asyncio.Event()

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                sending.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(
            response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        )
        await sending.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(lifecycle.aborted(tm)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
