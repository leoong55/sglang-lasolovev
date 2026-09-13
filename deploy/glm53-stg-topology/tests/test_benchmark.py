import importlib.util
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SPEC = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkContractTests(unittest.TestCase):
    def command(self, kind):
        return benchmark.build_command(kind, "http://127.0.0.1:1234", "/model/weights",
                                       Path("/results/test"), "test-")

    def option(self, command, name):
        return command[command.index(name) + 1]

    def test_short_preserves_natural_eos_and_sampling_defaults(self):
        command = self.command("short")
        for forbidden in ("--ignore-eos", "--temperature", "--extra-body", "--num-warmups"):
            self.assertNotIn(forbidden, command)
        for key, value in {"--random-input-len": "1000", "--random-output-len": "1000",
                           "--num-prompts": "400", "--metric-percentiles": "50,90,95,99",
                           "--max-concurrency": "40", "--seed": "0"}.items():
            self.assertEqual(self.option(command, key), value)

    def test_long_exact_contract_and_identical_cold_warm_payload(self):
        cold, warm = self.command("long-cold"), self.command("long-warm")
        self.assertEqual(cold, warm)
        self.assertIn("--ignore-eos", cold)
        for key, value in {"--prefix-repetition-prefix-len": "60000",
                           "--prefix-repetition-suffix-len": "15000",
                           "--prefix-repetition-output-len": "1000",
                           "--prefix-repetition-num-prefixes": "20", "--num-prompts": "300",
                           "--temperature": "0.3", "--num-warmups": "1",
                           "--metric-percentiles": "50,95,99", "--request-rate": "inf"}.items():
            self.assertEqual(self.option(cold, key), value)
        self.assertEqual(json.loads(self.option(cold, "--extra-body")),
                         {"chat_template_kwargs": {"enable_thinking": True}})

    def test_identifier_hook_preserves_content_order_and_existing_ids(self):
        samples = [SimpleNamespace(prompt="prefix-suffix1", request_id=None, expected_output_len=1000),
                   SimpleNamespace(prompt="prefix-suffix2", request_id="existing", expected_output_len=1000)]
        before = [(item.prompt, item.expected_output_len) for item in samples]
        result = benchmark.assign_missing_request_ids(samples, "measured-")
        self.assertIs(result, samples)
        self.assertEqual([item.request_id for item in result], ["measured-0", "existing"])
        self.assertEqual([(item.prompt, item.expected_output_len) for item in result], before)

    def test_short_hook_restores_natural_eos_after_vllm_random_override(self):
        short = SimpleNamespace(dataset_name="random", ignore_eos=True, request_id_prefix="short-")
        benchmark.prepare_samples(short, [])
        self.assertFalse(short.ignore_eos)
        long = SimpleNamespace(dataset_name="prefix_repetition", ignore_eos=True, request_id_prefix="long-")
        benchmark.prepare_samples(long, [])
        self.assertTrue(long.ignore_eos)

    def sampling(self, long=True):
        body = {"max_completion_tokens": 1000, "stream": True,
                "stream_options": {"include_usage": True}}
        if long:
            body.update(ignore_eos=True, temperature=0.3, chat_template_kwargs={"enable_thinking": True})
        return body

    def test_actual_wire_sampling_detects_silent_short_eos_or_temperature_change(self):
        self.assertTrue(benchmark.sampling_matches(self.sampling(False), "short"))
        self.assertFalse(benchmark.sampling_matches(self.sampling(False) | {"ignore_eos": True}, "short"))
        self.assertFalse(benchmark.sampling_matches(self.sampling(False) | {"temperature": 0}, "short"))
        self.assertFalse(benchmark.sampling_matches({}, "long-cold"))

    def test_sse_split_utf8_usage_and_finish_can_share_an_event(self):
        observer = benchmark.ResponseEvidence()
        body = b":ping\r\n\r\n" + (
            "data: " + json.dumps({"id": "reply1", "choices": [{"index": 0,
                "delta": {"content": "тест"}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 1000, "prompt_tokens": 75012}},
                ensure_ascii=False) + "\r\n\r\ndata: [DONE]\n\n").encode()
        for octet in body:
            observer.feed(bytes([octet]))
        evidence = observer.result()
        self.assertEqual(evidence["finish_reasons"], {"0": "length"})
        self.assertEqual(evidence["completion_tokens"], 1000)
        self.assertEqual(evidence["cache_observability"], "not_observable")
        self.assertIsNone(evidence["cached_tokens"])
        self.assertTrue(evidence["sse_done"])
        self.assertEqual(evidence["evidence_errors"], [])

    def test_explicit_zero_cache_is_reported_and_not_missing(self):
        observer = benchmark.ResponseEvidence()
        observer.observe({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}})
        self.assertEqual(observer.result()["cache_observability"], "reported")
        self.assertEqual(observer.result()["cached_tokens"], 0)

    def valid_record(self, **changes):
        record = {"measured": True, "request_id": "test-0", "status": 200, "sse_done": True,
                  "finish_reasons": {"0": "length"}, "completion_tokens": 1000,
                  "sampling": self.sampling(),
                  "evidence_errors": [], "started_at": 1.0, "finished_at": 2.0}
        return record | changes

    def test_long_requires_exact_length_and_finish_and_evidence(self):
        self.assertTrue(benchmark.validate_requests([self.valid_record()], "long-cold", 1)["functional_valid"])
        for change in ({"completion_tokens": 999}, {"finish_reasons": {"0": "stop"}},
                       {"finish_reasons": {}}, {"status": 500}, {"sse_done": False},
                       {"evidence_errors": ["bad stream"]}, {"proxy_error": "disconnected"}):
            with self.subTest(change=change):
                result = benchmark.validate_requests([self.valid_record(**change)], "long-cold", 1)
                self.assertFalse(result["functional_valid"])

    def test_short_accepts_early_natural_stop(self):
        record = self.valid_record(completion_tokens=23, finish_reasons={"0": "stop"},
                                   sampling=self.sampling(False))
        self.assertTrue(benchmark.validate_requests([record], "short", 1)["functional_valid"])

    def test_warmups_are_excluded_and_duplicate_measurement_rejected(self):
        records = [self.valid_record(measured=False, request_id=None), self.valid_record()]
        self.assertTrue(benchmark.validate_requests(records, "long-warm", 1)["functional_valid"])
        records.append(self.valid_record())
        self.assertFalse(benchmark.validate_requests(records, "long-warm", 2)["functional_valid"])

    def test_dp_counts_only_unique_complete_groups(self):
        data = {"loads": [{"dp_rank": 0, "num_running_reqs": 21},
                          {"dp_rank": 1, "num_running_reqs": 19}]}
        self.assertEqual(benchmark.dp_activity(data, "dpa2")["running"], 40)
        self.assertFalse(benchmark.dp_activity(data, "dpa4")["observable"])
        data["loads"][1]["dp_rank"] = 0
        self.assertFalse(benchmark.dp_activity(data, "dpa2")["observable"])
        self.assertFalse(benchmark.dp_activity(data, "pp2")["observable"])

    def test_activity_excludes_warmup_and_never_infers_pp_concurrency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "telemetry.jsonl"
            for timestamp, count in ((1, 48), (3, 38), (4, 40), (6, 48)):
                benchmark.append_json(path, {"timestamp": timestamp,
                    "activity": {"observable": True, "running": count}})
            verdict = benchmark.activity_verdict(path, "dpa2", 2, 5)
            self.assertEqual(verdict["server_running_peak"], 40)
            self.assertEqual(verdict["server_activity_samples"], 2)
            self.assertTrue(verdict["server_c40_confirmed"])
            self.assertEqual(benchmark.activity_verdict(path, "dpa2", 2, 3)["capacity_status"], "C40_UNPROVEN")
            self.assertFalse(benchmark.activity_verdict(path, "pp2", 2, 5)["server_c40_confirmed"])

    def test_redacts_secrets_without_dropping_resolved_config(self):
        self.assertEqual(benchmark.redact({"api_key": "secret", "dp_size": 4,
            "internal_states": [{"admin_api_key": "secret2"}]}),
            {"api_key": "[REDACTED]", "dp_size": 4,
             "internal_states": [{"admin_api_key": "[REDACTED]"}]})


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp not installed; pure contract tests still run")
class RecorderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_preserves_body_headers_and_split_stream_bytes(self):
        from aiohttp import ClientSession, web
        received = {}
        stream = (
            'data: {"id":"reply","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}],'
            '"usage":{"prompt_tokens":75012,"completion_tokens":1000,"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
            'data: [DONE]\n\n'
        ).encode()

        async def upstream(request):
            received["body"] = await request.read()
            received["id"] = request.headers.get("x-request-id")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "x-request-id": "server-id"})
            await response.prepare(request)
            for offset in range(0, len(stream), 7):
                await response.write(stream[offset:offset + 7])
                await asyncio.sleep(0)
            await response.write_eof()
            return response

        origin = web.Application()
        origin.router.add_post("/v1/chat/completions", upstream)
        origin_runner = web.AppRunner(origin)
        await origin_runner.setup()
        origin_site = web.TCPSite(origin_runner, "127.0.0.1", 0)
        await origin_site.start()
        origin_url = f"http://127.0.0.1:{origin_site._server.sockets[0].getsockname()[1]}"
        try:
            async with ClientSession() as session:
                recorder = benchmark.Recorder(session, origin_url)
                proxy = web.Application()
                proxy.router.add_route("*", "/{tail:.*}", recorder.handle)
                proxy_runner = web.AppRunner(proxy)
                await proxy_runner.setup()
                proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
                await proxy_site.start()
                proxy_url = f"http://127.0.0.1:{proxy_site._server.sockets[0].getsockname()[1]}"
                try:
                    with tempfile.TemporaryDirectory() as folder:
                        recorder.begin(Path(folder), "test-")
                        body = b'{"model":"GLM-5.3","messages":[{"role":"user","content":"exact bytes"}],"stream":true}'
                        async with session.post(proxy_url + "/v1/chat/completions", data=body,
                            headers={"x-request-id": "test-0", "Content-Type": "application/json"}) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(response.headers["x-request-id"], "server-id")
                            self.assertEqual(await response.read(), stream)
                        await asyncio.sleep(0)
                        self.assertEqual(received, {"body": body, "id": "test-0"})
                        self.assertEqual(recorder.active, 0)
                        self.assertEqual(len(recorder.records), 1)
                        evidence = recorder.records[0]
                        self.assertTrue(evidence["measured"])
                        self.assertEqual(evidence["finish_reasons"], {"0": "length"})
                        self.assertEqual(evidence["completion_tokens"], 1000)
                        self.assertEqual(evidence["cached_tokens"], 0)
                        self.assertTrue(evidence["sse_done"])
                        self.assertEqual(evidence["evidence_errors"], [])
                finally:
                    await proxy_runner.cleanup()
        finally:
            await origin_runner.cleanup()


if __name__ == "__main__":
    unittest.main()
