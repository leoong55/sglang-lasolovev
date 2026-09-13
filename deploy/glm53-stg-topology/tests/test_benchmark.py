import asyncio
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).parents[1] / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkContractTests(unittest.TestCase):
    def test_long_smoke_tokenizes_concatenated_text_before_exact_roundtrip_slice(self):
        class BoundaryMergingTokenizer:
            def __init__(self, trim_last=False):
                self.parts, self.vocabulary = [], {}
                self.trim_last = trim_last

            def encode(self, text, add_special_tokens=False):
                result = []
                # Leading spaces merge into the following word, whereas a
                # trailing isolated space is its own token, as with BPE.
                for part in re.findall(r" ?[A-Za-z]+|[. ]", text):
                    if part not in self.vocabulary:
                        self.vocabulary[part] = len(self.parts)
                        self.parts.append(part)
                    result.append(self.vocabulary[part])
                return result

            def decode(self, ids):
                if self.trim_last:
                    ids = ids[:-1]
                return "".join(self.parts[index] for index in ids)

        tokenizer = BoundaryMergingTokenizer()
        fragment = tokenizer.encode(
            "The benchmark verifies long context model execution. "
        )
        old_text = tokenizer.decode((fragment * (75000 // len(fragment) + 1))[:75000])
        self.assertLess(len(tokenizer.encode(old_text)), 74000)
        for tokenizer in (tokenizer, BoundaryMergingTokenizer(trim_last=True)):
            with (
                self.subTest(trim_last=tokenizer.trim_last),
                patch.dict(
                    sys.modules,
                    {
                        "transformers": SimpleNamespace(
                            AutoTokenizer=SimpleNamespace(
                                from_pretrained=lambda *args, **kwargs: tokenizer
                            )
                        )
                    },
                ),
            ):
                text, metadata = benchmark.build_long_smoke_input("/model")
            self.assertEqual(len(tokenizer.encode(text)), 75000)
            self.assertEqual(metadata["input_tokens_before_chat_template"], 75000)
            self.assertEqual(
                metadata["sha256"], hashlib.sha256(text.encode()).hexdigest()
            )
            self.assertEqual(
                metadata["roundtrip_adjustment_attempts"],
                2 if tokenizer.trim_last else 1,
            )

    def test_long_smoke_rejects_unrecoverable_roundtrip_before_returning_input(self):
        class BrokenTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1] if text == "broken" else [1] * max(1, len(text))

            def decode(self, ids):
                return "broken"

        with patch.dict(
            sys.modules,
            {
                "transformers": SimpleNamespace(
                    AutoTokenizer=SimpleNamespace(
                        from_pretrained=lambda *args, **kwargs: BrokenTokenizer()
                    )
                )
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "required 74000..76000"):
                benchmark.build_long_smoke_input("/model")

    def command(self, kind):
        return benchmark.build_command(
            kind,
            "http://127.0.0.1:1234",
            "/model/weights",
            Path("/results/test"),
            "test-",
        )

    def option(self, command, name):
        return command[command.index(name) + 1]

    def test_short_preserves_natural_eos_and_sampling_defaults(self):
        command = self.command("short")
        for forbidden in (
            "--ignore-eos",
            "--temperature",
            "--extra-body",
            "--num-warmups",
        ):
            self.assertNotIn(forbidden, command)
        for key, value in {
            "--random-input-len": "1000",
            "--random-output-len": "1000",
            "--num-prompts": "400",
            "--metric-percentiles": "50,90,95,99",
            "--max-concurrency": "40",
            "--seed": "0",
        }.items():
            self.assertEqual(self.option(command, key), value)

    def test_long_exact_contract_and_identical_cold_warm_payload(self):
        cold, warm = self.command("long-cold"), self.command("long-warm")
        self.assertEqual(cold, warm)
        self.assertIn("--ignore-eos", cold)
        for key, value in {
            "--prefix-repetition-prefix-len": "60000",
            "--prefix-repetition-suffix-len": "15000",
            "--prefix-repetition-output-len": "1000",
            "--prefix-repetition-num-prefixes": "20",
            "--num-prompts": "300",
            "--temperature": "0.3",
            "--num-warmups": "1",
            "--metric-percentiles": "50,95,99",
            "--request-rate": "inf",
        }.items():
            self.assertEqual(self.option(cold, key), value)
        self.assertEqual(
            json.loads(self.option(cold, "--extra-body")),
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_identifier_hook_preserves_content_order_and_existing_ids(self):
        samples = [
            SimpleNamespace(
                prompt="prefix-suffix1", request_id=None, expected_output_len=1000
            ),
            SimpleNamespace(
                prompt="prefix-suffix2", request_id="existing", expected_output_len=1000
            ),
        ]
        before = [(item.prompt, item.expected_output_len) for item in samples]
        result = benchmark.assign_missing_request_ids(samples, "measured-")
        self.assertIs(result, samples)
        self.assertEqual(
            [item.request_id for item in result], ["measured-0", "existing"]
        )
        self.assertEqual(
            [(item.prompt, item.expected_output_len) for item in result], before
        )

    def test_short_hook_restores_natural_eos_after_vllm_random_override(self):
        short = SimpleNamespace(
            dataset_name="random", ignore_eos=True, request_id_prefix="short-"
        )
        benchmark.prepare_samples(short, [])
        self.assertFalse(short.ignore_eos)
        long = SimpleNamespace(
            dataset_name="prefix_repetition", ignore_eos=True, request_id_prefix="long-"
        )
        benchmark.prepare_samples(long, [])
        self.assertTrue(long.ignore_eos)

    def sampling(self, long=True):
        body = {
            "max_completion_tokens": 1000,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if long:
            body.update(
                ignore_eos=True,
                temperature=0.3,
                chat_template_kwargs={"enable_thinking": True},
            )
        return body

    def test_actual_wire_sampling_detects_silent_short_eos_or_temperature_change(self):
        self.assertTrue(benchmark.sampling_matches(self.sampling(False), "short"))
        self.assertFalse(
            benchmark.sampling_matches(
                self.sampling(False) | {"ignore_eos": True}, "short"
            )
        )
        self.assertFalse(
            benchmark.sampling_matches(
                self.sampling(False) | {"temperature": 0}, "short"
            )
        )
        self.assertFalse(benchmark.sampling_matches({}, "long-cold"))

    def test_sse_split_utf8_usage_and_finish_can_share_an_event(self):
        observer = benchmark.ResponseEvidence()
        body = (
            b":ping\r\n\r\n"
            + (
                "data: "
                + json.dumps(
                    {
                        "id": "reply1",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "тест"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {"completion_tokens": 1000, "prompt_tokens": 75012},
                    },
                    ensure_ascii=False,
                )
                + "\r\n\r\ndata: [DONE]\n\n"
            ).encode()
        )
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
        record = {
            "measured": True,
            "request_id": "test-0",
            "status": 200,
            "sse_done": True,
            "finish_reasons": {"0": "length"},
            "completion_tokens": 1000,
            "sampling": self.sampling(),
            "evidence_errors": [],
            "started_at": 1.0,
            "finished_at": 2.0,
        }
        return record | changes

    def test_long_requires_exact_length_and_finish_and_evidence(self):
        self.assertTrue(
            benchmark.validate_requests([self.valid_record()], "long-cold", 1)[
                "functional_valid"
            ]
        )
        for change in (
            {"completion_tokens": 999},
            {"finish_reasons": {"0": "stop"}},
            {"finish_reasons": {}},
            {"status": 500},
            {"sse_done": False},
            {"evidence_errors": ["bad stream"]},
            {"proxy_error": "disconnected"},
        ):
            with self.subTest(change=change):
                result = benchmark.validate_requests(
                    [self.valid_record(**change)], "long-cold", 1
                )
                self.assertFalse(result["functional_valid"])

    def test_short_accepts_early_natural_stop(self):
        record = self.valid_record(
            completion_tokens=23,
            finish_reasons={"0": "stop"},
            sampling=self.sampling(False),
        )
        self.assertTrue(
            benchmark.validate_requests([record], "short", 1)["functional_valid"]
        )

    def test_warmups_are_excluded_and_duplicate_measurement_rejected(self):
        records = [
            self.valid_record(measured=False, request_id=None),
            self.valid_record(),
        ]
        self.assertTrue(
            benchmark.validate_requests(records, "long-warm", 1)["functional_valid"]
        )
        records.append(self.valid_record())
        self.assertFalse(
            benchmark.validate_requests(records, "long-warm", 2)["functional_valid"]
        )

    def test_dp_counts_only_unique_complete_groups(self):
        data = {
            "loads": [
                {"dp_rank": 0, "num_running_reqs": 21},
                {"dp_rank": 1, "num_running_reqs": 19},
            ]
        }
        diagnostic = benchmark.dp_activity(data, "dpa2")
        self.assertEqual(diagnostic["diagnostic_running_sum"], 40)
        self.assertEqual(diagnostic["qualification"], "diagnostic_only")
        self.assertNotIn("running", diagnostic)
        self.assertFalse(benchmark.dp_activity(data, "dpa4")["observable"])
        data["loads"][1]["dp_rank"] = 0
        self.assertFalse(benchmark.dp_activity(data, "dpa2")["observable"])
        self.assertFalse(benchmark.dp_activity(data, "pp2")["observable"])

    def test_activity_excludes_warmup_and_never_confirms_raw_concurrency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "telemetry.jsonl"
            for timestamp, count in ((1, 48), (3, 38), (4, 40), (6, 48)):
                benchmark.append_json(
                    path,
                    {
                        "timestamp": timestamp,
                        "activity": {
                            "observable": True,
                            "diagnostic_running_sum": count,
                        },
                    },
                )
            verdict = benchmark.activity_verdict(path, "dpa2", 2, 5)
            self.assertEqual(
                verdict["dp_load_diagnostics"]["peak_independent_slot_sum"], 40
            )
            self.assertEqual(verdict["dp_load_diagnostics"]["samples"], 2)
            self.assertEqual(
                verdict["dp_load_diagnostics"]["samples_slot_sum_at_least_40"], 1
            )
            self.assertFalse(verdict["server_c40_confirmed"])
            self.assertEqual(verdict["capacity_status"], "C40_UNPROVEN")
            self.assertNotIn("server_running_peak", verdict)
            self.assertEqual(
                benchmark.activity_verdict(path, "dpa2", 2, 3)["capacity_status"],
                "C40_UNPROVEN",
            )
            self.assertFalse(
                benchmark.activity_verdict(path, "pp2", 2, 5)["server_c40_confirmed"]
            )

    def test_skewed_and_legacy_high_slot_sums_are_diagnostics_for_every_dpa_profile(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "telemetry.jsonl"
            for profile in ("dpa2", "dpa4", "dpa8"):
                size = int(profile[3:])
                loads = [
                    {
                        "dp_rank": rank,
                        "timestamp": 10 + rank,
                        "num_running_reqs": 40 // size,
                    }
                    for rank in range(size)
                ]
                activity = benchmark.dp_activity(loads, profile)
                path.write_text("")
                benchmark.append_json(
                    path, {"timestamp": 18, "loads": loads, "activity": activity}
                )
                # Old telemetry's unqualified 'running' field must not resurrect PASS.
                benchmark.append_json(
                    path,
                    {
                        "timestamp": 19,
                        "activity": {"observable": True, "running": 1000},
                    },
                )
                verdict = benchmark.activity_verdict(path, profile, 0, 20)
                self.assertEqual(
                    verdict["dp_load_diagnostics"]["peak_independent_slot_sum"], 1000
                )
                self.assertFalse(verdict["server_c40_confirmed"])
                self.assertEqual(verdict["capacity_status"], "C40_UNPROVEN")
            pp = benchmark.activity_verdict(path, "pp2", 0, 20)
            self.assertEqual(pp["dp_load_diagnostics"]["samples"], 0)
            self.assertFalse(pp["server_c40_confirmed"])
            self.assertEqual(pp["capacity_status"], "C40_UNPROVEN")

    def test_redacts_secrets_without_dropping_resolved_config(self):
        self.assertEqual(
            benchmark.redact(
                {
                    "api_key": "secret",
                    "dp_size": 4,
                    "internal_states": [{"admin_api_key": "secret2"}],
                }
            ),
            {
                "api_key": "[REDACTED]",
                "dp_size": 4,
                "internal_states": [{"admin_api_key": "[REDACTED]"}],
            },
        )

    def test_host_cache_counters_survive_telemetry_filter(self):
        counters = [
            'sglang:prefill_effective_tokens_total{dp_rank="0",source="host"} 60000',
            'sglang:load_back_tokens_total{dp_rank="0"} 60000',
        ]
        text = "\n".join(counters + ['unrelated:large_bucket{le="1000"} 99'])
        self.assertEqual(benchmark.select_metrics(text), counters)

    def test_suite_catalog_is_saved_offline_with_exact_file_hash_and_no_payload_log(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                mode="suite", results_dir=Path(folder), tokenizer="/model/weights"
            )
            payload = {
                "schema_version": 1,
                "samples": [{"sample_index": 0, "request_body_sha256": "a" * 64}],
            }

            def build(path):
                self.assertEqual(path, Path("/model/weights"))
                print("suppressed tokenizer or catalog details")
                return payload

            builder = Mock(side_effect=build)
            output = io.StringIO()
            with (
                patch.dict(
                    sys.modules, {"dataset_catalog": SimpleNamespace(build=builder)}
                ),
                patch.dict(benchmark.os.environ, {}, clear=False),
                patch("sys.stdout", output),
            ):
                provenance = benchmark.prepare_dataset_catalog(args)
                self.assertEqual(benchmark.os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(benchmark.os.environ["TRANSFORMERS_OFFLINE"], "1")
            builder.assert_called_once_with(Path("/model/weights"))
            path = Path(folder) / "dataset-catalog.json"
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(
                provenance["dataset_catalog_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertNotIn("sample_index", output.getvalue())
            self.assertNotIn("suppressed", output.getvalue())
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                benchmark.prepare_dataset_catalog(args)

    def test_smoke_skips_offline_dataset_generation(self):
        builder = Mock(
            side_effect=AssertionError("Smoke must not generate the dataset")
        )
        with patch.dict(
            sys.modules, {"dataset_catalog": SimpleNamespace(build=builder)}
        ):
            self.assertEqual(
                benchmark.prepare_dataset_catalog(SimpleNamespace(mode="smoke")), {}
            )
        builder.assert_not_called()

    def test_compile_inventory_records_only_artifact_metadata_without_following_links(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for backend in benchmark.COMPILE_BACKENDS:
                path = root / backend / "kernel.bin"
                path.parent.mkdir()
                path.write_bytes(b"PRIVATE KERNEL SOURCE")
            for relative in (
                "triton/work.lock",
                "inductor/tmp/intermediate",
                "cuda/.hidden",
                "hf/token",
                "xdg/config",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("PRIVATE")
            (root / "triton" / "linked-file").symlink_to(root / "hf" / "token")
            (root / "triton" / "linked-directory").symlink_to(
                root / "hf", target_is_directory=True
            )
            output = benchmark.compile_cache_inventory(str(root))
            self.assertTrue(output["observable"])
            self.assertEqual(output["status"], "observed")
            self.assertEqual(
                {row["path"] for row in output["files"]},
                {name + "/kernel.bin" for name in benchmark.COMPILE_BACKENDS},
            )
            self.assertTrue(
                all(
                    row["size"] == 21 and row["mtime_ns"] > 0 for row in output["files"]
                )
            )
            self.assertEqual(output["errors"], [])
            self.assertNotIn(folder, json.dumps(output))
            self.assertNotIn("PRIVATE", json.dumps(output))
        self.assertFalse(benchmark.compile_cache_inventory(None)["observable"])
        self.assertFalse(benchmark.compile_cache_inventory(folder)["observable"])

    def test_compile_inventory_marks_unreadable_listing_as_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "triton").mkdir()

            def unreadable(path, *, followlinks, onerror):
                onerror(PermissionError("private host/path details"))
                return iter(())

            with patch.object(benchmark.os, "walk", side_effect=unreadable):
                output = benchmark.compile_cache_inventory(folder)
            self.assertFalse(output["observable"])
            self.assertEqual(output["status"], "partial")
            self.assertEqual(output["errors"], [{"type": "PermissionError"}])
            self.assertNotIn("private host", json.dumps(output))


class SmokeReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_provenance_purpose_defaults_keep_unknown_tooling_revision_explicit(
        self,
    ):
        for mode, purpose in (("smoke", "admission"), ("suite", "measurement")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                args = SimpleNamespace(
                    mode=mode,
                    results_dir=Path(folder),
                    tokenizer="/model",
                    profile="pp2",
                    source_commit="a" * 40,
                    image_digest="sha256:test",
                    base_url="http://unused",
                    workload_timeout=60,
                )
                with (
                    patch.dict(benchmark.os.environ, {}, clear=True),
                    patch.object(benchmark, "prepare_dataset_catalog", return_value={}),
                    patch.object(
                        benchmark.importlib.metadata, "version", return_value="0.23.0"
                    ),
                    patch("aiohttp.TCPConnector", return_value=object()),
                    patch(
                        "aiohttp.ClientSession",
                        side_effect=RuntimeError("session boundary"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "session boundary"):
                        await benchmark.run(args)
                provenance = json.loads((Path(folder) / "provenance.json").read_text())
                self.assertEqual(provenance["purpose"], purpose)
                self.assertIsNone(provenance["tooling_commit"])
                self.assertIsNone(provenance["configmap_sha256"])

    async def test_run_catalog_own_event_loop_finishes_before_http_session(self):
        class ReachedHttpSession(Exception):
            pass

        async def capture():
            return {"schema_version": 1, "samples": []}

        def build(path):
            return asyncio.run(capture())

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                mode="suite",
                results_dir=Path(folder),
                tokenizer="/model",
                profile="pp2",
                source_commit="a" * 40,
                image_digest="sha256:test",
                base_url="http://internal:8080",
                workload_timeout=60,
            )
            with (
                patch.dict(
                    sys.modules, {"dataset_catalog": SimpleNamespace(build=build)}
                ),
                patch.object(
                    benchmark.importlib.metadata, "version", return_value="0.23.0"
                ),
                patch("aiohttp.TCPConnector", return_value=object()),
                patch("aiohttp.ClientSession", side_effect=ReachedHttpSession),
                patch.dict(
                    benchmark.os.environ,
                    {
                        "TOOLING_COMMIT": "b" * 40,
                        "SCRIPT_CONFIG_SHA256": "c" * 64,
                        "BENCHMARK_PURPOSE": "preparation",
                    },
                ),
            ):
                with self.assertRaises(ReachedHttpSession):
                    await benchmark.run(args)
            catalog = Path(folder) / "dataset-catalog.json"
            provenance = json.loads((Path(folder) / "provenance.json").read_text())
            self.assertEqual(
                provenance["dataset_catalog_sha256"],
                hashlib.sha256(catalog.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["source_commit"], "a" * 40)
            self.assertEqual(provenance["tooling_commit"], "b" * 40)
            self.assertEqual(provenance["configmap_sha256"], "c" * 64)
            self.assertEqual(provenance["purpose"], "preparation")
            self.assertEqual(
                provenance["benchmark_script_sha256"],
                hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
            )

    async def check_smoke(self, response_mode="separated"):
        calls = []

        class Response:
            def __init__(self, payload):
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def json(self):
                return self.payload

        class Session:
            def post(self, url, json, headers):
                if headers != {"Connection": "close"}:
                    raise AssertionError("Smoke probes must use their own connections")
                calls.append(json)
                if len(calls) == 1:
                    nonce = json["messages"][0]["content"].removeprefix(
                        "Reply with exactly "
                    )
                    # Model behavior: the checkpoint always emits reasoning.
                    # Only an enabled parser separates it from final content.
                    separated = json["chat_template_kwargs"]["enable_thinking"]
                    content = nonce if separated else f"<think>reasoning</think>{nonce}"
                    reason = "stop"
                    # The real checkpoint can use 128 tokens before its EOS.
                    if json["max_completion_tokens"] < 256:
                        reason = "length"
                    if response_mode == "raw_tags":
                        content = f"<think>reasoning</think>{nonce}"
                    elif response_mode == "wrong_nonce":
                        content = nonce + " extra"
                    elif response_mode == "reasoning_only":
                        content = ""
                    elif response_mode == "length":
                        reason = "length"
                    return Response(
                        {
                            "choices": [
                                {
                                    "finish_reason": reason,
                                    "message": {
                                        "content": content,
                                        "reasoning_content": "Reasoning kept separately.",
                                    },
                                }
                            ]
                        }
                    )
                return Response(
                    {
                        "choices": [{"finish_reason": "length"}],
                        "usage": {"completion_tokens": 16, "prompt_tokens": 75012},
                    }
                )

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1] * (
                    75000
                    if text == "long smoke input"
                    else text.count(
                        "The benchmark verifies long context model execution. "
                    )
                )

            def decode(self, ids):
                if len(ids) != 75000:
                    raise AssertionError("Long smoke must retain its 75k input")
                return "long smoke input"

        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                results_dir=Path(folder),
                base_url="http://internal:8080",
                tokenizer="/model",
            )
            with (
                patch.dict(sys.modules, {"transformers": transformers}),
                patch.object(benchmark, "snapshot", new_callable=AsyncMock),
                patch.object(benchmark, "flush_cache", new_callable=AsyncMock) as flush,
            ):
                if response_mode == "separated":
                    await benchmark.smoke(Session(), args)
                    self.assertEqual(len(calls), 2)
                    self.assertIs(
                        calls[0]["chat_template_kwargs"]["enable_thinking"], True
                    )
                    saved = json.loads((Path(folder) / "smoke/short.json").read_text())
                    self.assertEqual(
                        saved["choices"][0]["message"]["reasoning_content"],
                        "Reasoning kept separately.",
                    )
                    flush.assert_awaited_once()
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "exact-nonce/finish_reason=stop"
                    ):
                        await benchmark.smoke(Session(), args)
                    self.assertEqual(len(calls), 1)
                    flush.assert_not_awaited()

    async def test_smoke_accepts_separate_reasoning_with_exact_nonce_and_stop(self):
        await self.check_smoke()

    async def test_smoke_rejects_unparsed_tags_wrong_content_or_nonstop_finish(self):
        for mode in ("raw_tags", "wrong_nonce", "reasoning_only", "length"):
            with self.subTest(mode=mode):
                await self.check_smoke(mode)

    async def test_compile_inventory_snapshots_are_outside_measured_subprocess(self):
        events = []

        class Process:
            async def wait(self):
                events.append("child-finished")
                return 0

        async def launch(*args, **kwargs):
            events.append("child-started")
            return Process()

        async def inventory(directory, label):
            events.append(label + "-inventory")

        async def telemetry(session, args, directory, stop):
            await stop.wait()

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                results_dir=Path(folder),
                profile="pp2",
                tokenizer="/model",
                base_url="http://unused",
                workload_timeout=60,
            )
            recorder = benchmark.Recorder(None, args.base_url)
            valid = {
                "functional_valid": True,
                "started_at": 1,
                "finished_at": 2,
                "expected_requests": 400,
                "errors": [],
            }

            async def launch_with_result(*argv, **kwargs):
                benchmark.write_json(
                    Path(folder) / "r01-short/vllm.json",
                    {"completed": 400, "failed": 0},
                )
                return await launch(*argv, **kwargs)

            with (
                patch.object(benchmark, "flush_cache", new_callable=AsyncMock),
                patch.object(benchmark, "snapshot", new_callable=AsyncMock),
                patch.object(
                    benchmark, "snapshot_compile_cache", side_effect=inventory
                ),
                patch.object(benchmark, "collect_telemetry", side_effect=telemetry),
                patch.object(
                    benchmark.asyncio,
                    "create_subprocess_exec",
                    side_effect=launch_with_result,
                ),
                patch.object(benchmark, "validate_requests", return_value=valid),
            ):
                await benchmark.run_workload(
                    None, args, recorder, "http://local", "short", 1
                )
            self.assertEqual(
                events,
                [
                    "before-inventory",
                    "child-started",
                    "child-finished",
                    "after-inventory",
                ],
            )


@unittest.skipUnless(
    importlib.util.find_spec("aiohttp"),
    "aiohttp not installed; pure contract tests still run",
)
class RecorderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_long_request_survives_slow_builder_and_short_keepalive(self):
        from aiohttp import web

        loop = asyncio.get_running_loop()
        loop_resumed = threading.Event()
        main_thread = threading.get_ident()
        builder_threads, transports, requests, closed_while_building = [], [], [], []

        async def upstream(request):
            body = await request.json()
            requests.append(body)
            transports.append(request.transport)
            self.assertEqual(request.headers.get("Connection"), "close")
            if len(requests) == 1:
                nonce = body["messages"][0]["content"].removeprefix(
                    "Reply with exactly "
                )
                return web.json_response(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": nonce,
                                    "reasoning_content": "separate reasoning",
                                },
                            }
                        ]
                    }
                )
            self.assertEqual(body["messages"][0]["content"], "long smoke input")
            return web.json_response(
                {
                    "choices": [{"finish_reason": "length"}],
                    "usage": {"completion_tokens": 16, "prompt_tokens": 75012},
                }
            )

        class SlowTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1] * (
                    75000
                    if text == "long smoke input"
                    else text.count(
                        "The benchmark verifies long context model execution. "
                    )
                )

            def decode(self, ids):
                if len(ids) != 75000:
                    raise AssertionError("Expected the complete 75k smoke input")
                return "long smoke input"

        def after_keepalive():
            closed_while_building.append(transports[0].is_closing())
            loop_resumed.set()

        def load_tokenizer(*args, **kwargs):
            builder_threads.append(threading.get_ident())
            # Four times the server keepalive. This callback cannot execute if
            # synchronous tokenizer work is still blocking the HTTP event loop.
            loop.call_soon_threadsafe(lambda: loop.call_later(0.08, after_keepalive))
            if not loop_resumed.wait(timeout=1):
                raise AssertionError("Tokenizer build blocked connection cleanup")
            return SlowTokenizer()

        origin = web.Application()
        origin.router.add_post("/v1/chat/completions", upstream)
        runner = web.AppRunner(origin, keepalive_timeout=0.02)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                args = SimpleNamespace(
                    mode="smoke",
                    results_dir=Path(folder),
                    tokenizer="/model",
                    profile="pp2",
                    source_commit="a" * 40,
                    image_digest="sha256:test",
                    base_url=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
                    workload_timeout=5,
                )
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "transformers": SimpleNamespace(
                                AutoTokenizer=SimpleNamespace(
                                    from_pretrained=load_tokenizer
                                )
                            )
                        },
                    ),
                    patch.object(
                        benchmark.importlib.metadata, "version", return_value="0.23.0"
                    ),
                    patch.object(benchmark, "snapshot", new_callable=AsyncMock),
                    patch.object(benchmark, "flush_cache", new_callable=AsyncMock),
                ):
                    self.assertEqual(await benchmark.run(args), 0)
                metadata = json.loads(
                    (Path(folder) / "smoke/long-input.json").read_text()
                )
                self.assertEqual(metadata["input_tokens_before_chat_template"], 75000)
                self.assertEqual(
                    metadata["sha256"], hashlib.sha256(b"long smoke input").hexdigest()
                )
            self.assertEqual(len(requests), 2)
            self.assertEqual(
                [body["max_completion_tokens"] for body in requests], [2048, 16]
            )
            self.assertIsNot(transports[0], transports[1])
            self.assertEqual(closed_while_building, [True])
            self.assertTrue(
                builder_threads
                and all(value != main_thread for value in builder_threads)
            )
        finally:
            await runner.cleanup()

    async def test_runtime_session_reads_metrics_without_decoding_compressed_bytes(
        self,
    ):
        import gzip

        from aiohttp import web

        metrics = b"sglang:num_running_reqs 0\n"

        async def upstream(request):
            if "gzip" in request.headers.get("Accept-Encoding", ""):
                return web.Response(
                    body=gzip.compress(metrics), headers={"Content-Encoding": "gzip"}
                )
            return web.Response(body=metrics)

        origin = web.Application()
        origin.router.add_get("/metrics", upstream)
        runner = web.AppRunner(origin)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                args = SimpleNamespace(
                    mode="smoke",
                    results_dir=Path(folder),
                    tokenizer="/model",
                    profile="pp2",
                    source_commit="a" * 40,
                    image_digest="sha256:test",
                    base_url=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
                    workload_timeout=60,
                )

                async def probe(session, args):
                    async with session.get(args.base_url + "/metrics") as response:
                        self.assertEqual(await response.text(), metrics.decode())

                with (
                    patch.object(
                        benchmark.importlib.metadata, "version", return_value="0.23.0"
                    ),
                    patch.object(benchmark, "smoke", side_effect=probe),
                ):
                    self.assertEqual(await benchmark.run(args), 0)
        finally:
            await runner.cleanup()

    async def test_proxy_preserves_body_headers_and_split_stream_bytes(self):
        from aiohttp import ClientSession, web

        received = {}
        stream = (
            'data: {"id":"reply","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}],'
            '"usage":{"prompt_tokens":75012,"completion_tokens":1000,"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
            "data: [DONE]\n\n"
        ).encode()

        async def upstream(request):
            received["body"] = await request.read()
            received["id"] = request.headers.get("x-request-id")
            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "x-request-id": "server-id",
                }
            )
            await response.prepare(request)
            for offset in range(0, len(stream), 7):
                await response.write(stream[offset : offset + 7])
                await asyncio.sleep(0)
            await response.write_eof()
            return response

        origin = web.Application()
        origin.router.add_post("/v1/chat/completions", upstream)
        origin_runner = web.AppRunner(origin)
        await origin_runner.setup()
        origin_site = web.TCPSite(origin_runner, "127.0.0.1", 0)
        await origin_site.start()
        origin_url = (
            f"http://127.0.0.1:{origin_site._server.sockets[0].getsockname()[1]}"
        )
        try:
            async with ClientSession() as session:
                recorder = benchmark.Recorder(session, origin_url)
                proxy = web.Application()
                proxy.router.add_route("*", "/{tail:.*}", recorder.handle)
                proxy_runner = web.AppRunner(proxy)
                await proxy_runner.setup()
                proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
                await proxy_site.start()
                proxy_url = (
                    f"http://127.0.0.1:{proxy_site._server.sockets[0].getsockname()[1]}"
                )
                try:
                    with tempfile.TemporaryDirectory() as folder:
                        recorder.begin(Path(folder), "test-")
                        body = b'{"model":"GLM-5.3","messages":[{"role":"user","content":"exact bytes"}],"stream":true}'
                        async with session.post(
                            proxy_url + "/v1/chat/completions",
                            data=body,
                            headers={
                                "x-request-id": "test-0",
                                "Content-Type": "application/json",
                            },
                        ) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(
                                response.headers["x-request-id"], "server-id"
                            )
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
