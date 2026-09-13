"""SYNTHETIC CLIENT HARNESS: real pinned vLLM CLI, no model or GPU inference.

Run in the pinned vLLM image with the real model/tokenizer mounted read-only:
  python /scripts/client_harness.py --benchmark-file /scripts/benchmark.py \
    --tokenizer /model --results-dir /results/client-validation-<unique-id>

All timings, output tokens, server responses and throughput in the child artifacts
are synthetic validation fixtures. Never publish them as model performance.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

LABEL = "SYNTHETIC CLIENT HARNESS — NOT MODEL PERFORMANCE"
MODEL = "GLM-5.3"
KINDS = {"short": 400, "long-cold": 300, "long-warm": 300}
MEASURED_ID = re.compile(
    r"glm53-pp2-r01-(short|long-cold|long-warm)-[0-9a-f]{8}-(\d+)\Z"
)
SYNTHETIC_IMAGE = "synthetic-client-harness-no-serving-image"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def single_input_token_count(encoded):
    """Count actual IDs across Transformers list/BatchEncoding return contracts."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("chat template mapping has no input_ids")
        ids = encoded["input_ids"]
    else:
        ids = encoded
    metadata = {
        "result_type": type(encoded).__name__,
        "input_ids_type": type(ids).__name__,
    }
    if callable(getattr(ids, "tolist", None)):
        ids = ids.tolist()
    if not isinstance(ids, (list, tuple)):
        raise ValueError("chat template input_ids is not a token sequence")
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("chat template returned multiple input batches")
        ids = ids[0]
        metadata["input_ids_shape"] = [1, len(ids)]
    else:
        metadata["input_ids_shape"] = [len(ids)]
    if not ids or any(
        not isinstance(token, Integral) or isinstance(token, bool) or token < 0
        for token in ids
    ):
        raise ValueError("chat template input_ids contains invalid token IDs")
    return len(ids), metadata


def single_text_content(content):
    """Accept smoke strings and vLLM 0.23.0's single text content block.

    The upstream endpoint_request_func._get_chat_content wraps a plain prompt
    in [{"type": "text", "text": prompt}]. No multimodal input is in this suite.
    Only extract text for validation; hash the untouched wire body separately.
    """
    if isinstance(content, str):
        return content
    if (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and set(content[0]) == {"type", "text"}
        and content[0]["type"] == "text"
        and isinstance(content[0]["text"], str)
    ):
        return content[0]["text"]
    raise ValueError("unexpected_message_content_shape")


class Harness:
    def __init__(self, root, tokenizer, benchmark):
        self.root, self.tokenizer, self.benchmark = root, tokenizer, benchmark
        self.calls = Counter()
        self.measured = Counter()
        self.unlabelled = Counter()
        self.smoke_counts = Counter()
        self.seen_ids = set()
        self.active = 0
        self.response_sequence = 0
        self.flushes = []
        self.violations = []
        self.stage = "smoke"
        self.catalog = None
        self.events = root / "synthetic-server-events.jsonl"

    def require(self, condition, reason):
        if not condition:
            self.violations.append(reason)
            raise ValueError(reason)

    def event(self, value):
        with self.events.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "purpose": "validation",
                        "synthetic": True,
                        "timestamp": time.time(),
                        **value,
                    }
                )
                + "\n"
            )

    def load_catalog(self):
        if self.catalog is None:
            raw = json.loads((self.root / "suite/dataset-catalog.json").read_text())
            self.require(raw.get("dataset") == "prefix_repetition", "catalog_dataset")
            self.require(
                raw.get("configuration")
                == {
                    "seed": 0,
                    "prefix_len": 60000,
                    "suffix_len": 15000,
                    "output_len": 1000,
                    "num_prefixes": 20,
                    "num_requests": 300,
                },
                "catalog_configuration",
            )
            self.require(len(raw.get("samples", [])) == 300, "catalog_sample_count")
            self.catalog = {row["sample_index"]: row for row in raw["samples"]}
            self.require(set(self.catalog) == set(range(300)), "catalog_sample_indices")
        return self.catalog

    async def handle(self, request):
        from aiohttp import web

        self.calls[f"{request.method} {request.path}"] += 1
        try:
            if request.method == "GET" and request.path in {"/models", "/v1/models"}:
                self.require(not request.query, "unexpected_models_query")
                return web.json_response(
                    {
                        "object": "list",
                        "synthetic": True,
                        "purpose": "validation",
                        "data": [
                            {
                                "id": MODEL,
                                "object": "model",
                                "created": 0,
                                "owned_by": "synthetic-client-harness",
                            }
                        ],
                    }
                )
            if request.method == "GET" and request.path == "/get_server_info":
                self.require(not request.query, "unexpected_server_info_query")
                # Deliberately no plausible KV capacity or real serving identity.
                return web.json_response(
                    {
                        "synthetic": True,
                        "purpose": "validation",
                        "status": "synthetic_fixture",
                        "version": "no-serving-engine",
                        "internal_states": [],
                    }
                )
            if request.method == "GET" and request.path == "/metrics":
                self.require(not request.query, "unexpected_metrics_query")
                return web.Response(
                    text="# SYNTHETIC CLIENT HARNESS: no model/GPU metrics\n"
                )
            if request.method == "GET" and request.path == "/v1/loads":
                self.require(not request.query, "unexpected_loads_query")
                return web.json_response(
                    {"synthetic": True, "purpose": "validation", "loads": []}
                )
            if request.method == "POST" and request.path == "/flush_cache":
                self.require(
                    dict(request.query) == {"timeout": "0"}, "unexpected_flush_query"
                )
                self.require(self.active == 0, "flush_while_requests_active")
                self.require(not await request.read(), "unexpected_flush_body")
                self.flushes.append(
                    {"stage": self.stage, "after_measured_counts": dict(self.measured)}
                )
                self.event({"event": "synthetic_flush", "ordinal": len(self.flushes)})
                return web.Response(text="SYNTHETIC cache flush acknowledged")
            self.require(
                request.method == "POST"
                and request.path == "/v1/chat/completions"
                and not request.query,
                "unexpected_endpoint_or_method",
            )
            return await self.completion(request)
        except Exception as error:
            if not isinstance(error, ValueError):
                self.violations.append("handler_" + type(error).__name__)
            self.event(
                {
                    "event": "synthetic_handler_failure",
                    "error_type": type(error).__name__,
                }
            )
            return web.json_response(
                {
                    "error": {
                        "message": "Synthetic client harness rejected unexpected request",
                        "type": "harness_validation_error",
                    }
                },
                status=400,
            )

    async def completion(self, request):
        from aiohttp import web

        raw = await request.read()
        body = json.loads(raw)
        self.require(isinstance(body, dict), "body_not_object")
        allowed = {
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "ignore_eos",
            "chat_template_kwargs",
            "stream",
            "stream_options",
        }
        self.require(set(body) <= allowed, "unexpected_completion_field")
        self.require(body.get("model") == MODEL, "unexpected_model")
        messages = body.get("messages")
        self.require(
            isinstance(messages, list) and len(messages) == 1, "unexpected_messages"
        )
        message = messages[0]
        self.require(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message["role"] == "user",
            "unexpected_message_shape",
        )
        try:
            prompt_text = single_text_content(message["content"])
        except ValueError:
            self.require(False, "unexpected_message_content_shape")
        limit = body.get("max_completion_tokens", body.get("max_tokens"))
        self.require(
            not ("max_completion_tokens" in body and "max_tokens" in body),
            "duplicate_token_limits",
        )
        self.active += 1
        try:
            self.response_sequence += 1
            rid = "synthetic-" + str(self.response_sequence)
            if body.get("stream") is not True:
                self.require(self.stage == "smoke", "nonstream_outside_smoke")
                self.require(
                    body.get("chat_template_kwargs") == {"enable_thinking": True},
                    "smoke_thinking",
                )
                self.require("stream_options" not in body, "nonstream_stream_options")
                if limit == 2048:
                    match = re.fullmatch(
                        r"Reply with exactly (GLM53_READY_[0-9a-f]{12})",
                        prompt_text,
                    )
                    self.require(
                        match is not None
                        and body.get("temperature") == 0
                        and "ignore_eos" not in body,
                        "short_smoke_payload",
                    )
                    content, finish, generated = match[1], "stop", 7
                    prompt_tokens = await asyncio.to_thread(
                        self.prompt_tokens, messages
                    )
                    self.smoke_counts["short"] += 1
                else:
                    self.require(
                        limit == 16
                        and body.get("temperature") == 0.3
                        and body.get("ignore_eos") is True,
                        "long_smoke_payload",
                    )
                    prompt_tokens = await asyncio.to_thread(
                        self.prompt_tokens, messages
                    )
                    self.require(
                        74000 <= prompt_tokens <= 76000,
                        "long_smoke_actual_tokenizer_length",
                    )
                    content, finish, generated = (
                        "SYNTHETIC long admission response",
                        "length",
                        16,
                    )
                    self.smoke_counts["long"] += 1
                self.event(
                    {
                        "event": "synthetic_smoke",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generated,
                        "finish_reason": finish,
                    }
                )
                return web.json_response(
                    {
                        "id": rid,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": MODEL,
                        "synthetic": True,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": finish,
                            }
                        ],
                        "usage": self.usage(prompt_tokens, generated),
                    }
                )

            self.require(
                self.stage == "suite" and limit == 1000,
                "streamed_payload_stage_or_limit",
            )
            self.require(
                body.get("stream_options") == {"include_usage": True},
                "streamed_usage_required",
            )
            header_id = request.headers.get("x-request-id")
            match = MEASURED_ID.fullmatch(header_id or "")
            long = body.get("ignore_eos") is True
            kind = match[1] if match else "long-warmup" if long else "short-warmup"
            self.require(not header_id or match is not None, "unexpected_request_id")
            self.require(
                self.benchmark.sampling_matches(body, "long-cold" if long else "short"),
                "sampling_contract",
            )
            if match:
                self.require(
                    (match[1] != "short") == long, "request_id_sampling_mismatch"
                )
                self.require(
                    header_id not in self.seen_ids, "duplicate_measured_request_id"
                )
                self.seen_ids.add(header_id)
                self.measured[kind] += 1
            else:
                self.unlabelled[kind] += 1
            if long:
                catalog = self.load_catalog()
                body_hash = digest(raw)
                if match:
                    entry = catalog.get(int(match[2]))
                    self.require(
                        entry is not None and body_hash == entry["request_body_sha256"],
                        "long_catalog_body_id_join",
                    )
                else:
                    entries = [
                        row
                        for row in catalog.values()
                        if row["request_body_sha256"] == body_hash
                    ]
                    self.require(len(entries) == 1, "warmup_body_not_catalog_sample")
                    entry = entries[0]
                self.require(
                    digest(prompt_text.encode()) == entry["full_prompt_sha256"],
                    "long_catalog_prompt_join",
                )
                prompt_tokens = entry["prompt_tokens_before_chat_template"]
                finish, generated = "length", 1000
            else:
                self.require(len(prompt_text) > 0, "empty_short_prompt")
                # Actual output length is deliberately shorter than max1000:
                # this tests that natural EOS is preserved by the real client.
                prompt_tokens, finish, generated = 1000, "stop", 7
            self.event(
                {
                    "event": "synthetic_stream",
                    "workload": kind,
                    "measured": bool(match),
                    "request_body_sha256": digest(raw),
                    "completion_tokens": generated,
                    "finish_reason": finish,
                }
            )
            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "x-request-id": rid,
                }
            )
            await response.prepare(request)
            common = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL,
            }
            # Usage is a synthetic protocol fixture, not a generated-token claim.
            for content in ("SYNTHETIC CLIENT ", "HARNESS OUTPUT"):
                chunk = common | {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ]
                }
                await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                await asyncio.sleep(0)
            final = common | {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
            }
            usage_chunk = common | {
                "choices": [],
                "usage": self.usage(prompt_tokens, generated),
            }
            await response.write(
                (
                    "data: "
                    + json.dumps(final)
                    + "\n\ndata: "
                    + json.dumps(usage_chunk)
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
            )
            await response.write_eof()
            return response
        finally:
            self.active -= 1

    def prompt_tokens(self, messages):
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        try:
            count, metadata = single_input_token_count(encoded)
        except ValueError:
            self.event(
                {
                    "event": "synthetic_prompt_tokenization",
                    "result_type": type(encoded).__name__,
                    "prompt_tokens": None,
                    "status": "unsupported_input_ids_shape",
                }
            )
            self.require(False, "chat_template_input_ids_shape")
        # Record the actual count before long-smoke admission validates its
        # bounds. Never emit prompt text, token IDs, or attention masks.
        self.event(
            {
                "event": "synthetic_prompt_tokenization",
                "status": "observed",
                **metadata,
                "prompt_tokens": count,
            }
        )
        return count

    @staticmethod
    def usage(prompt, generated):
        return {
            "prompt_tokens": prompt,
            "completion_tokens": generated,
            "total_tokens": prompt + generated,
            "prompt_tokens_details": {"cached_tokens": 0},
        }


def load_benchmark(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("synthetic_client_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_artifacts(root, harness, benchmark):
    harness.require(dict(harness.measured) == KINDS, "measured_request_matrix")
    harness.require(
        dict(harness.smoke_counts) == {"short": 1, "long": 1}, "smoke_request_counts"
    )
    harness.require(len(harness.flushes) == 3, "flush_count_expected_smoke_short_cold")
    harness.require(harness.flushes[0]["stage"] == "smoke", "first_flush_after_smoke")
    harness.require(
        harness.flushes[1]["after_measured_counts"] == {}, "short_flush_before_requests"
    )
    harness.require(
        harness.flushes[2]["after_measured_counts"] == {"short": 400},
        "cold_flush_after_short_before_long",
    )
    checks = {}
    catalog = harness.load_catalog()
    for kind, expected in KINDS.items():
        directory = root / "suite" / f"r01-{kind}"
        records = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        verdict = benchmark.validate_requests(records, kind, expected)
        harness.require(verdict["functional_valid"], "recorder_validation_" + kind)
        raw = json.loads((directory / "vllm.json").read_text())
        harness.require(
            raw.get("completed") == expected and raw.get("failed", 0) == 0,
            "actual_vllm_counts_" + kind,
        )
        measured = [row for row in records if row.get("measured")]
        if kind != "short":
            for row in measured:
                match = MEASURED_ID.fullmatch(row["request_id"])
                harness.require(
                    row["request_sha256"]
                    == catalog[int(match[2])]["request_body_sha256"],
                    "recorded_catalog_join_" + kind,
                )
        expected_tokens, expected_finish = (
            (7, "stop") if kind == "short" else (1000, "length")
        )
        harness.require(
            all(
                row["completion_tokens"] == expected_tokens
                and row["finish_reasons"] == {"0": expected_finish}
                for row in measured
            ),
            "synthetic_usage_finish_" + kind,
        )
        checks[kind] = {
            "measured_requests": len(measured),
            "vllm_completed": raw["completed"],
            "synthetic_completion_tokens_per_request": expected_tokens,
            "finish_reason": expected_finish,
            "functional_valid": True,
        }
    for mode in ("admission", "suite"):
        provenance = json.loads((root / mode / "provenance.json").read_text())
        harness.require(
            provenance.get("purpose") == "validation", "provenance_validation_purpose"
        )
        harness.require(
            provenance.get("serving_image_digest") == SYNTHETIC_IMAGE,
            "provenance_no_real_serving_image",
        )
        harness.require(
            provenance.get("benchmark_script_sha256")
            == digest(Path(benchmark.__file__).read_bytes()),
            "provenance_benchmark_script_hash",
        )
    harness.require(not harness.violations, "previous_handler_violation")
    return checks


async def execute(args):
    from aiohttp import web
    from transformers import AutoTokenizer

    root = args.results_dir
    if not root.name.startswith("client-validation"):
        raise ValueError("results directory name must start with client-validation")
    if root.exists() and any(root.iterdir()):
        raise ValueError("refusing to overwrite existing client validation artifacts")
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        BENCHMARK_PURPOSE="validation",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        NO_PROXY="*",
        no_proxy="*",
    )
    benchmark = load_benchmark(args.benchmark_file)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "transformers", "aiohttp")
    }
    if versions["vllm"].split("+")[0] != "0.23.0":
        raise ValueError("the real pinned vLLM0.23.0 client is required")
    write_json(
        root / "SYNTHETIC_CLIENT_HARNESS.json",
        {
            "label": LABEL,
            "purpose": "validation",
            "model_performance": False,
            "gpu_inference": False,
            "harness_sha256": digest(Path(__file__).read_bytes()),
            "benchmark_sha256": digest(args.benchmark_file.read_bytes()),
            "dataset_catalog_script_sha256": digest(
                (args.benchmark_file.parent / "dataset_catalog.py").read_bytes()
            ),
            "versions": versions,
            "declared_client_image": os.environ.get("BENCHMARK_IMAGE"),
            "tooling_commit": os.environ.get("TOOLING_COMMIT"),
            "configmap_sha256": os.environ.get("SCRIPT_CONFIG_SHA256"),
        },
    )
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    harness = Harness(root, tokenizer, benchmark)
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", harness.handle)
    server = web.AppRunner(app, access_log=None)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    outcome = {
        "label": LABEL,
        "purpose": "validation",
        "model_performance": False,
        "status": "failed",
    }
    try:
        for mode, directory in (("smoke", "admission"), ("suite", "suite")):
            harness.stage = mode
            options = benchmark.parse_args(
                [
                    "--mode",
                    mode,
                    "--profile",
                    "pp2",
                    "--base-url",
                    url,
                    "--tokenizer",
                    args.tokenizer,
                    "--results-dir",
                    str(root / directory),
                    "--source-commit",
                    "synthetic-no-serving-commit",
                    "--image-digest",
                    SYNTHETIC_IMAGE,
                    "--workload-timeout",
                    str(args.timeout),
                ]
            )
            print(
                json.dumps(
                    {"label": LABEL, "event": "validation_stage_started", "stage": mode}
                ),
                flush=True,
            )
            code = await benchmark.run(options)
            harness.require(code == 0, "benchmark_exit_" + mode)
        outcome.update(
            status="passed", workload_checks=verify_artifacts(root, harness, benchmark)
        )
    except Exception as error:
        outcome["failure_type"] = type(error).__name__
        raise
    finally:
        await server.cleanup()
        outcome.update(
            endpoint_calls=dict(harness.calls),
            smoke_requests=dict(harness.smoke_counts),
            measured_requests=dict(harness.measured),
            unlabelled_requests=dict(harness.unlabelled),
            flushes=harness.flushes,
            violations=harness.violations[:30],
            inference_or_capacity_claims=False,
        )
        write_json(root / "client-validation-summary.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-file", type=Path, default=Path("/scripts/benchmark.py")
    )
    parser.add_argument("--tokenizer", default="/model")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        asyncio.run(execute(args))
    except Exception as error:
        print(
            json.dumps(
                {
                    "label": LABEL,
                    "status": "failed",
                    "failure_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
