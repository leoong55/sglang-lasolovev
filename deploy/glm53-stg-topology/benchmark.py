"""Run the STG topology experiment in the pinned vLLM 0.23.0 image.

The loopback recorder forwards request bodies and SSE bytes unchanged. vLLM
already requests stream_options.include_usage; the recorder adds no sampling
fields. Its request-id prefix separates measured requests from vLLM's initial
ready check and warmup. GPU/PP evidence is collected separately from serving
logs; client concurrency is never substituted for scheduler concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

MODEL = "GLM-5.3"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
SAMPLING_FIELDS = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "ignore_eos",
    "chat_template_kwargs",
    "stream",
    "stream_options",
)
METRIC_PREFIXES = (
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_retracted_reqs",
    "sglang:num_used_tokens",
    "sglang:num_tokens",
    "sglang:token_usage",
    "sglang:cache_hit_rate",
    "sglang:cache_hit_tokens",
    "sglang:cache_hit_count",
    "sglang:gen_throughput",
    "sglang:hicache",
    "sglang:kv_cache",
    "sglang:num_cached_tokens",
    "sglang:cached_tokens",
    "sglang:token_pool",
    "sglang:gpu_memory",
    "sglang:gpu_utilization",
    "sglang:retract",
    "sglang:prefill_effective_tokens",
    "sglang:load_back_tokens",
)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def append_json(path: Path, data: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower()
                in {
                    "api_key",
                    "admin_api_key",
                    "authorization",
                    "token",
                    "password",
                    "hf_token",
                    "access_token",
                    "secret_key",
                }
                and item
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def build_command(
    kind: str, base_url: str, tokenizer: str, result_dir: Path, request_prefix: str
) -> list[str]:
    if kind not in {"short", "long-cold", "long-warm"}:
        raise ValueError(f"Unknown workload: {kind}")
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        MODEL,
        "--tokenizer",
        tokenizer,
        "--seed",
        "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        "40",
        "--request-id-prefix",
        request_prefix,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "vllm.json",
        "--disable-tqdm",
    ]
    if kind == "short":
        command += [
            "--dataset-name",
            "random",
            "--random-input-len",
            "1000",
            "--random-output-len",
            "1000",
            "--num-prompts",
            "400",
            "--metric-percentiles",
            "50,90,95,99",
        ]
    else:
        command += [
            "--dataset-name",
            "prefix_repetition",
            "--prefix-repetition-prefix-len",
            "60000",
            "--prefix-repetition-suffix-len",
            "15000",
            "--prefix-repetition-output-len",
            "1000",
            "--prefix-repetition-num-prefixes",
            "20",
            "--num-prompts",
            "300",
            "--num-warmups",
            "1",
            "--ignore-eos",
            "--temperature",
            "0.3",
            "--extra-body",
            '{"chat_template_kwargs":{"enable_thinking":true}}',
            "--metric-percentiles",
            "50,95,99",
        ]
    return command


def assign_missing_request_ids(samples: list[Any], prefix: str) -> list[Any]:
    """vLLM 0.23.0's prefix_repetition sample() omits its request_id_prefix.

    Only transport identifiers are filled in; prompts, lengths, sample order,
    RNG state and sampling parameters are untouched. The ready-check/warmup
    RequestFuncInput remains unlabelled in upstream benchmark().
    """
    for index, sample in enumerate(samples):
        if sample.request_id is None:
            sample.request_id = prefix + str(index)
    return samples


def prepare_samples(args: Any, samples: list[Any]) -> list[Any]:
    # vLLM 0.23.0 forces random datasets to ignore EOS before get_samples().
    # Restore the experiment's explicitly requested natural-EOS short behavior.
    if args.dataset_name == "random":
        args.ignore_eos = False
    return assign_missing_request_ids(samples, args.request_id_prefix)


def invoke_vllm_cli(argv: list[str]) -> None:
    import vllm.benchmarks.serve as serving_benchmark
    from vllm.entrypoints.cli.main import main as vllm_main

    original = serving_benchmark.get_samples

    def identified_samples(args: Any, tokenizer: Any) -> Any:
        return prepare_samples(args, original(args, tokenizer))

    serving_benchmark.get_samples = identified_samples
    sys.argv = ["vllm", *argv]
    vllm_main()


class ResponseEvidence:
    """Bounded observer for SSE events, including chunks split inside UTF-8."""

    def __init__(self) -> None:
        self.pending = b""
        self.finish_reasons: dict[str, str] = {}
        self.usage: dict[str, Any] | None = None
        self.response_id: str | None = None
        self.content_sha256 = hashlib.sha256()
        self.content_bytes = 0
        self.preview = ""
        self.done = False
        self.errors: list[str] = []

    def observe(self, data: dict[str, Any]) -> None:
        if data.get("id"):
            self.response_id = data["id"]
        if isinstance(data.get("usage"), dict):
            self.usage = data["usage"]
        if data.get("error"):
            self.errors.append(str(data["error"])[:400])
        for choice in data.get("choices", []):
            if choice.get("finish_reason") is not None:
                self.finish_reasons[str(choice.get("index", 0))] = choice[
                    "finish_reason"
                ]
            delta = choice.get("delta", choice.get("message", {}))
            for key in ("content", "reasoning_content", "reasoning"):
                content = delta.get(key)
                if isinstance(content, str) and content:
                    encoded = content.encode()
                    self.content_sha256.update(key.encode() + b":" + encoded)
                    self.content_bytes += len(encoded)
                    if len(self.preview) < 160:
                        self.preview = (self.preview + content)[:160]

    def feed(self, chunk: bytes) -> None:
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.done = True
            elif payload:
                try:
                    self.observe(json.loads(payload))
                except (ValueError, TypeError, AttributeError) as exc:
                    if len(self.errors) < 5:
                        self.errors.append(f"SSE parse: {type(exc).__name__}")
        if len(self.pending) > 1024 * 1024:
            self.errors.append("SSE line exceeds 1 MiB; evidence incomplete")
            self.pending = b""

    def result(self) -> dict[str, Any]:
        usage = self.usage or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", usage.get("cached_tokens"))
        return {
            "response_id": self.response_id,
            "finish_reasons": self.finish_reasons,
            "usage": self.usage,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached,
            "cache_observability": (
                "reported" if cached is not None else "not_observable"
            ),
            "content_bytes": self.content_bytes,
            "content_sha256": self.content_sha256.hexdigest(),
            "output_preview": self.preview,
            "sse_done": self.done,
            "evidence_errors": self.errors,
        }


def select_metrics(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.startswith(METRIC_PREFIXES)
        and not re.search(r"_(?:bucket|sum|count)(?:\{| )", line.split("{", 1)[0])
    ]


def dp_activity(payload: Any, profile: str) -> dict[str, Any]:
    """Never add TP/PP replicas. Only a complete unique set of DP leaders counts."""
    if profile == "pp2":
        return {
            "observable": False,
            "reason": "PP unique-rid evidence requires serving observer logs",
        }
    expected = int(profile.removeprefix("dpa"))
    rows = payload.get("loads", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != expected:
        return {"observable": False, "reason": "Incomplete DP-group snapshot"}
    if sorted(row.get("dp_rank", -1) for row in rows) != list(range(expected)):
        return {"observable": False, "reason": "Missing or duplicate DP ranks"}
    if any(not isinstance(row.get("num_running_reqs"), int) for row in rows):
        return {"observable": False, "reason": "Missing running-request counter"}
    return {
        "observable": True,
        "running": sum(row["num_running_reqs"] for row in rows),
        "queued": sum(row.get("num_waiting_reqs", 0) for row in rows),
        "groups": {str(row["dp_rank"]): row["num_running_reqs"] for row in rows},
    }


def sampling_matches(sampling: dict[str, Any], kind: str) -> bool:
    if (
        sampling.get("max_completion_tokens", sampling.get("max_tokens")) != 1000
        or sampling.get("stream") is not True
        or sampling.get("stream_options", {}).get("include_usage") is not True
    ):
        return False
    if kind == "short":
        return (
            not sampling.get("ignore_eos")
            and "temperature" not in sampling
            and "chat_template_kwargs" not in sampling
        )
    return (
        sampling.get("ignore_eos") is True
        and sampling.get("temperature") == 0.3
        and sampling.get("chat_template_kwargs") == {"enable_thinking": True}
    )


def validate_requests(
    records: list[dict[str, Any]], kind: str, expected: int
) -> dict[str, Any]:
    measured = [item for item in records if item.get("measured")]
    errors = []
    if len(measured) != expected:
        errors.append(
            f"Expected {expected} measured requests, observed {len(measured)}"
        )
    ids = [item.get("request_id") for item in measured]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate measured request IDs")
    bad = []
    for item in measured:
        reasons = item.get("finish_reasons", {})
        tokens = item.get("completion_tokens")
        valid = (
            item.get("status") == 200
            and item.get("sse_done")
            and not item.get("evidence_errors")
            and not item.get("proxy_error")
            and set(reasons) == {"0"}
            and reasons["0"] in {"length", "stop"}
            and isinstance(tokens, int)
            and 0 < tokens <= 1000
            and sampling_matches(item.get("sampling", {}), kind)
        )
        if kind.startswith("long"):
            valid = valid and reasons.get("0") == "length" and tokens == 1000
        if not valid:
            bad.append(item.get("request_id"))
    if bad:
        errors.append(
            f"{len(bad)} requests failed completion/length/evidence validation"
        )
    return {
        "functional_valid": not errors,
        "expected_requests": expected,
        "observed_requests": len(measured),
        "invalid_request_ids": bad,
        "errors": errors,
        "finish_reasons": dict(
            Counter(
                reason
                for item in measured
                for reason in item.get("finish_reasons", {}).values()
            )
        ),
        "completion_tokens": dict(
            Counter(str(item.get("completion_tokens")) for item in measured)
        ),
        "cache_observability": dict(
            Counter(
                item.get("cache_observability", "not_observable") for item in measured
            )
        ),
        "started_at": min((item["started_at"] for item in measured), default=None),
        "finished_at": max((item["finished_at"] for item in measured), default=None),
    }


class Recorder:
    def __init__(self, session: Any, target: str) -> None:
        self.session = session
        self.target = target.rstrip("/")
        self.directory: Path | None = None
        self.request_prefix = ""
        self.records: list[dict[str, Any]] = []
        self.active = 0
        self.peak = 0
        self.sequence = 0

    def begin(self, directory: Path, request_prefix: str) -> None:
        if self.active:
            raise RuntimeError("Cannot start a run with active proxy requests")
        self.directory, self.request_prefix = directory, request_prefix
        self.records, self.peak, self.sequence = [], 0, 0

    async def handle(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
        }
        headers["Accept-Encoding"] = "identity"
        if request.path != "/v1/chat/completions":
            async with self.session.request(
                request.method,
                self.target + request.rel_url.path_qs,
                data=body,
                headers=headers,
            ) as response:
                return web.Response(
                    status=response.status,
                    body=await response.read(),
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() not in HOP_HEADERS
                    },
                )
        self.sequence += 1
        request_id = request.headers.get("x-request-id")
        record = {
            "sequence": self.sequence,
            "request_id": request_id,
            "measured": bool(request_id and request_id.startswith(self.request_prefix)),
            "started_at": time.time(),
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "request_bytes": len(body),
        }
        try:
            parsed = json.loads(body)
            record["sampling"] = {
                key: parsed[key] for key in SAMPLING_FIELDS if key in parsed
            }
        except ValueError:
            record["request_json_invalid"] = True
        observer = ResponseEvidence()
        self.active += 1
        self.peak = max(self.peak, self.active)
        output = None
        try:
            async with self.session.request(
                request.method,
                self.target + request.rel_url.path_qs,
                data=body,
                headers=headers,
            ) as upstream:
                record["status"] = upstream.status
                record["response_request_id"] = upstream.headers.get("x-request-id")
                output = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() not in HOP_HEADERS
                    },
                )
                await output.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await output.write(chunk)
                    observer.feed(chunk)
                if observer.pending:
                    observer.feed(b"\n")
                await output.write_eof()
                return output
        except Exception as exc:
            record["proxy_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if output is None:
                return web.json_response(
                    {"error": "Benchmark recorder upstream failure"}, status=502
                )
            raise
        finally:
            self.active -= 1
            record.update(observer.result())
            record["finished_at"] = time.time()
            self.records.append(record)
            if self.directory:
                append_json(self.directory / "requests.jsonl", record)


async def get_json(session: Any, url: str, timeout: int = 20) -> Any:
    async with session.get(url, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()


async def collect_telemetry(
    session: Any, args: argparse.Namespace, directory: Path, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        started = time.monotonic()
        sample: dict[str, Any] = {"timestamp": time.time()}
        try:
            async with session.get(args.base_url + "/metrics", timeout=5) as response:
                sample["metrics_status"] = response.status
                if response.status == 200:
                    sample["metrics"] = select_metrics(await response.text())
        except Exception as exc:
            sample["metrics_error"] = type(exc).__name__
        try:
            payload = await get_json(session, args.base_url + "/v1/loads", timeout=5)
            sample["loads"] = payload
            sample["activity"] = dp_activity(payload, args.profile)
        except Exception as exc:
            sample["loads_error"] = type(exc).__name__
        append_json(directory / "telemetry.jsonl", sample)
        try:
            await asyncio.wait_for(
                stop.wait(), max(0.01, 1 - (time.monotonic() - started))
            )
        except asyncio.TimeoutError:
            pass


async def flush_cache(
    session: Any, args: argparse.Namespace, recorder: Recorder, directory: Path
) -> None:
    if recorder.active:
        raise RuntimeError("Cannot flush while benchmark requests are active")
    # Scheduler itself rejects a flush while any request is running or queued,
    # including PP microbatches. A local counter alone is never sufficient.
    async with session.post(
        args.base_url + "/flush_cache?timeout=0", timeout=120
    ) as response:
        result = {
            "timestamp": time.time(),
            "status": response.status,
            "body": (await response.text())[:4096],
        }
        write_json(directory / "cache-flush.json", result)
        if response.status != 200:
            raise RuntimeError(f"Cache flush was rejected (HTTP {response.status})")


async def snapshot(
    session: Any, args: argparse.Namespace, directory: Path, label: str
) -> None:
    for endpoint, filename in (
        ("/get_server_info", "server-info"),
        ("/v1/models", "models"),
    ):
        try:
            result = await get_json(session, args.base_url + endpoint)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        write_json(directory / f"{label}-{filename}.json", redact(result))
    try:
        async with session.get(args.base_url + "/metrics", timeout=20) as response:
            response.raise_for_status()
            (directory / f"{label}-metrics.prom").write_text(await response.text())
    except Exception as exc:
        write_json(
            directory / f"{label}-metrics-error.json", {"error": type(exc).__name__}
        )


def activity_verdict(
    path: Path, profile: str, start: float | None, end: float | None
) -> dict[str, Any]:
    points = []
    if profile != "pp2" and path.exists() and start is not None and end is not None:
        for line in path.read_text().splitlines():
            point = json.loads(line)
            if start <= point["timestamp"] <= end and point.get("activity", {}).get(
                "observable"
            ):
                points.append(point["activity"]["running"])
    return {
        "server_c40_confirmed": bool(points and max(points) >= 40),
        "server_running_peak": max(points, default=None),
        "server_activity_samples": len(points),
        "server_samples_running_at_least_40": sum(value >= 40 for value in points),
        "server_activity_source": (
            "serving PP observer logs required"
            if profile == "pp2"
            else "/v1/loads unique DP leaders"
        ),
        "capacity_status": (
            "C40_OBSERVED" if points and max(points) >= 40 else "C40_UNPROVEN"
        ),
    }


async def run_workload(
    session: Any,
    args: argparse.Namespace,
    recorder: Recorder,
    local_url: str,
    kind: str,
    repetition: int,
) -> dict[str, Any]:
    directory = args.results_dir / f"r{repetition:02d}-{kind}"
    directory.mkdir(parents=True, exist_ok=False)
    prefix = f"glm53-{args.profile}-{directory.name}-{uuid.uuid4().hex[:8]}-"
    recorder.begin(directory, prefix)
    if kind in {"short", "long-cold"}:
        await flush_cache(session, args, recorder, directory)
    command = build_command(kind, local_url, args.tokenizer, directory, prefix)
    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--vllm-cli",
        *command[1:],
    ]
    write_json(
        directory / "command.json",
        {
            "argv": command,
            "invocation": invocation,
            "upstream": args.base_url,
            "recorder": "loopback byte-forwarding; no sampling mutation",
            "request_id_hook": "fills missing sample.request_id only; header metadata, not request body",
            "natural_eos_hook": "undo vLLM 0.23.0 automatic random ignore_eos=True; short only",
            "cache_state": (
                "flushed before vLLM ready check and warmup"
                if kind != "long-warm"
                else "immediate repeated workload; no cache flush"
            ),
        },
    )
    # Take configuration once per run; it is outside vLLM's timed interval.
    await snapshot(session, args, directory, "before")
    stop = asyncio.Event()
    telemetry = asyncio.create_task(collect_telemetry(session, args, directory, stop))
    print(
        json.dumps(
            {
                "event": "benchmark_started",
                "kind": kind,
                "repetition": repetition,
                "directory": str(directory),
            }
        ),
        flush=True,
    )
    environment = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
    )
    # vLLM trusts proxy environment; loopback and service traffic must stay local.
    environment["NO_PROXY"] = "*"
    environment["no_proxy"] = "*"
    try:
        with (directory / "vllm.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *invocation,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
            )
            try:
                returncode = await asyncio.wait_for(
                    process.wait(), args.workload_timeout
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 30)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise
    finally:
        stop.set()
        await telemetry
    verdict = validate_requests(recorder.records, kind, 400 if kind == "short" else 300)
    verdict.update(
        activity_verdict(
            directory / "telemetry.jsonl",
            args.profile,
            verdict["started_at"],
            verdict["finished_at"],
        )
    )
    verdict.update(
        {
            "workload": kind,
            "repetition": repetition,
            "exit_code": returncode,
            "client_proxy_peak_inflight": recorder.peak,
            "client_concurrency_is_not_server_concurrency": True,
        }
    )
    if returncode != 0 or not (directory / "vllm.json").is_file():
        verdict["functional_valid"] = False
        verdict["errors"].append("vLLM failed or its detailed result is missing")
    else:
        result = json.loads((directory / "vllm.json").read_text())
        verdict["benchmark_metrics"] = {
            key: value
            for key, value in result.items()
            if not isinstance(value, (list, dict))
        }
        if result.get("completed") != verdict["expected_requests"] or result.get(
            "failed", 0
        ):
            verdict["functional_valid"] = False
            verdict["errors"].append("vLLM result completion count failed validation")
    write_json(directory / "verdict.json", verdict)
    print(
        json.dumps(
            {
                "event": "benchmark_finished",
                "directory": str(directory),
                "functional_valid": verdict["functional_valid"],
                "capacity_status": verdict["capacity_status"],
            }
        ),
        flush=True,
    )
    return verdict


async def smoke(session: Any, args: argparse.Namespace) -> None:
    directory = args.results_dir / "smoke"
    directory.mkdir(parents=True, exist_ok=False)
    await snapshot(session, args, directory, "before")
    nonce = "GLM53_READY_" + uuid.uuid4().hex[:12]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply with exactly {nonce}"}],
        # The checkpoint reasons before answering even for a nonce. This is a
        # correctness probe, so allow reasoning to finish before checking EOS.
        "max_completion_tokens": 2048,
        "temperature": 0,
        # This checkpoint's template always opens <think>. Keep glm45 reasoning
        # extraction enabled so only the final answer is checked against nonce.
        "chat_template_kwargs": {"enable_thinking": True},
    }
    async with session.post(
        args.base_url + "/v1/chat/completions", json=body
    ) as response:
        response.raise_for_status()
        result = await response.json()
    write_json(directory / "short.json", result)
    choice = result.get("choices", [{}])[0]
    if (
        choice.get("finish_reason") != "stop"
        or choice.get("message", {}).get("content", "").strip() != nonce
    ):
        raise RuntimeError(
            "Short smoke failed exact-nonce/finish_reason=stop validation"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    # The smoke checks long admission and KV independently from the benchmark
    # dataset; the exact prefix-repetition data remains generated by vLLM.
    fragment = tokenizer.encode(
        "The benchmark verifies long context model execution. ",
        add_special_tokens=False,
    )
    token_ids = (fragment * (75000 // len(fragment) + 1))[:75000]
    text = tokenizer.decode(token_ids)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_completion_tokens": 16,
        "ignore_eos": True,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    async with session.post(
        args.base_url + "/v1/chat/completions", json=body
    ) as response:
        response.raise_for_status()
        result = await response.json()
    write_json(directory / "long.json", result)
    write_json(
        directory / "long-input.json",
        {
            "tokenizer": args.tokenizer,
            "input_tokens_before_chat_template": len(
                tokenizer.encode(text, add_special_tokens=False)
            ),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )
    if (
        result.get("choices", [{}])[0].get("finish_reason") != "length"
        or result.get("usage", {}).get("completion_tokens") != 16
        or not 74000 <= result.get("usage", {}).get("prompt_tokens", 0) <= 76000
    ):
        raise RuntimeError("Long smoke failed prompt length/completion validation")
    await snapshot(session, args, directory, "after")
    await flush_cache(session, args, Recorder(session, args.base_url), directory)
    print(
        json.dumps({"event": "smoke_passed", "directory": str(directory)}), flush=True
    )


def prepare_dataset_catalog(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "smoke":
        return {}
    path = args.results_dir / "dataset-catalog.json"
    if path.exists():
        raise RuntimeError("Refusing to overwrite an existing dataset catalog")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    print(json.dumps({"event": "dataset_catalog_started"}), flush=True)
    # Generate offline in the supervisor before launching any measured vLLM
    # subprocess. Each subprocess seeds its own RNG, independently of this work.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        from dataset_catalog import build

        catalog = build(Path(args.tokenizer))
    write_json(path, catalog)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"event": "dataset_catalog_saved", "sha256": digest}), flush=True)
    return {
        "dataset_catalog_sha256": digest,
        "dataset_catalog_schema_version": catalog["schema_version"],
    }


async def run(args: argparse.Namespace) -> int:
    from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

    args.results_dir.mkdir(parents=True, exist_ok=True)
    # Fail on accidental reruns rather than overwriting existing evidence.
    provenance_path = args.results_dir / "provenance.json"
    if provenance_path.exists():
        raise RuntimeError(f"Result directory already used: {args.results_dir}")
    version = importlib.metadata.version("vllm")
    if version.split("+")[0] != "0.23.0":
        raise RuntimeError(f"Expected pinned vLLM 0.23.0, found {version}")
    # Catalog capture has its own asyncio.run() for the pinned vLLM client.
    # Complete it offline in a worker before starting the HTTP session.
    catalog_provenance = await asyncio.to_thread(prepare_dataset_catalog, args)
    write_json(
        provenance_path,
        {
            "timestamp": time.time(),
            "profile": args.profile,
            "mode": args.mode,
            "vllm_version": version,
            "model": MODEL,
            "source_commit": args.source_commit,
            "serving_image_digest": args.image_digest,
            "benchmark_image": os.environ.get("BENCHMARK_IMAGE"),
            "pod_uid": os.environ.get("POD_UID"),
            "pod_name": os.environ.get("POD_NAME"),
            "server_url": args.base_url,
            "tokenizer": args.tokenizer,
            "argv": vars(args) | {"results_dir": str(args.results_dir)},
            "short_sampling": "natural EOS restored after vLLM 0.23.0 random override; temperature/thinking omitted",
            "cold_definition": "flush before vLLM initial ready check and explicit warmup",
            "server_pp_c40": "requires separate unique-rid observer evidence from serving logs",
            **catalog_provenance,
        },
    )
    async with ClientSession(
        timeout=ClientTimeout(total=args.workload_timeout),
        connector=TCPConnector(limit=0),
        trust_env=False,
        auto_decompress=False,
    ) as session:
        if args.mode == "smoke":
            await smoke(session, args)
            return 0
        recorder = Recorder(session, args.base_url)
        application = web.Application(client_max_size=32 * 1024 * 1024)
        application.router.add_route("*", "/{tail:.*}", recorder.handle)
        server = web.AppRunner(application, access_log=None)
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        local_url = f"http://127.0.0.1:{port}"
        summaries = []
        try:
            for repetition in range(
                args.start_repetition, args.start_repetition + args.repetitions
            ):
                kinds = (["short"] if not args.skip_short else []) + [
                    "long-cold",
                    "long-warm",
                ]
                for kind in kinds:
                    verdict = await run_workload(
                        session, args, recorder, local_url, kind, repetition
                    )
                    summaries.append(verdict)
                    write_json(args.results_dir / "benchmark-summary.json", summaries)
                    if not verdict["functional_valid"]:
                        raise RuntimeError(
                            f"{kind} failed: {'; '.join(verdict['errors'])}"
                        )
            await snapshot(session, args, args.results_dir, "final")
        finally:
            await server.cleanup()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "suite", "repeat"), default="suite")
    parser.add_argument(
        "--profile", choices=("pp2", "dpa2", "dpa4", "dpa8"), required=True
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("BASE_URL", "http://glm53-topology:8080")
    )
    parser.add_argument("--tokenizer", default=os.environ.get("MODEL_PATH", "/model"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--start-repetition", type=int, default=1)
    parser.add_argument("--skip-short", action="store_true")
    parser.add_argument("--source-commit", default=os.environ.get("SOURCE_COMMIT"))
    parser.add_argument("--image-digest", default=os.environ.get("SERVING_IMAGE"))
    parser.add_argument("--workload-timeout", type=int, default=21600)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.start_repetition < 1:
        parser.error("repetition counts must be positive")
    args.base_url = args.base_url.rstrip("/")
    return args


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--vllm-cli":
        invoke_vllm_cli(sys.argv[2:])
        return 0
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "event": "benchmark_failed",
            "timestamp": time.time(),
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }
        # A failure does not erase previous partial results.
        write_json(args.results_dir / f"failure-{time.time_ns()}.json", error)
        print(json.dumps(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
