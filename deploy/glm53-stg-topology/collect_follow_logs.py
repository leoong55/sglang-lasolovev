"""Read-only binary Kubernetes log following with explicit connection segments.

No segments are joined across reconnects. A segment reports observed transport
continuity, not an infallible guarantee about kubelet/runtime log retention.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_SHA = hashlib.sha256(SOURCE_BYTES).hexdigest()
PROFILES = {"pp2", "dpa2", "dpa4", "dpa8"}
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def cri_time(line):
    token = line.split(b" ", 1)[0]
    try:
        parsed = datetime.fromisoformat(token.decode("ascii").replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, UnicodeError, OverflowError):
        return None


def container_status(pod):
    return next(
        (
            row
            for row in pod.get("status", {}).get("containerStatuses", [])
            if row["name"] == "sglang"
        ),
        {},
    )


class Follower:
    def __init__(self, pod, directory, prefix, max_bytes):
        self.pod, self.directory, self.prefix, self.max_bytes = (
            pod,
            directory,
            prefix,
            max_bytes,
        )
        self.process = None
        self.stopping = None
        self.pending = b""
        self.segment = None
        self.hasher = hashlib.sha256()
        self.stderr_bytes = 0
        self.fatal = False
        self.last_checkpoint = 0
        paths = [
            directory / name
            for name in (
                "server-follow.log",
                "server-follow.meta.json",
                "server-follow.collector.py",
                "server-follow.stderr.log",
            )
        ]
        if any(path.exists() for path in paths):
            raise ValueError("follow_artifact_exists_refusing_overwrite")
        self.log_path, self.meta_path, source_path, self.stderr_path = paths
        self.output = self.log_path.open("xb", buffering=0)
        with source_path.open("xb") as stream:
            stream.write(SOURCE_BYTES)
        self.stderr = self.stderr_path.open("xb", buffering=0)
        status = container_status(pod)
        container = next(
            row for row in pod["spec"]["containers"] if row["name"] == "sglang"
        )
        env = {row["name"]: row.get("value") for row in container.get("env", [])}
        self.metadata = {
            "schema_version": 1,
            "format": "kubectl-follow-timestamps-lf-v1",
            "capture_mode": "continuous_follow",
            "container": "sglang",
            "collector_sha256": SOURCE_SHA,
            "pod_uid": pod["metadata"]["uid"],
            "container_id": status.get("containerID"),
            "restart_count": status.get("restartCount"),
            "source_commit": env.get("SOURCE_COMMIT"),
            "image_id": status.get("imageID"),
            "capture_started_at": time.time(),
            "log_sha256": self.hasher.hexdigest(),
            "log_bytes": 0,
            "collection_errors": 0,
            "segments": [],
        }
        self.checkpoint(force=True)

    def error(self, kind):
        self.metadata["collection_errors"] += 1
        emit("follow_collection_error", type=kind)
        self.checkpoint(force=True)

    def checkpoint(self, force=False):
        if not force and time.monotonic() - self.last_checkpoint < 2:
            return
        self.output.flush()
        os.fsync(self.output.fileno())
        self.metadata["log_sha256"] = self.hasher.hexdigest()
        if self.segment is not None:
            self.segment["byte_end"] = self.metadata["log_bytes"]
            self.segment["partial_line_bytes"] = len(self.pending)
        atomic_json(self.meta_path, self.metadata)
        self.last_checkpoint = time.monotonic()

    def append(self, data):
        if self.metadata["log_bytes"] + len(data) > self.max_bytes:
            raise ValueError("follow_log_size_limit")
        view = memoryview(data)
        while view:
            written = self.output.write(view)
            if not written:
                raise OSError("follow_log_short_write")
            self.hasher.update(view[:written])
            self.metadata["log_bytes"] += written
            view = view[written:]
        self.pending += data
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            self.segment["records"] += 1
            timestamp = cri_time(line)
            if timestamp is None:
                self.segment["unframed_records"] += 1
                continue
            previous = self.segment["last_cri_timestamp"]
            if previous is not None and timestamp < previous:
                self.segment["clock_regressions"] += 1
            if self.segment["first_cri_timestamp"] is None:
                self.segment["first_cri_timestamp"] = timestamp
            self.segment["last_cri_timestamp"] = timestamp
        if len(self.pending) > MAX_LINE_BYTES:
            raise ValueError("follow_line_size_limit")
        self.checkpoint()

    async def stdout_reader(self):
        while data := await self.process.stdout.read(65536):
            self.append(data)

    async def stderr_reader(self):
        while data := await self.process.stderr.read(65536):
            remaining = max(0, MAX_STDERR_BYTES - self.stderr_bytes)
            if remaining:
                self.stderr.write(data[:remaining])
            self.stderr_bytes += len(data)
            if self.stderr_bytes > MAX_STDERR_BYTES:
                raise ValueError("follow_stderr_size_limit")

    async def stop(self, reason):
        self.stopping = reason
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def run(self):
        try:
            while not self.stopping:
                if len(self.metadata["segments"]) >= 4096:
                    self.error("follow_segment_limit")
                    self.fatal = True
                    break
                status = container_status(self.pod)
                self.pending = b""
                self.segment = {
                    "segment_id": len(self.metadata["segments"]),
                    "first_cri_timestamp": None,
                    "last_cri_timestamp": None,
                    "started_at": time.time(),
                    "ended_at": None,
                    "returncode": None,
                    "interruption": None,
                    "byte_start": self.metadata["log_bytes"],
                    "byte_end": self.metadata["log_bytes"],
                    "records": 0,
                    "unframed_records": 0,
                    "partial_line_bytes": 0,
                    "clock_regressions": 0,
                    "container_id": status.get("containerID"),
                    "restart_count": status.get("restartCount"),
                }
                self.metadata["segments"].append(self.segment)
                self.checkpoint(force=True)
                readers = []
                interruption = None
                try:
                    self.process = await asyncio.create_subprocess_exec(
                        *self.prefix,
                        "--request-timeout=0",
                        "logs",
                        self.pod["metadata"]["name"],
                        "-c",
                        "sglang",
                        "--follow",
                        "--timestamps",
                        "--tail=-1",
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    readers = [
                        asyncio.create_task(self.stdout_reader()),
                        asyncio.create_task(self.stderr_reader()),
                    ]
                    await asyncio.gather(*readers)
                    await self.process.wait()
                    interruption = self.stopping or "unexpected_stream_end"
                    if not self.stopping:
                        self.error(interruption)
                except Exception as error:
                    interruption = type(error).__name__
                    self.error(interruption)
                    self.fatal = True
                    for task in readers:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)
                    # A failed writer must not leave subprocess.wait blocked on
                    # an undrained PIPE while the child is being terminated.
                    drain = (
                        asyncio.create_task(self.process.communicate())
                        if self.process is not None
                        else None
                    )
                    await self.stop(interruption)
                    if drain is not None:
                        await drain
                finally:
                    if self.process is not None and self.process.returncode is None:
                        await self.stop(interruption or "collector_shutdown")
                    for task in readers:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)
                    self.segment.update(
                        ended_at=time.time(),
                        returncode=(
                            self.process.returncode
                            if self.process is not None
                            else None
                        ),
                        interruption=interruption or self.stopping,
                    )
                    self.checkpoint(force=True)
                    emit(
                        "follow_segment_closed",
                        segment_id=self.segment["segment_id"],
                        records=self.segment["records"],
                        interruption=self.segment["interruption"],
                    )
                    self.process = None
                if not self.stopping:
                    await asyncio.sleep(5)
        finally:
            self.checkpoint(force=True)
            self.output.close()
            self.stderr.close()


async def pod_snapshot(prefix):
    async def bounded_read(stream, maximum):
        result = bytearray()
        while chunk := await stream.read(65536):
            result.extend(chunk)
            if len(result) > maximum:
                raise ValueError("pod_list_response_size_limit")
        return bytes(result)

    process = await asyncio.create_subprocess_exec(
        *prefix,
        "--request-timeout=45s",
        "get",
        "pods",
        "-l",
        "app=glm53-topology-serving",
        "-o",
        "json",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    readers = [
        asyncio.create_task(bounded_read(process.stdout, 2 * 1024 * 1024)),
        asyncio.create_task(bounded_read(process.stderr, 65536)),
    ]
    try:
        stdout, _ = await asyncio.wait_for(asyncio.gather(*readers), timeout=65)
        await process.wait()
    finally:
        for task in readers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        if process.returncode is None:
            process.kill()
            await process.communicate()
    if process.returncode or len(stdout) > 2 * 1024 * 1024:
        raise RuntimeError("follow_pod_list_failed")
    return json.loads(stdout)["items"]


async def collect(args):
    root = args.campaign_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = [
        "kubectl",
        "--kubeconfig",
        str(args.kubeconfig.resolve()),
        "-n",
        "inf-glm53",
    ]
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    follower = task = None
    completed = set()
    report = {
        "collector_sha256": SOURCE_SHA,
        "started_at": time.time(),
        "api_errors": 0,
        "collector_errors": 0,
        "stopped_at": None,
        "stop_reason": None,
    }
    status_path = root / "follow-collector-status.json"
    try:
        while not stopped.is_set():
            try:
                pods = await pod_snapshot(prefix)
                candidates = []
                for pod in pods:
                    labels = pod["metadata"].get("labels", {})
                    profile = labels.get("glm53-profile")
                    if (
                        labels.get("app.kubernetes.io/part-of") != "glm53-stg-pp2-dpa"
                        or profile not in PROFILES
                    ):
                        continue
                    phase = (
                        "hicache"
                        if labels.get("glm53-hicache") == "true"
                        else args.phase
                    )
                    directory = root / f"{root.name}-{profile}-{phase}"
                    if directory.is_dir() and "running" in container_status(pod).get(
                        "state", {}
                    ):
                        candidates.append((pod, directory))
                if len(candidates) > 1:
                    raise RuntimeError("multiple_serving_pods")
                candidate = candidates[0] if candidates else None
                if follower is not None:
                    same = (
                        candidate is not None
                        and candidate[0]["metadata"]["uid"]
                        == follower.metadata["pod_uid"]
                    )
                    if same:
                        follower.pod = candidate[0]
                    if not same or task.done():
                        await follower.stop(
                            "pod_quiesced" if not same else "follower_finished"
                        )
                        await task
                        report["collector_errors"] += follower.metadata[
                            "collection_errors"
                        ]
                        completed.add(follower.metadata["pod_uid"])
                        follower = task = None
                if follower is None and candidate is not None:
                    pod, directory = candidate
                    if pod["metadata"]["uid"] not in completed:
                        follower = Follower(pod, directory, prefix, args.max_log_bytes)
                        task = asyncio.create_task(follower.run())
                        emit(
                            "follow_started",
                            profile=pod["metadata"]["labels"]["glm53-profile"],
                        )
                campaign = root / "campaign.json"
                if not pods and follower is None and campaign.exists():
                    rows = json.loads(campaign.read_text())
                    if isinstance(rows, list) and len(rows) == args.expected_profiles:
                        report["stop_reason"] = "campaign_quiesced"
                        break
            except Exception as error:
                report["api_errors"] += 1
                emit("follow_controller_error", type=type(error).__name__)
                if follower is not None and not task.done():
                    follower.error(type(error).__name__)
            atomic_json(status_path, report)
            try:
                await asyncio.wait_for(stopped.wait(), timeout=args.interval)
            except asyncio.TimeoutError:
                pass
    finally:
        if follower is not None:
            await follower.stop("collector_stop_requested")
            await task
            report["collector_errors"] += follower.metadata["collection_errors"]
        report.update(
            stopped_at=time.time(), stop_reason=report["stop_reason"] or "signal"
        )
        atomic_json(status_path, report)
        emit("follow_collector_stopped", **report)
    return 1 if report["api_errors"] or report["collector_errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("baseline", "repeat", "hicache"), default="baseline"
    )
    parser.add_argument("--expected-profiles", type=int, default=4)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--max-log-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    args = parser.parse_args()
    if (
        args.expected_profiles < 1
        or args.max_log_bytes < 1
        or not math.isfinite(args.interval)
        or args.interval <= 0
    ):
        parser.error("expected-profiles, max-log-bytes and interval must be positive")
    return asyncio.run(collect(args))


if __name__ == "__main__":
    raise SystemExit(main())
