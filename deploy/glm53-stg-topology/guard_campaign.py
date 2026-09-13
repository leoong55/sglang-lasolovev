"""Apply user-requested early screening and archive completed campaign Jobs."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROFILES = ("dpa2", "pp2", "dpa4", "dpa8")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def budget_decision(row, observed_at, budget):
    if not row.get("stage", "").endswith(("long-cold", "long-warm")):
        return None
    start = row.get("first_measured_started_at")
    end = (
        row.get("last_measured_finished_at")
        if row.get("verdict_present")
        else observed_at
    )
    if (
        not number(start)
        or not number(end)
        or end < start
        or row.get("malformed_records")
    ):
        return None
    elapsed = end - start
    if elapsed <= budget:
        return None
    return {
        "reason": "long_workload_exceeded_user_screening_budget",
        "stage": row["stage"],
        "observed_elapsed_seconds": elapsed,
        "budget_seconds": budget,
        "completed_measured": row["completed_measured"],
        "error_measured": row["error_measured"],
        "qualified_performance_measurement": False,
    }


def no_completion_decision(log, rows, now, budget):
    current = None
    for line in log.splitlines():
        try:
            timestamp = datetime.fromisoformat(
                line.split(" ", 1)[0].replace("Z", "+00:00")
            ).timestamp()
            value = json.loads(line[line.index("{") :])
        except (ValueError, IndexError):
            continue
        if value.get("event") == "benchmark_started":
            current = (timestamp, value)
        elif (
            value.get("event") == "benchmark_finished"
            and current
            and value.get("directory") == current[1].get("directory")
        ):
            current = None
    if not current or current[1].get("kind") not in ("long-cold", "long-warm"):
        return None
    timestamp, event = current
    directory = event.get("directory", "")
    stage = ("preparation/" if "/preparation/" in directory else "") + directory.rsplit(
        "/", 1
    )[-1]
    if any(
        r.get("stage") == stage and number(r.get("first_measured_started_at"))
        for r in rows
    ):
        return None
    if now - timestamp <= budget:
        return None
    return {
        "reason": "no_completed_measured_request_after_child_budget",
        "stage": stage,
        "observed_elapsed_seconds": now - timestamp,
        "budget_seconds": budget,
        "budget_includes_initialization_and_warmup": True,
        "completed_measured": 0,
        "qualified_performance_measurement": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kubeconfig", required=True)
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    prefix = [
        "kubectl",
        "--kubeconfig",
        a.kubeconfig,
        "-n",
        "inf-glm53",
        "--request-timeout=45s",
    ]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cleaned = set()

    def call(*args):
        return subprocess.check_output(
            [*prefix, *args], stderr=subprocess.PIPE, timeout=60
        )

    def record(event):
        event.update(timestamp=time.time(), guard_sha256=source_hash)
        with (a.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(event), flush=True)

    while True:
        try:
            policy = json.loads(a.policy.read_text())
            campaign = policy["campaign"]
            completed = (
                json.loads((a.campaign_root / "campaign.json").read_text())
                if (a.campaign_root / "campaign.json").exists()
                else []
            )
            finished = {
                r["profile"]
                for r in completed
                if r.get("results_path") and not r.get("export_error")
            }
            if finished - cleaned:
                target = a.output / ("cleanup-" + str(time.time_ns()))
                args = [
                    "python3",
                    str(Path(__file__).with_name("cleanup_completed_jobs.py")),
                    "--kubeconfig",
                    a.kubeconfig,
                    "--output",
                    str(target),
                ]
                for profile in set(PROFILES) - finished:
                    args += ["--keep-run-id", f"{campaign}-{profile}-baseline"]
                result = subprocess.run(args, capture_output=True, timeout=600)
                (a.output / (target.name + ".log")).write_bytes(
                    result.stdout + result.stderr
                )
                record(
                    {
                        "event": "completed_job_cleanup",
                        "profiles": sorted(finished),
                        "returncode": result.returncode,
                    }
                )
                if result.returncode == 0:
                    cleaned |= finished
            if len(completed) == len(PROFILES):
                watcher_name = "glm53-" + campaign + "-progress"
                raw = call(
                    "get", "job", watcher_name, "--ignore-not-found", "-o", "json"
                )
                if raw.strip():
                    watcher = json.loads(raw)
                    labels = watcher["metadata"].get("labels", {})
                    if (
                        labels.get("app.kubernetes.io/part-of") != "glm53-stg-pp2-dpa"
                        or labels.get("glm53-role") != "progress"
                    ):
                        raise ValueError("Refusing to remove an unowned progress Job")
                    (a.output / "final-progress-job.json").write_text(
                        json.dumps(watcher, indent=2)
                    )
                    (a.output / "final-progress.log").write_bytes(
                        call("logs", "job/" + watcher_name, "--timestamps")
                    )
                    options = a.output / "final-progress-delete.json"
                    options.write_text(
                        json.dumps(
                            {
                                "apiVersion": "v1",
                                "kind": "DeleteOptions",
                                "propagationPolicy": "Foreground",
                                "preconditions": {"uid": watcher["metadata"]["uid"]},
                            }
                        )
                    )
                    record(
                        {
                            "event": "delete_completed_campaign_watcher",
                            "job": watcher_name,
                            "uid": watcher["metadata"]["uid"],
                        }
                    )
                    (a.output / "final-progress-delete-response.json").write_bytes(
                        call(
                            "delete",
                            "--raw=/apis/batch/v1/namespaces/inf-glm53/jobs/"
                            + watcher_name,
                            "-f",
                            str(options),
                        )
                    )
                record(
                    {
                        "event": "campaign_guard_finished",
                        "reason": "all_profile_attempts_recorded",
                    }
                )
                return
            snapshot = json.loads(
                call("logs", "job/glm53-" + campaign + "-progress", "--tail=1")
            )
            now = time.time()
            if (
                not number(snapshot.get("timestamp"))
                or not 0 <= now - snapshot["timestamp"] <= 90
            ):
                raise ValueError("Progress snapshot is stale or future dated")
            (a.output / "latest-progress.json").write_text(
                json.dumps(snapshot, indent=2) + "\n"
            )
            jobs = json.loads(
                call(
                    "get",
                    "jobs",
                    "-l",
                    "app.kubernetes.io/part-of=glm53-stg-pp2-dpa",
                    "-o",
                    "json",
                )
            )["items"]
            for job in jobs:
                name, uid = job["metadata"]["name"], job["metadata"]["uid"]
                allowed = {
                    f"glm53-{campaign}-{profile}-baseline{suffix}"
                    for profile in PROFILES
                    for suffix in ("", "-prep")
                }
                if name not in allowed or not job.get("status", {}).get("active"):
                    continue
                destination = a.output / uid
                if (destination / "delete-response.json").exists():
                    continue
                rows = [r for r in snapshot["workloads"] if r.get("job") == name]
                decisions = [
                    budget_decision(
                        r, snapshot["timestamp"], policy["long_measured_budget_seconds"]
                    )
                    for r in rows
                ]
                decision = next((d for d in decisions if d), None)
                if not decision:
                    decision = no_completion_decision(
                        call("logs", "job/" + name, "--timestamps").decode(),
                        rows,
                        now,
                        policy["zero_completed_child_budget_seconds"],
                    )
                if not decision:
                    continue
                current = json.loads(call("get", "job", name, "-o", "json"))
                if current["metadata"]["uid"] != uid or not current.get(
                    "status", {}
                ).get("active"):
                    continue
                destination.mkdir(exist_ok=True)
                (destination / "job.json").write_text(
                    json.dumps(current, indent=2) + "\n"
                )
                (destination / "job.log").write_bytes(
                    call("logs", "job/" + name, "--timestamps")
                )
                decision.update(
                    job=name,
                    uid=uid,
                    classification="operator_screening_stop_not_runtime_failure",
                )
                (destination / "decision.json").write_text(
                    json.dumps(decision, indent=2) + "\n"
                )
                options = destination / "delete-options.json"
                options.write_text(
                    json.dumps(
                        {
                            "apiVersion": "v1",
                            "kind": "DeleteOptions",
                            "propagationPolicy": "Foreground",
                            "preconditions": {"uid": uid},
                        }
                    )
                )
                record({"event": "stop_job_for_screening_budget", **decision})
                response = call(
                    "delete",
                    "--raw=/apis/batch/v1/namespaces/inf-glm53/jobs/" + name,
                    "-f",
                    str(options),
                )
                (destination / "delete-response.json").write_bytes(response)
        except Exception as error:
            record(
                {
                    "event": "guard_check_error",
                    "type": type(error).__name__,
                    "detail": str(error)[:250],
                }
            )
        time.sleep(20)


if __name__ == "__main__":
    main()
