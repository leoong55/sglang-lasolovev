"""Archive and remove terminal or suspended idle Jobs owned by this experiment."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--kubeconfig", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--keep-run-id", action="append", default=[])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
prefix = [
    "kubectl",
    "--kubeconfig",
    a.kubeconfig,
    "-n",
    "inf-glm53",
    "--request-timeout=45s",
]
owner = "glm53-stg-pp2-dpa"


def call(*args):
    return subprocess.run(
        [*prefix, *args], capture_output=True, timeout=60, check=True
    ).stdout


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def eligible(job):
    status = job.get("status", {})
    terminal = any(
        c.get("status") == "True" and c.get("type") in ("Complete", "Failed")
        for c in status.get("conditions", [])
    )
    return (
        job["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") == owner
        and not status.get("active", 0)
        and (terminal or job["spec"].get("suspend") is True)
    )


inventory = json.loads(
    call("get", "jobs,pods", "-l", "app.kubernetes.io/part-of=" + owner, "-o", "json")
)
save(a.output / "inventory.json", inventory)
jobs = [
    r
    for r in inventory["items"]
    if r["kind"] == "Job"
    and eligible(r)
    and not any(
        run in r["metadata"]["name"]
        or any(
            run in arg
            for c in r["spec"]["template"]["spec"]["containers"]
            for arg in c.get("command", [])
        )
        for run in a.keep_run_id
    )
]
pods = [r for r in inventory["items"] if r["kind"] == "Pod"]


def backup(job):
    name, uid = job["metadata"]["name"], job["metadata"]["uid"]
    directory = a.output / name
    directory.mkdir()
    save(directory / "job.json", job)
    rows, errors = [], []
    for pod in pods:
        if not any(
            r.get("kind") == "Job" and r.get("uid") == uid
            for r in pod["metadata"].get("ownerReferences", [])
        ):
            continue
        podname = pod["metadata"]["name"]
        save(directory / (podname + ".json"), pod)
        if pod.get("status", {}).get("phase") not in ("Succeeded", "Failed"):
            errors.append("pod_not_terminal:" + podname)
            continue
        if pod.get("status", {}).get("phase") == "Failed" and not pod.get(
            "status", {}
        ).get("containerStatuses"):
            rows.append(
                {
                    "pod": podname,
                    "logs_unavailable": "No container was created; admission failure is preserved in pod metadata",
                }
            )
            continue
        for container in pod["spec"]["containers"]:
            path = directory / (podname + "-" + container["name"] + ".log")
            with path.open("wb") as stream:
                result = subprocess.run(
                    [*prefix, "logs", podname, "-c", container["name"], "--timestamps"],
                    stdout=stream,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
            path.chmod(0o600)
            if result.returncode:
                path.with_suffix(".stderr").write_bytes(result.stderr)
                errors.append("log_read_failed:" + podname)
            rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "returncode": result.returncode,
                }
            )
    report = {"job": name, "uid": uid, "logs": rows, "errors": errors}
    save(directory / "backup.json", report)
    return report


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    backups = list(pool.map(backup, jobs))
save(a.output / "backups.json", backups)
print(
    json.dumps(
        {
            "archived_jobs": len(backups),
            "backup_errors": sum(bool(x["errors"]) for x in backups),
        }
    ),
    flush=True,
)
deleted, skipped = [], []
for report in backups:
    name, uid = report["job"], report["uid"]
    if report["errors"]:
        skipped.append({"job": name, "reason": "backup_errors"})
        continue
    current = json.loads(call("get", "job", name, "-o", "json"))
    if current["metadata"]["uid"] != uid or not eligible(current):
        skipped.append({"job": name, "reason": "identity_or_state_changed"})
        continue
    directory = a.output / name
    options = directory / "delete-options.json"
    save(
        options,
        {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Background",
            "preconditions": {"uid": uid},
        },
    )
    with (a.output / "mutations.jsonl").open("a") as journal:
        journal.write(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "action": "delete_archived_job",
                    "job": name,
                    "uid": uid,
                }
            )
            + "\n"
        )
        journal.flush()
        os.fsync(journal.fileno())
    response = call(
        "delete",
        "--raw=/apis/batch/v1/namespaces/inf-glm53/jobs/" + name,
        "-f",
        str(options),
    )
    (directory / "delete-response.json").write_bytes(response)
    deleted.append(name)
    if len(deleted) % 20 == 0:
        print(json.dumps({"deleted_jobs": len(deleted)}), flush=True)
save(
    a.output / "summary.json",
    {
        "deleted_jobs": deleted,
        "skipped_jobs": skipped,
        "scope": "Only experiment-owned terminal or suspended idle Jobs; PVCs and active workloads untouched",
    },
)
print(json.dumps({"deleted_jobs": len(deleted), "skipped_jobs": skipped}), flush=True)
