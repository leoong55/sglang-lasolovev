"""Run the authorized isolated matrix using ordinary Kubernetes API operations.

Every mutation is preceded by a durable local intent record. Failed profiles are
recorded, their GPUs released, and independent profiles continue. No exec/cp or
streaming API is used. No default Kubernetes context is used.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import tarfile
import time
from collections import deque
from pathlib import Path

import render

SERVING_CRITICAL_FILES = (
    "Dockerfile",
    "install.py",
    "manifest.json",
    "launch.py",
    "observe.py",
    "sitecustomize.py",
    "supervise.py",
    "render.py",
)


def verify_revisions(repository, serving_commit, tooling_commit):
    """Keep model execution fixed while independently versioning the client."""
    for value in (serving_commit, tooling_commit):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("Both source revisions must be full commit SHAs")
    package = "deploy/glm53-stg-topology"
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", package],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    if untracked.stdout.strip():
        raise RuntimeError(
            "Untracked experiment files prevent a complete tooling revision proof"
        )
    checks = [
        (
            tooling_commit,
            [package],
            "Local experiment scripts differ from tooling commit",
        ),
        (
            serving_commit,
            [f"{package}/{name}" for name in SERVING_CRITICAL_FILES],
            "Serving-critical files differ from the image source commit",
        ),
    ]
    for revision, paths, message in checks:
        checked = subprocess.run(
            ["git", "diff", "--quiet", revision, "--", *paths], cwd=repository
        )
        if checked.returncode:
            raise RuntimeError(message)
    return {
        name: hashlib.sha256((repository / package / name).read_bytes()).hexdigest()
        for name in SERVING_CRITICAL_FILES
    }


def benchmark_job(args, run_id, profile, purpose, repetitions=1):
    suffix = {"admission": "-smoke", "preparation": "-prep", "measurement": ""}[purpose]
    obj = render.benchmark(
        run_id + suffix,
        profile,
        args.image,
        args.commit,
        args.node,
        repetitions,
        mode="smoke" if purpose == "admission" else "suite",
    )
    spec = obj["spec"]["template"]["spec"]
    container = spec["containers"][0]
    directory = "/results/" + run_id
    if purpose != "measurement":
        directory += "/" + purpose
    command = container["command"]
    command[command.index("--results-dir") + 1] = directory
    container["env"] += [
        {"name": "TOOLING_COMMIT", "value": args.tooling_commit},
        {"name": "SCRIPT_CONFIG_SHA256", "value": args.configmap_sha256},
        {"name": "BENCHMARK_PURPOSE", "value": purpose},
        {"name": "COMPILE_CACHE_PATH", "value": f"/cache/{args.commit[:12]}/{profile}"},
    ]
    container["volumeMounts"].append(
        {"name": "compile-cache", "mountPath": "/cache", "readOnly": True}
    )
    spec["volumes"].append(
        {
            "name": "compile-cache",
            "persistentVolumeClaim": {
                "claimName": "glm53-topology-compile-cache",
                "readOnly": True,
            },
        }
    )
    return obj


def serving_objects(args, profile, phase):
    objects = render.serving(
        profile, args.image, args.commit, args.node, hicache=phase == "hicache"
    )
    template = objects[0]["spec"]["template"]
    container = template["spec"]["containers"][0]
    # Native /health requires a generated token within20 seconds. A legitimate
    # first long prefill can exceed that while workers continue making progress.
    # Keep generated startup readiness, then check HTTP API availability only.
    # Worker watchdogs, admission and full workload validation remain mandatory.
    container["readinessProbe"]["httpGet"]["path"] = "/model_info"
    template["metadata"].setdefault("annotations", {})[
        "glm53-readiness-policy"
    ] = "api-after-generated-startup"
    return objects


class Cluster:
    def __init__(self, kubeconfig, output):
        self.prefix = [
            "kubectl",
            "--kubeconfig",
            str(Path(kubeconfig).resolve()),
            "--namespace",
            render.NAMESPACE,
            "--request-timeout=45s",
        ]
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seen = set()
        self.recent = deque()
        self.expected_uid = None
        self.interrupted = False

    def intent(self, action, **details):
        row = {"timestamp": time.time(), "action": action, **details}
        with (self.output / "operations.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row), flush=True)

    def call(self, *args, data=None, check=True):
        p = subprocess.run(
            [*self.prefix, *args],
            input=data,
            text=True,
            capture_output=True,
            timeout=75,
        )
        if check and p.returncode:
            raise RuntimeError(
                "kubectl " + " ".join(args[:4]) + ": " + p.stderr[-3000:]
            )
        return p

    def get(self, kind, name=None, selector=None):
        argv = ["get", kind]
        if name:
            argv.append(name)
        if selector:
            argv += ["-l", selector]
        argv += ["-o", "json"]
        p = self.call(*argv)
        return json.loads(p.stdout)

    def apply(self, objects):
        for obj in objects:
            name = obj["metadata"]["name"]
            old = self.call(
                "get", obj["kind"], name, "-o", "json", "--ignore-not-found"
            )
            if old.stdout.strip():
                labels = json.loads(old.stdout)["metadata"].get("labels", {})
                if labels.get("app.kubernetes.io/part-of") != render.PART:
                    raise RuntimeError(
                        "Refusing to overwrite unowned resource: " + name
                    )
        self.intent(
            "apply",
            resources=[o["kind"] + "/" + o["metadata"]["name"] for o in objects],
        )
        print(
            self.call(
                "apply",
                "-f",
                "-",
                data=json.dumps({"apiVersion": "v1", "kind": "List", "items": objects}),
            ).stdout,
            flush=True,
        )

    def stop(self):
        p = self.call(
            "get", "deployment", render.SERVICE, "--ignore-not-found", "-o", "json"
        )
        if p.stdout.strip():
            obj = json.loads(p.stdout)
            if (
                obj["metadata"].get("labels", {}).get("app.kubernetes.io/part-of")
                != render.PART
            ):
                raise RuntimeError("Refusing to scale unowned deployment")
            self.intent("release_gpus", deployment=render.SERVICE)
            self.call("scale", "deployment", render.SERVICE, "--replicas=0")
        # Deleting a Deployment can leave its terminating pods behind. Still
        # wait for those GPUs to be released before applying the next profile.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            pods = self.get("pods", selector="app=" + render.SERVICE)["items"]
            if not pods:
                return
            time.sleep(5)
        raise TimeoutError(
            "Serving pods did not terminate; refusing overlapping GPU profiles"
        )

    def abort_job(self, name):
        result = self.call("get", "job", name, "--ignore-not-found", "-o", "json")
        if not result.stdout.strip():
            return
        obj = json.loads(result.stdout)
        if (
            obj["metadata"].get("labels", {}).get("app.kubernetes.io/part-of")
            != render.PART
        ):
            raise RuntimeError("Refusing to abort unowned job")
        if any(
            c.get("status") == "True" and c.get("type") in ("Complete", "Failed")
            for c in obj.get("status", {}).get("conditions", [])
        ):
            return
        self.intent("abort_job", job=name)
        self.call("delete", "job", name, "--wait=false")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if not self.get("pods", selector="job-name=" + name)["items"]:
                return
            time.sleep(5)
        raise TimeoutError("Aborted benchmark pods did not terminate")

    def validate_serving_generation(self, pods):
        if len(pods) != 1 or pods[0]["metadata"]["uid"] != self.expected_uid:
            raise RuntimeError("Serving generation changed during benchmark")
        status = pods[0].get("status", {})
        if any(
            c.get("restartCount", 0) or "terminated" in c.get("state", {})
            for c in status.get("containerStatuses", [])
        ):
            raise RuntimeError("Serving restarted or terminated during benchmark")
        if not any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in status.get("conditions", [])
        ):
            raise RuntimeError("Serving lost readiness during benchmark")

    def collect(self, target):
        target.mkdir(parents=True, exist_ok=True)
        pods = self.get("pods", selector="app=" + render.SERVICE)["items"]
        for pod in pods:
            name = pod["metadata"]["name"]
            (target / (name + ".json")).write_text(json.dumps(pod, indent=2))
            log = self.call(
                "logs", name, "-c", "sglang", "--timestamps", "--since=90s", check=False
            )
            with (target / "server.log").open("a") as f:
                for line in log.stdout.splitlines():
                    key = hashlib.sha256((name + line).encode()).digest()
                    if key not in self.seen:
                        f.write(line + "\n")
                        self.seen.add(key)
                        self.recent.append(key)
                    if len(self.recent) > 30000:
                        self.seen.remove(self.recent.popleft())
        events = self.get("events")
        events["items"] = [
            e
            for e in events["items"]
            if e.get("involvedObject", {}).get("name", "").startswith("glm53-")
        ]
        (target / "events.json").write_text(json.dumps(events, indent=2))
        return pods

    def ready(self, target):
        deadline = time.monotonic() + 7200
        last = None
        while time.monotonic() < deadline:
            pods = self.collect(target)
            status = [
                (
                    p["metadata"]["name"],
                    p.get("status", {}).get("phase"),
                    [
                        (
                            c["name"],
                            c.get("ready"),
                            c.get("restartCount", 0),
                            c.get("state"),
                        )
                        for c in p.get("status", {}).get("containerStatuses", [])
                    ],
                )
                for p in pods
            ]
            if status != last:
                print("SERVING_STATUS " + json.dumps(status), flush=True)
                last = status
            for pod in pods:
                st = pod.get("status", {})
                if any(
                    c.get("restartCount", 0) > 0 or "terminated" in c.get("state", {})
                    for c in st.get("containerStatuses", [])
                ):
                    raise RuntimeError(
                        "Serving container failed or restarted before readiness"
                    )
                if st.get("phase") == "Failed":
                    raise RuntimeError(
                        "Serving pod failed admission: " + st.get("message", "")
                    )
                if any(
                    c.get("type") == "Ready" and c.get("status") == "True"
                    for c in st.get("conditions", [])
                ):
                    return pod
            time.sleep(20)
        raise TimeoutError("Serving readiness deadline reached")

    def wait_job(self, job, target=None, timeout=43200):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            obj = self.get("job", job)
            status = obj.get("status", {})
            if status != last:
                print(
                    "JOB_STATUS " + json.dumps({"job": job, "status": status}),
                    flush=True,
                )
                last = status
            if target is not None:
                self.validate_serving_generation(self.collect(target))
                (target / (job + ".json")).write_text(json.dumps(obj, indent=2))
            terminal = next(
                (
                    c
                    for c in status.get("conditions", [])
                    if c.get("status") == "True" and c["type"] in ("Complete", "Failed")
                ),
                None,
            )
            if terminal:
                logs = self.call("logs", "job/" + job, check=False)
                if target is not None:
                    (target / (job + ".log")).write_text(
                        logs.stdout + "\n" + logs.stderr
                    )
                return terminal["type"] == "Complete", logs.stdout
            time.sleep(15)
        raise TimeoutError("Job deadline reached: " + job)

    def export(self, run_id, node):
        # Each completed Job logs at most one small part, avoiding kubelet log
        # rotation. Export jobs never mount credentials or contact the internet.
        name = run_id + "-exm"
        obj = render.job(
            name, node, ["/scripts/export.py", "--run-id", run_id], role="export"
        )
        self.apply([obj])
        ok, logs = self.wait_job(obj["metadata"]["name"], timeout=1800)
        if not ok:
            raise RuntimeError("Export manifest job failed: " + logs[-2000:])
        manifest = next(
            json.loads(s.split(" ", 1)[1])
            for s in logs.splitlines()
            if s.startswith("GLM53_EXPORT_MANIFEST ")
        )
        archive = self.output / (run_id + ".tar.gz")
        with archive.open("wb") as f:
            for i, expected in enumerate(manifest["parts"]):
                obj = render.job(
                    run_id + "-ex" + str(i),
                    node,
                    ["/scripts/export.py", "--run-id", run_id, "--part", str(i)],
                    role="export",
                )
                self.apply([obj])
                ok, logs = self.wait_job(obj["metadata"]["name"], timeout=1800)
                if not ok:
                    raise RuntimeError("Export part job failed")
                row = next(
                    json.loads(s.split(" ", 1)[1])
                    for s in logs.splitlines()
                    if s.startswith("GLM53_EXPORT_PART ")
                )
                data = base64.b64decode(row["data"], validate=True)
                if row["index"] != i or hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Export part checksum mismatch")
                f.write(data)
        if (
            archive.stat().st_size != manifest["bytes"]
            or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]
        ):
            raise ValueError("Export archive checksum mismatch")
        (self.output / (run_id + "-export.json")).write_text(
            json.dumps(manifest, indent=2)
        )
        with tarfile.open(archive) as tar:
            tar.extractall(self.output / "results", filter="data")
        return self.output / "results" / run_id


def run_profile(cluster, args, profile, phase, repetitions):
    run_id = args.campaign + "-" + profile + "-" + phase
    evidence = cluster.output / run_id
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    row = {
        "run_id": run_id,
        "profile": profile,
        "phase": phase,
        "purpose": "measurement",
        "measurement_job_started": False,
        "repetitions": repetitions,
        "started": started,
        "source_commit": args.commit,
        "tooling_commit": args.tooling_commit,
        "configmap_sha256": args.configmap_sha256,
        "serving_critical_sha256": args.serving_critical_sha256,
        "preparation_skipped": args.skip_preparation,
        "readiness_policy": "API model_info after generated startup; worker watchdog, admission and workload validation retained",
        "image": args.image,
        "node": args.node,
    }
    job_started = False
    jobs = []
    try:
        cluster.stop()
        objects = serving_objects(args, profile, phase)
        (evidence / "serving-manifest.json").write_text(json.dumps(objects, indent=2))
        cluster.apply(objects)
        pod = cluster.ready(evidence)
        row["pod_uid"] = pod["metadata"]["uid"]
        cluster.expected_uid = row["pod_uid"]
        smoke = benchmark_job(args, run_id, profile, "admission")
        (evidence / "smoke-manifest.json").write_text(json.dumps(smoke, indent=2))
        cluster.apply([smoke])
        jobs.append(smoke["metadata"]["name"])
        job_started = True
        ok, _ = cluster.wait_job(smoke["metadata"]["name"], evidence)
        if not ok:
            raise RuntimeError("Correctness or long-context admission smoke failed")
        if not args.skip_preparation:
            preparation = benchmark_job(args, run_id, profile, "preparation")
            (evidence / "preparation-manifest.json").write_text(
                json.dumps(preparation, indent=2)
            )
            cluster.apply([preparation])
            jobs.append(preparation["metadata"]["name"])
            ok, _ = cluster.wait_job(preparation["metadata"]["name"], evidence)
            row["preparation_job_success"] = ok
            if not ok:
                raise RuntimeError("Full workload preparation failed")
        obj = benchmark_job(args, run_id, profile, "measurement", repetitions)
        (evidence / "benchmark-manifest.json").write_text(json.dumps(obj, indent=2))
        row["measurement_job_started"] = True
        cluster.apply([obj])
        jobs.append(obj["metadata"]["name"])
        job_started = True
        ok, _ = cluster.wait_job(obj["metadata"]["name"], evidence)
        row["job_success"] = ok
        row["status"] = "measured" if ok else "benchmark_failed"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    finally:
        try:
            cluster.collect(evidence)
        except Exception as exc:
            row["final_collection_error"] = str(exc)
        cleanup_errors = []
        for name in jobs:
            try:
                cluster.abort_job(name)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        try:
            cluster.stop()
        except Exception as exc:
            cleanup_errors.append(str(exc))
        if cleanup_errors:
            row["cleanup_error"] = "; ".join(cleanup_errors)
        row["finished"] = time.time()
        (evidence / "run-state.json").write_text(json.dumps(row, indent=2))
    if "cleanup_error" in row:
        raise RuntimeError(
            "GPU cleanup failed; refusing to continue: " + row["cleanup_error"]
        )
    if job_started:
        try:
            row["results_path"] = str(cluster.export(run_id, args.node))
        except Exception as exc:
            row["export_error"] = str(exc)
    (evidence / "run-state.json").write_text(json.dumps(row, indent=2))
    print("PROFILE_RESULT " + json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kubeconfig", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument(
        "--tooling-commit",
        help="Defaults to serving commit; serving-critical files must match both revisions",
    )
    p.add_argument(
        "--skip-preparation",
        action="store_true",
        help="Only reuse preparation already verified for this profile, image and cache mode",
    )
    p.add_argument(
        "--campaign",
        required=True,
        help="Unique short lowercase identifier, e.g. s0913a",
    )
    p.add_argument(
        "--node",
        required=True,
        help="Verified GPU node name; not stored in the source repository",
    )
    p.add_argument(
        "--profiles",
        nargs="+",
        choices=["pp2", "dpa2", "dpa4", "dpa8"],
        default=["pp2", "dpa2", "dpa4", "dpa8"],
    )
    p.add_argument(
        "--phase", choices=["baseline", "repeat", "hicache"], default="baseline"
    )
    p.add_argument("--repetitions", type=int, default=1)
    args = p.parse_args()
    args.tooling_commit = args.tooling_commit or args.commit
    cluster = Cluster(args.kubeconfig, args.output)
    repository = Path(__file__).resolve().parents[2]
    args.serving_critical_sha256 = verify_revisions(
        repository, args.commit, args.tooling_commit
    )

    def interrupted(signum, frame):
        cluster.interrupted = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise RuntimeError("Experiment interrupted; canceling jobs and releasing GPUs")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    cm = render.configmap()
    args.configmap_sha256 = cm["metadata"]["annotations"]["glm53-script-sha256"]
    (cluster.output / "scripts-configmap.json").write_text(json.dumps(cm, indent=2))
    cluster.apply([cm, *render.storage()])
    rows = []
    for profile in args.profiles:
        rows.append(run_profile(cluster, args, profile, args.phase, args.repetitions))
        (cluster.output / "campaign.json").write_text(json.dumps(rows, indent=2))
        if cluster.interrupted:
            return 130
    return (
        0
        if all(r["status"] == "measured" and "export_error" not in r for r in rows)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
