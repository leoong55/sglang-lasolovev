"""Run one serving process group and report GPU/cgroup telemetry without exec."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def emit(kind, **values):
    print(kind + " " + json.dumps({"timestamp": time.time(), **values}), flush=True)


def memory_counters():
    values, errors = {}, {}
    for field, filename in (
        ("cgroup_memory_bytes", "memory.current"),
        ("cgroup_memory_peak_bytes", "memory.peak"),
        ("cgroup_memory_limit_bytes", "memory.max"),
    ):
        try:
            value = Path("/sys/fs/cgroup", filename).read_text().strip()
            values[field] = None if value == "max" else int(value)
            if values[field] is not None and values[field] < 0:
                raise ValueError("negative memory counter")
        except (OSError, ValueError) as error:
            values[field] = None
            errors[field] = type(error).__name__
    return {**values, "cgroup_memory_counter_errors": errors}


def collect(stop):
    while not stop.is_set():
        try:
            p = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,memory.used,memory.total,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=8,
            )
            emit(
                "GLM53_GPU",
                rows=p.stdout.strip().splitlines(),
                returncode=p.returncode,
                **memory_counters(),
            )
        except Exception as e:
            emit("GLM53_GPU_ERROR", error=str(e))
        stop.wait(2)


def main():
    argv = [sys.executable, "/opt/glm53-stg-topology/launch.py", *sys.argv[1:]]
    emit(
        "GLM53_PROCESS",
        argv=argv,
        source_commit=os.environ.get("SOURCE_COMMIT"),
        image=os.environ.get("SERVING_IMAGE"),
        pod_uid=os.environ.get("POD_UID"),
    )
    proc = subprocess.Popen(argv, start_new_session=True)
    stop = threading.Event()

    def terminate(signum, frame):
        stop.set()
        if proc.poll() is None:
            os.killpg(proc.pid, signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    threading.Thread(target=collect, args=(stop,), daemon=True).start()
    rc = proc.wait()
    stop.set()
    emit("GLM53_PROCESS_EXIT", returncode=rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
