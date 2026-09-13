"""Run one serving process group and report GPU/cgroup telemetry without exec."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def emit(kind, **values):
    print(kind + ' ' + json.dumps({'timestamp': time.time(), **values}), flush=True)


def collect(stop):
    while not stop.is_set():
        try:
            p = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=8)
            memory = Path('/sys/fs/cgroup/memory.current')
            emit('GLM53_GPU', rows=p.stdout.strip().splitlines(), returncode=p.returncode,
                 cgroup_memory_bytes=int(memory.read_text()) if memory.exists() else None)
        except Exception as e:
            emit('GLM53_GPU_ERROR', error=str(e))
        stop.wait(2)


def main():
    argv = [sys.executable, '/opt/glm53-stg-topology/launch.py', *sys.argv[1:]]
    emit('GLM53_PROCESS', argv=argv, source_commit=os.environ.get('SOURCE_COMMIT'), image=os.environ.get('SERVING_IMAGE'), pod_uid=os.environ.get('POD_UID'))
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
    emit('GLM53_PROCESS_EXIT', returncode=rc)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
