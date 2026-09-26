"""Bounded ABBA test, preserving logs/config; no credentials are exported."""
import hashlib
import json
import os
import pathlib
import subprocess
import time

root = pathlib.Path('/results/prefill16k-20260926/nvls-probe-r1')
root.mkdir(exist_ok=False)
script = pathlib.Path('/scripts/probe_nccl_nvls.py')
(root/'probe.py').write_bytes(script.read_bytes())
(root/'topology.txt').write_bytes(subprocess.check_output(['nvidia-smi', 'topo', '-m']))
(root/'gpus.csv').write_bytes(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.free', '--format=csv']))
config = dict(NCCL_CUMEM_ENABLE='0', NCCL_GRAPH_MIXING_SUPPORT='0', CUDA_DEVICE_MAX_CONNECTIONS='8', NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,TUNING,NVLS', OMP_NUM_THREADS='1')
(root/'config.json').write_text(json.dumps(dict(fixed=config, order=['0a','1a','1b','0b'], script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), scope='PyNccl BF16 EP8-shaped collectives; not model performance'), indent=2))
for tag in ['0a','1a','1b','0b']:
    env = os.environ.copy()
    env.update(config, NCCL_NVLS_ENABLE=tag[0], PROBE_TAG=tag, PROBE_OUTPUT=str(root/(tag+'.json')))
    print('BEGIN', tag, flush=True)
    started = time.time()
    with (root/(tag+'.log')).open('wb') as f:
        result = subprocess.run(['torchrun','--standalone','--nproc-per-node=8',str(script)], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=240)
    (root/(tag+'-status.json')).write_text(json.dumps(dict(returncode=result.returncode, elapsed_seconds=time.time()-started)))
    if result.returncode:
        print('PROBE_FAILED', tag, flush=True)
        raise SystemExit(result.returncode)
    print('DONE', tag, flush=True)
(root/'complete.json').write_text(json.dumps(dict(completed_at=time.time(), runs=4)))
