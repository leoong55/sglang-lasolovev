"""Isolated PyNccl probe. No model traffic; environment differs only in NVLS."""
import datetime
import json
import os
import pathlib
import statistics

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
assert dist.get_world_size() == 8
comm = PyNcclCommunicator(dist.group.WORLD, device=rank)
assert comm.available
stream = torch.cuda.Stream()
rows = []

def check(op, output, n):
    if op == 'reduce_scatter':
        assert torch.all(output == 36).item(), 'reduce-scatter numeric mismatch'
    else:
        for r in range(8):
            assert torch.all(output[r * (n // 8):(r + 1) * (n // 8)] == r + 1).item(), 'all-gather numeric mismatch'

for n in [4096, 8192, 16384]:
    for op in ['reduce_scatter', 'all_gather']:
        inp = torch.full((n if op == 'reduce_scatter' else n // 8, 6144), rank + 1, dtype=torch.bfloat16, device='cuda')
        out = torch.empty((n // 8 if op == 'reduce_scatter' else n, 6144), dtype=torch.bfloat16, device='cuda')
        torch.cuda.synchronize()
        call = getattr(comm, op)
        with comm.change_state(enable=True), torch.cuda.stream(stream):
            for _ in range(5):
                call(out, inp)
        stream.synchronize()
        check(op, out, n)
        dist.barrier()
        with comm.change_state(enable=True), torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(10):
                    call(out, inp)
        graph.replay()
        torch.cuda.synchronize()
        check(op, out, n)
        for mode in ['eager', 'graph']:
            timings = []
            for _ in range(5):
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with comm.change_state(enable=True), torch.cuda.stream(stream):
                    start.record()
                    if mode == 'graph':
                        for _ in range(3):
                            graph.replay()
                    else:
                        for _ in range(30):
                            call(out, inp)
                    end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) / 30)
            check(op, out, n)
            rows.append(dict(n=n, hidden=6144, op=op, mode=mode, per_call_ms=timings, median_ms=statistics.median(timings), correct=True))
        del graph, inp, out
        torch.cuda.synchronize()
        if rank == 0:
            print('PROBE', os.environ['PROBE_TAG'], n, op, 'done', flush=True)

payload = dict(rank=rank, gpu=torch.cuda.get_device_name(rank), torch=torch.__version__, nccl=comm.nccl.ncclGetVersion(), nvls=os.environ['NCCL_NVLS_ENABLE'], rows=rows)
all_rows = [None] * 8
dist.all_gather_object(all_rows, payload)
if rank == 0:
    path = pathlib.Path(os.environ['PROBE_OUTPUT'])
    path.write_text(json.dumps(dict(tag=os.environ['PROBE_TAG'], ranks=all_rows), indent=2))
    print('PROBE_COMPLETE', path.name, flush=True)
dist.barrier()
dist.destroy_process_group()
