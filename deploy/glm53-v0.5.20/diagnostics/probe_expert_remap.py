"""Exact CUDA eager/graph validation of the candidate's logical/physical map.

Checks the real compiled TopK postprocess used by the model. This is an ID and
padding gate, not a full Humming arithmetic or model-quality comparison.
"""
import json
import os
import pathlib
import time

import torch
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

torch.cuda.set_device(0)
reset_context()
publish(ServerArgs(model_path='dummy'), role='tokenizer')
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.topk import _biased_grouped_topk_postprocess

mapping = json.loads(pathlib.Path('/scripts/expert-layout.json').read_text())['physical_to_logical_map']
rows = []
started = time.time()
for layer in [3, 41, 77]:
    assert sorted(mapping[layer]) == list(range(256))
    inverse = [mapping[layer].index(logical) for logical in range(256)]
    lookup = torch.tensor(inverse, dtype=torch.int64, device='cuda').reshape(256, 1)
    info = ExpertLocationDispatchInfo(
        ep_dispatch_algorithm='dynamic', rank_invariant=True,
        partial_logical_to_rank_dispatch_physical_map=None,
        partial_logical_to_all_physical_map=lookup,
        partial_logical_to_all_physical_map_num_valid=torch.ones(256, dtype=torch.int64, device='cuda'),
        num_physical_experts=256,
    )
    for n in [1, 257, 16384]:
        for valid_count in sorted(set([0, n, max(0, n - 3)])):
            cpu_ids = ((torch.arange(n * 8).reshape(n, 8) * 17 + layer) % 256).int()
            ids = cpu_ids.to('cuda')
            valid = torch.tensor(valid_count, dtype=torch.int32, device='cuda')
            expected = torch.tensor(inverse, dtype=torch.int32)[cpu_ids.long()]
            expected[valid_count:] = -1
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    actual = _biased_grouped_topk_postprocess(ids, info, valid)
            stream.synchronize()
            assert actual.dtype == torch.int32 and torch.equal(actual.cpu(), expected)
            rows.append(dict(layer=layer, rows=n, valid=valid_count, mode='eager', exact=True))
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    graph_output = _biased_grouped_topk_postprocess(ids, info, valid)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(graph_output.cpu(), expected)
            rows.append(dict(layer=layer, rows=n, valid=valid_count, mode='graph', exact=True))
            # Replay must read new logical IDs, rather than return captured values.
            changed = (cpu_ids + 1) % 256
            ids.copy_(changed.to('cuda'))
            graph.replay()
            torch.cuda.synchronize()
            expected_changed = torch.tensor(inverse, dtype=torch.int32)[changed.long()]
            expected_changed[valid_count:] = -1
            assert torch.equal(graph_output.cpu(), expected_changed)
            del graph

assert len(rows) == 48
path = pathlib.Path(os.environ['EXPERT_REMAP_PROBE_OUTPUT'])
path.write_text(json.dumps(dict(passed=len(rows), cases=rows, elapsed_s=time.time()-started,
                                gpu=torch.cuda.get_device_name(0),
                                scope='Exact compiled routing IDs/padding eager+graph on one H200; not full-model arithmetic'), indent=2))
print('EXPERT_REMAP_COMPLETE', len(rows), flush=True)
