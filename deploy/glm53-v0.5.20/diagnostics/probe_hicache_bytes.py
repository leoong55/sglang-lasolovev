"""Exact DMA roundtrip on actual GLM HiCache pool classes; no model weights.

This verifies transfer bytes and local DCP/sidecar index mapping. It does NOT
validate the scheduler's cache tree, distributed consensus, or model logits.
Run only in a campaign Job which has received an exclusive GPU allocation.
"""
import gc
import json
import os
import pathlib
import time

import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

OUT = pathlib.Path(os.environ['HICACHE_PROBE_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
torch.cuda.set_device(0)
reset_context()
publish(ServerArgs(model_path='dummy'), role='tokenizer')
PAGE, DCP, LAYERS = 64, 4, 6
LOGICAL_PAGE = PAGE * DCP
SIZE = 4096
rows = []


def indices(pages, page_size=LOGICAL_PAGE):
    return torch.cat([torch.arange(p * page_size, (p + 1) * page_size)
                      for p in pages]).long()


def bytes_per_row(buffer):
    return buffer.view(torch.uint8).reshape(buffer.shape[0], -1)


def compare(buffer, idx, expected, label):
    got = bytes_per_row(buffer)[idx.to(buffer.device)].cpu()
    assert torch.equal(got, expected), f'byte mismatch: {label}'


started = time.time()
for rank in range(DCP):
    for sparse in (False, True):
        # Reduced layer count; actual packed FP8 MLA, DSA and full draft shapes.
        # Both dense and indexShare placeholder layouts exercise producer maps.
        target = DSATokenToKVPool(
            size=SIZE, page_size=PAGE, kv_lora_rank=512,
            dtype=torch.float8_e4m3fn, qk_rope_head_dim=64, layer_num=LAYERS,
            device='cuda', index_head_dim=128, enable_memory_saver=False,
            kv_cache_dim=656, index_buf_size=(SIZE + PAGE) * DCP - PAGE,
            skip_topk_layers=[False, True, True, False, True, False]
            if sparse else [False] * LAYERS,
        )
        host = MLATokenToKVPoolHost(
            target, host_to_device_ratio=2, host_size=0,
            page_size=LOGICAL_PAGE, layout='layer_first',
            override_kv_cache_dim=656, dcp_size=DCP, dcp_rank=rank,
        )
        index_host = DSAIndexerPoolHost(target, host, 'layer_first')
        draft = MHATokenToKVPool(
            size=(SIZE + PAGE) * DCP - PAGE, page_size=PAGE,
            dtype=torch.bfloat16, head_num=1, head_dim=128,
            layer_num=6, device='cuda', enable_memory_saver=False,
        )
        draft_host = MHATokenToKVPoolHost(
            draft, host_to_device_ratio=host.logical_size / draft.size,
            host_size=0, page_size=LOGICAL_PAGE, layout='layer_first',
            pool_label='draft',
        )
        families = [
            ('mla', target.kv_buffer, host.data_refs),
            ('indexer', index_host.packed_device_index_buffers,
             index_host.index_k_data_refs),
            ('draft', draft.k_buffer + draft.v_buffer,
             draft_host.k_data_refs + draft_host.v_data_refs),
        ]
        for n, src_pages, host_pages, dst_pages in (
            (1, [1], [11], [31]),
            (3, [7, 2, 9], [11, 3, 8], [13, 10, 6]),
            (16, list(range(1, 17)), list(range(31, 15, -1)), list(range(33, 49))),
        ):
            src, h, dst = indices(src_pages), indices(host_pages), indices(dst_pages)
            # The production direct controller sorts host and pairs device rows.
            h, order = h.sort()
            src, dst = src[order], dst[order]
            producer, backup, restore = [torch.cuda.Stream() for _ in range(3)]
            produced, copied = torch.cuda.Event(), torch.cuda.Event()
            saved = []
            with torch.cuda.stream(producer):
                torch.manual_seed(1000 + rank * 100 + n + int(sparse))
                for name, buffers, _ in families:
                    si = (maybe_dcp_kernel_indices(src, DCP, rank) if name == 'mla'
                          else src[::PAGE] // PAGE if name == 'indexer' else src)
                    expected = []
                    for buffer in buffers:
                        raw = bytes_per_row(buffer)
                        raw.random_(0, 256)
                        expected.append(raw[si.to('cuda')].cpu())
                    saved.append(expected)
                produced.record()
            with torch.cuda.stream(backup):
                backup.wait_event(produced)
                host.backup_from_device_all_layer(target, h, src, 'direct')
                index_host.backup_from_device_all_layer(target, h, src, 'direct')
                draft_host.backup_from_device_all_layer(draft, h, src, 'direct')
                copied.record()
            copied.synchronize()
            checked_bytes = 0
            for (name, buffers, host_buffers), expected in zip(families, saved):
                hi = (maybe_dcp_kernel_indices(h, DCP, rank) if name == 'mla'
                      else h[::PAGE] // PAGE if name == 'indexer' else h)
                for i, (buffer, hb, want) in enumerate(zip(buffers, host_buffers, expected)):
                    compare(hb, hi, want, f'{name} D2H layer{i}')
                    checked_bytes += want.numel()
            with torch.cuda.stream(restore):
                restore.wait_event(copied)
                # Destroy all original device content to make stale hits fail.
                for _, buffers, _ in families:
                    for buffer in buffers:
                        bytes_per_row(buffer).fill_(17)
                for layer in range(LAYERS):
                    host.load_to_device_per_layer(target, h, dst, layer, 'direct')
                    index_host.load_to_device_per_layer(target, h, dst, layer, 'direct')
                    draft_host.load_to_device_per_layer(draft, h, dst, layer, 'direct')
            restore.synchronize()
            for (name, buffers, _), expected in zip(families, saved):
                di = (maybe_dcp_kernel_indices(dst, DCP, rank) if name == 'mla'
                      else dst[::PAGE] // PAGE if name == 'indexer' else dst)
                for i, (buffer, want) in enumerate(zip(buffers, expected)):
                    compare(buffer, di, want, f'{name} H2D layer{i}')
                    raw = bytes_per_row(buffer)
                    mask = torch.ones(raw.shape[0], dtype=torch.bool, device='cuda')
                    mask[di.to('cuda')] = False
                    assert torch.all(raw[mask] == 17).item(), f'{name} wrote outside destination'
            row = dict(dcp_rank=rank, sparse_indexer=sparse, logical_pages=n,
                       logical_tokens=n * LOGICAL_PAGE, roundtrip_bytes=checked_bytes,
                       exact_bytes=True, untouched_rows=True)
            rows.append(row)
            (OUT/'progress.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        del families, target, host, index_host, draft, draft_host, saved
        gc.collect()
        torch.cuda.empty_cache()

(OUT/'complete.json').write_text(json.dumps(dict(
    cases=rows, passed=len(rows), elapsed_s=time.time()-started,
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
    scope='Actual pool DMA, local DCP mapping and replicated sidecars; not full-model or consensus parity',
), indent=2))
print('HICACHE_BYTES_COMPLETE', len(rows), flush=True)
