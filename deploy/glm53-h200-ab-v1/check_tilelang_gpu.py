"""Run on H200 before serving: exercise raw FP8 KV writes, attention and graph replay.

This is a synthetic kernel test, not model-level accuracy or a serving benchmark.
It fails rather than skipping when no Hopper GPU is available.
"""

import json

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Run this validation on the target H200/Hopper GPU")
    torch.manual_seed(12345)
    records = []
    for heads, tail, seq in (
        (8, 64, 20),
        (16, 64, 40),
        (32, 64, 80),
        (64, 64, 16),
        (8, 0, 16),
        (8, 64, 512),
    ):
        dim = 512 + tail
        q = torch.randn(seq, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.5
        kv_source = torch.randn(4096, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.5
        # Call the actual pool writer used by the runtime, without model allocation.
        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        pool.use_dsa = True
        pool.dtype = torch.float8_e4m3fn
        pool.kv_cache_dim = dim
        pool.kv_lora_rank = 512
        pool.qk_rope_head_dim = tail
        pool.dsa_kv_cache_store_fp8 = True
        raw = torch.zeros_like(kv_source, dtype=torch.float8_e4m3fn)
        loc = torch.arange(4096, device="cuda", dtype=torch.int64)
        pool._write_mla_kv_buffer(
            raw.view(torch.uint8), loc, kv_source[:, :, :512], kv_source[:, :, 512:]
        )
        # Runtime reserves slot zero; the writer deliberately skips it.
        torch.testing.assert_close(
            raw[1:].float(), kv_source[1:].to(raw.dtype).float(), rtol=0, atol=0
        )
        assert torch.count_nonzero(raw[0].float()) == 0
        indices = torch.randint(
            1, 4096, (seq, 1, 2048), device="cuda", dtype=torch.int32
        )
        indices[:, :, -64:] = -1
        scale = dim**-0.5
        for _ in range(2):
            result = tilelang_sparse_fwd(q, raw, indices, scale).reshape(
                seq, heads, 512
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tilelang_sparse_fwd(q, raw, indices, scale).reshape(
                seq, heads, 512
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, result, rtol=0, atol=0)
        # Reference uses exactly the quantized Q/KV inputs. Remaining error includes
        # the kernel's FP8 probabilities and BF16 partial outputs.
        reference = []
        qf = q.to(raw.dtype).float()
        kvf = raw[:, 0].float()
        for i in range(seq):
            ids = indices[i, 0]
            selected = kvf[ids.clamp_min(0).long()]
            scores = qf[i] @ selected.T * scale
            scores[:, ids < 0] = -torch.inf
            reference.append(torch.softmax(scores, dim=-1) @ selected[:, :512])
        ref = torch.stack(reference)
        rel = float((result.float() - ref).norm() / ref.norm())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                result.float().flatten(), ref.flatten(), dim=0
            )
        )
        record = dict(
            heads=heads, tail=tail, queries=seq, relative_l2=rel, cosine=cosine
        )
        print(json.dumps(record), flush=True)
        if not torch.isfinite(result).all() or rel > 0.08 or cosine < 0.995:
            raise RuntimeError("TileLang FP8 numerical smoke check failed")
        records.append(record)
    print(
        json.dumps(dict(gpu=torch.cuda.get_device_name(), passed=True, cases=records))
    )


if __name__ == "__main__":
    main()
