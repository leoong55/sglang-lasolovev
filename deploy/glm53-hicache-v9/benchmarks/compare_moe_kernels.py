"""Synthetic one-rank GLM-5.3 EP8 MoE kernel comparison on one H200.

32 local experts, H=6144, I=2048, 8/256 routing. Identical representable
weights/routes for W4AFP8 CUTLASS, FP8 Triton, FP8 DeepGEMM. Includes local
permute/quantize/activation/combine, excludes gate GEMM/shared expert/collectives.
No full checkpoint download or model weight load. --model-path reads config only.
This measures kernels, not full-model throughput or checkpoint accuracy.
"""

import argparse
import json
import os
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[40, 48])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 41])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("moe-kernels.json"))
    args = p.parse_args()
    if min(args.batches) < 1 or args.iterations < 10:
        p.error("Use positive batch sizes and at least 10 iterations")

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "Run torchrun --nproc-per-node=1"
    assert torch.cuda.get_device_capability() == (9, 0), "This comparison targets H200/SM90"
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    config = ServerArgs(model_path=args.model_path, trust_remote_code=True,
                        tp_size=1, ep_size=1, moe_a2a_backend="none")
    publish(config, role="benchmark")
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel()

    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmMoeQuantInfo, DeepGemmRunnerCore,
        pre_permute_standard_to_deep_gemm, post_permute_deep_gemm_to_standard,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.quantization.w4afp8 import interleave_scales
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    E, H, I, K = 32, 6144, 2048, 8
    torch.manual_seed(123)
    def weights(rows, cols):
        integers = torch.randint(-8, 8, (E, rows, cols), dtype=torch.int8, device="cuda")
        packed = (integers[..., 1::2] << 4) | (integers[..., ::2] & 15)
        fp8 = integers.to(torch.float8_e4m3fn)
        scales4 = interleave_scales(torch.full((E, rows, cols // 128), 1/256,
                                              device="cuda", dtype=torch.bfloat16))
        scales8 = torch.full((E, rows // 128, cols // 128), 1/256, device="cuda")
        return packed.contiguous(), scales4, fp8, scales8
    w1, s1, f1, t1 = weights(2 * I, H)
    w2, s2, f2, t2 = weights(H, I)
    strides = [torch.full((E, 3), n, device="cuda", dtype=torch.int64)
               for n in (H, H, 2*I, I, I, H, 2*I, H)]
    offsets = torch.empty(E + 1, device="cuda", dtype=torch.int32)
    problems = [torch.empty((E, 3), device="cuda", dtype=torch.int32) for _ in range(2)]
    a1 = torch.tensor([1/64], device="cuda")
    a2 = torch.tensor([1/16], device="cuda")
    runner_cfg = MoeRunnerConfig(num_experts=256, num_local_experts=E, hidden_size=H,
        intermediate_size_per_partition=I, top_k=K, inplace=False,
        routed_scaling_factor=1.0, gate_up_interleaved=False)
    dg_quant = DeepGemmMoeQuantInfo(f1, f2, use_fp8=True, w13_scale=t1, w2_scale=t2,
                                   block_shape=[128, 128])
    dg = DeepGemmRunnerCore(runner_cfg)

    def measured(fn):
        # Warmup also resolves the DeepGEMM layout before graph capture.
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph):
            result = fn()
        graph.replay()
        torch.cuda.synchronize()
        values = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / args.iterations)
        return result.clone(), {"median_ms": statistics.median(values), "samples_ms": values}

    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  dimensions=dict(local_experts=E, total_experts=256, hidden=H, intermediate=I, topk=K),
                  synthetic=True, full_model_throughput=False, results=[])
    # Only W4's local-ID sentinel handling reads EP width here. No multi-GPU
    # communication is included; the real single-rank TP group remains intact.
    with get_parallel().override(moe_ep_size=8, moe_ep_rank=0):
        for batch in args.batches:
            for seed in args.seeds:
                torch.manual_seed(seed)
                x = torch.randn((batch, H), device="cuda", dtype=torch.bfloat16)
                logits = torch.randn((batch, 256), device="cuda")
                ids = logits.topk(K, dim=-1).indices.to(torch.int32)
                ids = torch.where(ids < E, ids, -1)
                probs = torch.full((batch, K), 1/K, device="cuda")
                topk = StandardTopKOutput(probs, ids, logits)
                dispatch = StandardDispatchOutput(x, None, topk)
                def w4():
                    return cutlass_w4a8_moe(x, w1, w2, s1, s2, probs, ids,
                        *strides, offsets, *problems, a1, a2)
                def fp8_triton():
                    return fused_experts(x, f1, f2, topk, runner_cfg, use_fp8_w8a8=True,
                        w1_scale=t1, w2_scale=t2, block_shape=[128, 128])
                def fp8_deepgemm():
                    state = {}
                    inp = pre_permute_standard_to_deep_gemm(dispatch, dg_quant, runner_cfg, state)
                    out = dg.run(inp, dg_quant, state)
                    return post_permute_deep_gemm_to_standard(out, dg_quant, runner_cfg, state).hidden_states

                # BF16 reference for local experts; no quantized activation.
                ref = torch.zeros_like(x, dtype=torch.float32)
                for expert in range(E):
                    rows, slots = torch.where(ids == expert)
                    if not rows.numel():
                        continue
                    up = F.linear(x[rows], f1[expert].to(torch.bfloat16) / 256)
                    gate, value = up.chunk(2, dim=-1)
                    down = F.linear(F.silu(gate) * value, f2[expert].to(torch.bfloat16) / 256)
                    ref.index_add_(0, rows, down.float() * probs[rows, slots, None])
                entry = dict(batch=batch, seed=seed, local_assignments=int((ids >= 0).sum()), kernels={})
                for name, fn in (("w4afp8_cutlass", w4), ("fp8_triton", fp8_triton), ("fp8_deepgemm", fp8_deepgemm)):
                    result, timing = measured(fn)
                    error = ((result.float() - ref).norm() / ref.norm().clamp_min(1e-6)).item()
                    if not torch.isfinite(result).all() or error > 0.08:
                        raise RuntimeError(f"{name}: reference relative L2={error}; reject timing comparison")
                    timing["relative_l2_vs_bf16"] = error
                    entry["kernels"][name] = timing
                report["results"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(entry), flush=True)
    dist.destroy_process_group()
    print(f"Saved {args.output}; do not extrapolate one-layer speedup to the entire model")


if __name__ == "__main__":
    main()
