"""Short EP8 local-MoE search, one real GLM layer, no full-model reloads.

Sequential gate/up then down search (not an exhaustive joint optimum).
Actual routing, quantization, both GEMMs and combine are timed in CUDA graphs.
Selection minimizes the mean of worst-rank times for uniform and skewed routes.
Synthetic activations/routes cannot establish serving speed or model accuracy.
"""
import argparse
import json
import os
import statistics
from pathlib import Path

from check_w4_humming_gpu import load_layer, reference_local, relative_error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--checkpoint-layer", type=int, default=3)
    p.add_argument("--batches", type=int, nargs="+", default=[40, 48])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--min-gain", type=float, default=0.03)
    p.add_argument("--output", type=Path, default=Path("/mnt/cache/glm53-cutlass-tuning.json"))
    args = p.parse_args()
    if (int(os.environ.get("WORLD_SIZE", "1")) != 8 or
            any(not 1 <= b <= 64 for b in args.batches) or args.iterations < 10 or
            not 0 <= args.min_gain < 1):
        p.error("Use torchrun with 8 ranks, batches 1..64 and at least 10 iterations")
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.moe import glm53_cutlass as tuning
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import publish
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (9, 0):
        p.error("SM90 required")
    hf = json.loads((args.model_path / "config.json").read_text())
    for key, value in dict(hidden_size=6144, moe_intermediate_size=2048,
                           n_routed_experts=256, num_experts_per_tok=8).items():
        if hf.get(key) != value:
            p.error(f"Unsupported model geometry: {key}")
    server = ServerArgs(model_path=str(args.model_path), trust_remote_code=True,
                        tp_size=8, ep_size=8, moe_a2a_backend="none",
                        moe_runner_backend="cutlass", quantization="w4afp8",
                        dtype="bfloat16", disable_shared_experts_fusion=True)
    publish(server, role="test")
    init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank, timeout=300)
    initialize_model_parallel(tensor_model_parallel_size=8, expert_model_parallel_size=8)
    tuning.load_library()  # No compilation, and never inside CUDA capture.
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    quant = W4AFp8Config.from_config(hf["quantization_config"])
    factor = float(hf.get("routed_scaling_factor", 1))
    with torch.device("cuda"):
        layer = FusedMoE(num_experts=256, hidden_size=6144, intermediate_size=2048,
                         layer_id=args.checkpoint_layer, top_k=8, params_dtype=torch.bfloat16,
                         quant_config=quant, quant_method=W4AFp8MoEMethod(quant),
                         inplace=False, gate_up_interleaved=False, routed_scaling_factor=factor,
                         prefix=f"model.layers.{args.checkpoint_layer}.mlp.experts")
    load_layer(layer, args.model_path, args.checkpoint_layer, rank, False)
    packed = tuple(getattr(layer, n + "_weight").detach() for n in ("w13", "w2"))
    scales = tuple(getattr(layer, n + "_weight_scale_inv").detach() for n in ("w13", "w2"))
    layer.quant_method.process_weights_after_loading(layer)
    results, pairs = [], {}

    for batch in sorted(set(args.batches)):
        x = torch.empty(batch, 6144, device="cuda", dtype=torch.bfloat16)
        ids = torch.empty(batch, 8, device="cuda", dtype=torch.int32)
        probabilities = torch.empty(batch, 8, device="cuda", dtype=torch.float32)
        topk = StandardTopKOutput(topk_weights=probabilities, topk_ids=ids, router_logits=None)
        def fill(seed, mode):
            torch.manual_seed(seed)
            x.normal_()
            logits = torch.randn(batch, 256, device="cuda")
            if mode == "skew":
                logits[:, :32] += 1.0
            chosen = logits.topk(8, dim=-1).indices
            if mode == "empty_peers":
                chosen = torch.arange(8, device="cuda").expand(batch, 8)
            ids.copy_(chosen)
            probabilities.copy_(torch.softmax(torch.randn(batch, 8, device="cuda"), -1))
        def forward():
            dispatch = layer.dispatcher.dispatch(x, topk)
            return layer.dispatcher.combine(layer.quant_method.apply(layer, dispatch))
        def capture(pair):
            tuning._benchmark_pair = pair
            for _ in range(4):
                forward()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with model_capture_mode(), torch.cuda.graph(graph):
                output = forward()
            return graph, output
        fill(7, "uniform")
        stock_graph, stock_out = capture((0, 0))
        # Independent accuracy reference at one small batch; all candidates
        # are also checked against the unchanged stock path at every batch.
        if batch == min(args.batches):
            local_ids = torch.where((ids >= rank*32) & (ids < (rank+1)*32), ids-rank*32, -1)
            ref = reference_local(x, local_ids, probabilities, packed, scales, factor)
            stock_graph.replay()
            torch.cuda.synchronize()
            if relative_error(stock_out, ref) > 0.08:
                raise AssertionError("Stock CUTLASS failed FP32 reference; no tuning file written")
            del ref
        measurements = {}
        def measure(pair, fresh=False):
            if pair in measurements and not fresh:
                return measurements[pair]
            fill(7, "uniform")
            graph, output = capture(pair)
            error = 0.0
            for seed, mode in ((19, "uniform"), (41, "empty_peers"), (73, "skew")):
                fill(seed, mode)
                stock_graph.replay()
                expected = stock_out.clone()
                graph.replay()
                torch.cuda.synchronize()
                # Includes empty-local ranks and stale-buffer detection.
                if torch.count_nonzero(expected) == 0:
                    err = 0.0 if torch.count_nonzero(output) == 0 else float("inf")
                else:
                    err = relative_error(output, expected)
                error = max(error, err)
            e = torch.tensor(error, device="cuda")
            dist.all_reduce(e, op=dist.ReduceOp.MAX)
            if float(e) > 0.01:
                raise AssertionError(f"Candidate {pair} differs from stock: {float(e)}; no config written")
            modes = {}
            for mode in ("uniform", "skew"):
                fill(101 if fresh else 73, mode)
                dist.barrier()
                samples = []
                for _ in range(3):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end) / args.iterations)
                worst = torch.tensor(samples, device="cuda")
                dist.all_reduce(worst, op=dist.ReduceOp.MAX)
                modes[mode] = statistics.median(worst.tolist())
            row = dict(pair=list(pair), score_ms=statistics.mean(modes.values()),
                       modes_ms=modes, max_relative_l2=float(e), fresh=fresh)
            measurements[pair] = row
            if rank == 0:
                print(json.dumps(dict(batch=batch, **row)), flush=True)
            return row
        baseline = measure((0, 0))
        gate = min([measure((v, 0)) for v in tuning.VARIANTS], key=lambda r:r["score_ms"])["pair"][0]
        candidate = min([measure((gate, v)) for v in tuning.VARIANTS], key=lambda r:r["score_ms"])
        chosen = tuple(candidate["pair"])
        # Confirm against a fresh stock timing and another routing seed.
        confirmed = measure(chosen, fresh=True)
        baseline = measure((0, 0), fresh=True)
        gain = 1 - confirmed["score_ms"] / baseline["score_ms"]
        if gain < args.min_gain:
            chosen = (0, 0)
        pairs[str(batch)] = list(chosen)
        results.append(dict(batch=batch, chosen=list(chosen), confirmed_gain=gain,
                            measurements=list(measurements.values())))
        del stock_graph, stock_out
    tuning._benchmark_pair = None
    if rank == 0:
        data = dict(schema=1, geometry=[6144, 2048, 256, 32, 8], sm=90,
                    gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), pairs=pairs,
                    checkpoint_layer=args.checkpoint_layer, synthetic_activations=True,
                    full_model_speed="not_tested", model_accuracy="not_tested", results=results)
        tuning.validate_config(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(args.output)
        print(f"Saved {args.output}; selected pairs: {pairs}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
