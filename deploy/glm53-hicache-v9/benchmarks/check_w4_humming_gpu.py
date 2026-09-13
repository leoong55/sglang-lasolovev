"""EP8 W4 MoE correctness and timing, using one actual checkpoint layer.

Run with torchrun --standalone --nproc-per-node=8. No serving process or full
model load: 32 local experts/rank, H=6144, I=2048. Timings cover local dispatch,
quantization, both expert GEMMs, SiLU and combine; not attention or collectives.
An independent FP32 calculation dequantizes the original signed INT4 weights.
"""

import argparse
import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path


def unpack_int4(packed, scales, group_size=128):
    import torch

    unsigned = packed.view(torch.uint8)
    low = (unsigned & 15).to(torch.int16)
    high = (unsigned >> 4).to(torch.int16)
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    signed = ((nibbles ^ 8) - 8).float()
    return signed * scales.to(torch.bfloat16).float().repeat_interleave(group_size, -1)


def reference_local(x, ids, probabilities, packed, scales, factor):
    import torch
    import torch.nn.functional as F

    ref = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(packed[0].shape[0]):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        w13 = unpack_int4(packed[0][expert], scales[0][expert])
        up = F.linear(x[rows].float(), w13)
        del w13
        gate, value = up.chunk(2, dim=-1)
        w2 = unpack_int4(packed[1][expert], scales[1][expert])
        down = F.linear(F.silu(gate) * value, w2)
        del w2
        ref.index_add_(0, rows, down * probabilities[rows, slots, None] * factor)
    return ref


def relative_error(actual, expected):
    import torch

    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite output/reference")
    return float((actual.float() - expected).norm() / expected.norm().clamp_min(1e-6))


def load_layer(layer, model_path, index, rank, synthetic):
    import torch
    from safetensors import safe_open

    if synthetic:
        torch.manual_seed(1000 + rank)
        for prefix in ("w13", "w2"):
            p = getattr(layer, prefix + "_weight")
            p.data.copy_(
                torch.randint(-128, 128, p.shape, device=p.device, dtype=torch.int8)
            )
            getattr(layer, prefix + "_weight_scale_inv").data.fill_(1 / 256)
            getattr(layer, prefix + "_input_scale").data.fill_(1 / 64)
        return

    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with ExitStack() as stack:
        readers = {}

        def read(suffix):
            matches = [name for name in weight_map if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one checkpoint tensor ending in {suffix!r}; got {matches}"
                )
            name = matches[0]
            file = weight_map[name]
            if file not in readers:
                readers[file] = stack.enter_context(
                    safe_open(str(model_path / file), framework="pt", device="cpu")
                )
            return name, readers[file].get_tensor(name)

        for expert in range(rank * 32, (rank + 1) * 32):
            for projection, shard, destination in (
                ("gate_proj", "w1", "w13"),
                ("up_proj", "w3", "w13"),
                ("down_proj", "w2", "w2"),
            ):
                for suffix in ("weight", "weight_scale_inv"):
                    key, tensor = read(
                        f"layers.{index}.mlp.experts.{expert}.{projection}.{suffix}"
                    )
                    param = getattr(layer, destination + "_" + suffix)
                    # Exercise the real checkpoint loader and global-to-local EP
                    # ownership, including gate/up packing and scale checks.
                    param.weight_loader(
                        param, tensor, key, shard_id=shard, expert_id=expert
                    )
            # Phala stores calibration tensors under w1/w2/w3.input_scale,
            # independently of the gate_proj/up_proj/down_proj weight names.
            # This matches FusedMoE.make_expert_input_scale_params_mapping().
            for shard, destination in (("w1", "w13"), ("w3", "w13"), ("w2", "w2")):
                key, tensor = read(
                    f"layers.{index}.mlp.experts.{expert}.{shard}.input_scale"
                )
                param = getattr(layer, destination + "_input_scale")
                param.weight_loader(
                    param, tensor, key, shard_id=shard, expert_id=expert
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--checkpoint-layer", type=int, default=3)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 40, 48, 2048])
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--output", type=Path, default=Path("/tmp/glm53-w4-humming.json"))
    args = p.parse_args()
    if min(args.batches) <= 0 or args.iterations < 10:
        p.error("Use positive batches and at least 10 iterations")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        p.error("Use torchrun --standalone --nproc-per-node=8")

    import torch
    import torch.distributed as dist

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (9, 0):
        p.error("This experiment targets H200/SM90")
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads((args.model_path / "config.json").read_text())
    for key, expected in dict(
        hidden_size=6144,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=8,
    ).items():
        if config.get(key) != expected:
            p.error(
                f"GLM53 geometry mismatch: {key}={config.get(key)}, expected {expected}"
            )

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import publish
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    server = ServerArgs(
        model_path=str(args.model_path),
        trust_remote_code=True,
        tp_size=8,
        ep_size=8,
        moe_a2a_backend="none",
        moe_runner_backend="humming",
        quantization="w4afp8",
        dtype="bfloat16",
        disable_shared_experts_fusion=True,
    )
    publish(server, role="test")
    init_distributed_environment(
        world_size=8, rank=rank, local_rank=local_rank, timeout=300
    )
    initialize_model_parallel(
        tensor_model_parallel_size=8, expert_model_parallel_size=8
    )

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
    from sglang.srt.layers.quantization.w4afp8_humming import W4AFp8HummingMoEMethod
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    factor = float(config.get("routed_scaling_factor", 1))
    quant = W4AFp8Config.from_config(config["quantization_config"])
    layers = {}
    with torch.device("cuda"):
        for name, method in (
            ("cutlass", W4AFp8MoEMethod),
            ("humming", W4AFp8HummingMoEMethod),
        ):
            layers[name] = FusedMoE(
                num_experts=256,
                hidden_size=6144,
                intermediate_size=2048,
                layer_id=args.checkpoint_layer,
                top_k=8,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                quant_method=method(quant),
                inplace=False,
                gate_up_interleaved=False,
                routed_scaling_factor=factor,
                prefix=f"model.layers.{args.checkpoint_layer}.mlp.experts",
            )
    load_layer(
        layers["cutlass"], args.model_path, args.checkpoint_layer, rank, args.synthetic
    )
    layers["humming"].load_state_dict(layers["cutlass"].state_dict(), strict=True)
    # Original packed tensors/scales remain available through the CUTLASS layer
    # or these references; never expand the entire layer to FP32/BF16.
    packed = tuple(
        getattr(layers["cutlass"], n + "_weight").detach() for n in ("w13", "w2")
    )
    scales = tuple(
        getattr(layers["cutlass"], n + "_weight_scale_inv").detach()
        for n in ("w13", "w2")
    )
    for layer in layers.values():
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.num_experts == 256 and layer.num_local_experts == 32
    assert layers["humming"].quant_method.runner.fused_func is None

    result = dict(
        rank=rank,
        synthetic=args.synthetic,
        checkpoint_layer=args.checkpoint_layer,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        activation_scales=dict(
            cutlass="checkpoint static", humming="dynamic per-token"
        ),
        scope="local MoE including dispatcher, excluding all-reduce/attention",
        cases=[],
    )
    for batch in args.batches:
        x = torch.empty((batch, 6144), device="cuda", dtype=torch.bfloat16)
        ids = torch.empty((batch, 8), device="cuda", dtype=torch.int32)
        probabilities = torch.empty((batch, 8), device="cuda", dtype=torch.float32)
        topk = StandardTopKOutput(probabilities, ids, None)

        def fill(seed, concentrated=False):
            torch.manual_seed(seed)  # Identical tokens/routes on all EP ranks.
            x.normal_()
            logits = torch.randn((batch, 256), device="cuda")
            chosen = logits.topk(8, dim=-1).indices
            if concentrated:
                # Every nonzero EP rank has zero local assignments. Also checks
                # that no stale contributions survive from the previous replay.
                chosen = torch.arange(8, device="cuda").expand(batch, 8)
            ids.copy_(chosen)
            probabilities.copy_(
                torch.softmax(torch.randn((batch, 8), device="cuda"), -1)
            )

        def forward(layer):
            dispatch = layer.dispatcher.dispatch(x, topk)
            return layer.dispatcher.combine(layer.quant_method.apply(layer, dispatch))

        fill(7)
        graphs, outputs = {}, {}
        for name, layer in layers.items():
            for _ in range(5):
                forward(layer)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with model_capture_mode(), torch.cuda.graph(graph):
                outputs[name] = forward(layer)
            graphs[name] = graph
        errors = []
        for seed, concentrated in ((19, False), (41, True), (73, False)):
            fill(seed, concentrated)
            local_ids = torch.where(
                (ids >= rank * 32) & (ids < (rank + 1) * 32), ids - rank * 32, -1
            )
            ref = reference_local(x, local_ids, probabilities, packed, scales, factor)
            entry = dict(
                seed=seed,
                concentrated=concentrated,
                local_assignments=int((local_ids >= 0).sum()),
            )
            for name, layer in layers.items():
                eager = forward(layer).clone()
                graphs[name].replay()
                torch.cuda.synchronize()
                actual = outputs[name].clone()
                torch.testing.assert_close(actual, eager, rtol=0.001, atol=0.002)
                error = relative_error(actual, ref)
                if error > 0.08:
                    raise AssertionError(
                        f"rank={rank} batch={batch} {name}: relative L2={error} > 0.08"
                    )
                entry[name + "_relative_l2"] = error
                # Check the EP sum as well, outside the timing section.
                summed, expected = actual.float(), ref.clone()
                dist.all_reduce(summed)
                dist.all_reduce(expected)
                if relative_error(summed, expected) > 0.08:
                    raise AssertionError(f"{name}: EP8 sum failed")
            errors.append(entry)

        fill(73)
        timing = {}
        for name, graph in graphs.items():
            dist.barrier()
            values = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(args.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) / args.iterations)
            timing[name] = dict(median_ms=statistics.median(values), samples_ms=values)
        result["cases"].append(dict(batch=batch, correctness=errors, timings=timing))
        if rank == 0:
            print(json.dumps(result["cases"][-1]), flush=True)
        del graphs, outputs, graph, actual, eager

    gathered = [None] * 8
    dist.all_gather_object(gathered, result)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    status="passed",
                    model_accuracy="not_tested",
                    full_model_speed="not_tested",
                    ranks=gathered,
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            f"Saved {args.output}. This is a component gate, not full-model validation.",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
