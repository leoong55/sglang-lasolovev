"""8-GPU kernel parity: exact TP8/EP8 peers, eager + changed-input graph replay.

Run in the built image, with the serving workload idle:
  torchrun --standalone --nproc-per-node=8 tests/check_cp_fusion_gpu.py
No model weights are loaded. Context fixtures isolate the real FlashInfer
workspace/kernel from model construction; this is not an end-to-end model test.
"""

import json
import os
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 8, "Use exactly 8 GPUs"
    assert torch.cuda.get_device_capability() == (9, 0), "Hopper SM90 test"
    cpu_group = dist.new_group(backend="gloo")
    ep_device = dist.new_group(backend="nccl")
    group = NS(world_size=8, ranks=list(range(8)), device_group=ep_device, cpu_group=cpu_group)

    from sglang.srt.layers import flashinfer_comm_fusion as fi
    from sglang.srt.layers.cp import glm53_decode_fusion as subject

    parallel = NS(moe_ep_size=8, moe_ep_rank=dist.get_rank(), nnodes=1)
    resources = NS(buffers={})
    config = NS(comm=NS(flashinfer_allreduce_fusion_backend="trtllm"))
    with ExitStack() as stack:
        for name, value in (
            ("get_parallel", lambda: parallel), ("get_resources", lambda: resources),
            ("get_exec", lambda: config), ("get_moe_ep_group", lambda: group),
            ("get_tp_group", lambda: group),
        ):
            stack.enter_context(patch.object(fi, name, value))
        assert fi.ensure_workspace_initialized(
            max_token_num=2048, hidden_dim=6144, dtype=torch.bfloat16,
            use_attn_tp_group=False), "Fusion workspace unavailable; do not enable the patch"
        # Any fallback makes this test fail, rather than pass using NCCL.
        stack.enter_context(patch("sglang.srt.distributed.tensor_model_parallel_all_reduce",
                                  side_effect=AssertionError("Unexpected unfused fallback")))
        norm = NS(weight=torch.ones(6144, device="cuda", dtype=torch.bfloat16), variance_epsilon=1e-6)
        records = []
        for n in (1, 2, 8, 32, 40, 48, 64):
            x = torch.empty((n, 6144), device="cuda", dtype=torch.bfloat16)
            residual = torch.empty_like(x)
            def fill(seed):
                torch.manual_seed(seed + dist.get_rank())
                x.normal_(0, 0.2)
                torch.manual_seed(seed + 1000)  # Residual replicated on TP ranks.
                residual.normal_()
            def reference():
                summed = x.float()
                dist.all_reduce(summed)
                # Baseline all-reduce materializes BF16 before add+RMSNorm.
                r = summed.to(torch.bfloat16).float() + residual.float()
                y = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + norm.variance_epsilon)
                return y.to(torch.bfloat16), r.to(torch.bfloat16)
            def check(result, ref):
                for actual, expected in zip(result, ref):
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.025)
            fill(42)
            ref = reference()
            for _ in range(3):
                check(subject.finish(x, residual, norm), ref)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = subject.finish(x, residual, norm)
            for seed in (101, 202, 303):
                fill(seed)
                ref = reference()
                graph.replay()
                torch.cuda.synchronize()
                check(out, ref)
            records.append({"rows": n, "eager": "pass", "graph_changed_input": "pass"})
        dist.barrier()
        if dist.get_rank() == 0:
            print(json.dumps({"kernel_parity": records, "model_accuracy": "not_tested"}))
        fi.cleanup_flashinfer_workspace()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
