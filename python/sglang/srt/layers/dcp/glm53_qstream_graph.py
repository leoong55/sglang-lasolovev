"""Capture one fixed Q-stream tile; dynamic request selection stays outside."""
import torch


def signature(kv, q_shape, topk, scale, group):
    # Contents/metadata may change, addresses and capacities may not.
    return (kv.data_ptr(), tuple(kv.shape), tuple(kv.stride()), kv.dtype,
            str(kv.device), tuple(q_shape), topk, float(scale), id(group))


class TileGraph:
    def __init__(self, runner, q_shape, topk, kv, layout, scale):
        from sglang.srt.layers.dcp.glm53_qstream import LocalLayout
        from sglang.srt.observability.glm53_prefill import record
        import time
        self.key = signature(kv,q_shape,topk,scale,runner.group)
        self.q = torch.zeros(q_shape,dtype=kv.dtype,device=kv.device)
        self.indices = torch.full((q_shape[0],topk),-1,dtype=torch.int32,device=kv.device)
        self.layout = LocalLayout(layout.widened.clone(),layout.physical)
        self.kv = kv  # Keep graph-captured storage alive until invalidated.
        self.graph = torch.cuda.CUDAGraph()
        caller = torch.cuda.current_stream()
        started = time.perf_counter()
        with runner.group.graph_capture() as context:
            # JIT and allocator warmup are outside capture. Every DCP rank
            # follows this identical shape-driven branch and collective order.
            runner.tile(self.q,self.indices,kv,self.layout,scale)
            context.stream.synchronize()
            with torch.cuda.graph(self.graph,stream=context.stream):
                self.output = runner.tile(self.q,self.indices,kv,self.layout,scale)
        caller.wait_stream(context.stream)
        record("attention_graph_capture", capacity=kv.shape[0], seconds=time.perf_counter()-started)

    def update_layout(self, layout):
        self.layout.widened.copy_(layout.widened)

    def replay(self,q,indices):
        rows=q.shape[0]
        self.q.view(torch.uint8).zero_()
        self.indices.fill_(-1)
        self.q[:rows].copy_(q)
        self.indices[:rows].copy_(indices)
        self.graph.replay()
        return self.output[:rows]
