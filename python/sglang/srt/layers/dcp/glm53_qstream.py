"""Experimental SM90 CP8/DCP4 prefill, with token-axis communication.

No persistent KV values are cached here: each layer repacks its local shard once.
All loop bounds use CPU shapes, never device-dependent counts or collective skips.
"""
from dataclasses import dataclass
import math
import torch

SENTINEL = torch.iinfo(torch.int32).max
TILE = 256


@dataclass
class LocalLayout:
    widened: torch.Tensor
    physical: torch.Tensor


def prepare_layout(translator, batch):
    table = translator.index_table_for_batch(batch)
    if table.entry_page_size != 1 or table.is_translated:
        raise ValueError("Q-stream needs allocator widened token read IDs")
    # Include every request, including those whose Q rows belong to another CP rank.
    lens = [int(x) for x in batch.seq_lens_cpu]
    wide = torch.cat([table.ids[table.row_ids[i], :n] for i, n in enumerate(lens)]).int()
    owned = translator.dcp_read_ownership(wide)
    from sglang.srt.runtime_context import get_parallel
    size = get_parallel().attn_dcp_size
    # The allocator's widened pages guarantee at most ceil(n/size) owned rows
    # per sequence. Sort virtual IDs, not logical positions or physical pages.
    capacity = max(1, sum((n + size - 1) // size for n in lens))
    # Stable KV capacities avoid recapturing for every context-length increment.
    capacity = 1 << (capacity - 1).bit_length()
    padded = torch.cat((torch.where(owned, wide, SENTINEL), wide.new_full((capacity,), SENTINEL)))
    selected = padded.sort().values[:capacity]
    physical = translator.translate_dcp_read_ids(torch.where(selected == SENTINEL, 0, selected)).int()
    return LocalLayout(selected, physical)


def compact_topk(widened, local_ids):
    """Search allocator IDs and stably compact only this shard's positions."""
    offsets = torch.searchsorted(local_ids, widened.contiguous()).clamp_max(local_ids.numel()-1)
    valid = (widened >= 0) & (widened != SENTINEL) & (local_ids[offsets] == widened)
    order = torch.argsort(valid.to(torch.int32), dim=-1, descending=True, stable=True)
    indices = torch.where(valid, offsets, -1).gather(-1, order).int().contiguous()
    lengths = valid.sum(-1, dtype=torch.int32)
    return indices, lengths


def correction_base2(partial, local_lse, all_lse):
    """FP32 correction, including a completely empty softmax domain."""
    maximum = all_lse.amax(0)
    safe = torch.where(torch.isfinite(maximum), maximum, 0)
    denominator = torch.exp2(all_lse - safe).sum(0)
    numerator = torch.exp2(local_lse - safe)
    scale = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(scale[..., None] > 0, partial.float() * scale[..., None], 0)


class QStreamAttention:
    def __init__(self, group):
        self.group = group
        self.kv_buffer = None
        self.identity = None
        self.tile_graph = None

    def prepare_kv(self, packed, layout, topk):
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import gather_dequant_requant_fp8_paged
        from sglang.srt.observability.glm53_prefill import stage
        rows = layout.physical.numel() + topk
        if self.kv_buffer is None or self.kv_buffer.shape[0] < rows:
            self.kv_buffer = torch.empty((rows, 1, 576), dtype=torch.float8_e4m3fn, device=packed.device)
        with stage("kv_convert"):
            kv = gather_dequant_requant_fp8_paged(packed, layout.physical, extra_rows=topk, out=self.kv_buffer[:rows])
        if self.identity is None:
            self.identity = torch.ones(1, dtype=torch.float32, device=packed.device)
        return kv

    def tile(self, q, indices, kv, layout, scale):
        from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import sparse_mla_q8kv8_prefill_fwd
        group = self.group
        q_all = torch.empty((TILE*group.world_size, *q.shape[1:]), dtype=q.dtype, device=q.device)
        ids_all = indices.new_empty((TILE*group.world_size, indices.shape[1]))
        # NCCL sees FP8 Q as bytes; no unsupported FP8 reduction/transport type.
        group.all_gather_into_tensor(q_all.view(torch.uint8), q.view(torch.uint8))
        group.all_gather_into_tensor(ids_all, indices)
        local, lengths = compact_topk(ids_all, layout.widened)
        out, _, lse = sparse_mla_q8kv8_prefill_fwd(
            q_all, kv, local.unsqueeze(1), scale, self.identity, self.identity,
            d_v=512, topk_length=lengths)
        empty = lengths == 0
        lse = torch.where(empty[:,None], -torch.inf, lse.float())
        out = torch.where(empty[:,None,None], 0, out)
        lses = group.all_gather(lse.contiguous(), dim=0).view(group.world_size, *lse.shape)
        corrected = correction_base2(out, lse, lses)
        # Queries are rank-major; every destination owns TILE tokens and ALL heads.
        return group.reduce_scatter_along_dim(corrected.contiguous(), dim=0).to(out.dtype)

    def run(self, q, indices, packed_kv, layout, scale):
        from sglang.srt.observability.glm53_prefill import stage
        kv = self.prepare_kv(packed_kv, layout, indices.shape[1])
        out = torch.empty((q.shape[0], q.shape[1], 512), dtype=torch.bfloat16, device=q.device)
        from sglang.srt.runtime_context import get_schedule
        from sglang.srt.model_executor.runner import get_is_capture_mode
        from sglang.srt.layers.dcp.glm53_qstream_graph import TileGraph, signature
        graph_on = get_schedule().glm53_prefill_attention_graph == "on" and not get_is_capture_mode()
        if graph_on:
            shape = (TILE,*q.shape[1:])
            key = signature(kv,shape,indices.shape[1],scale,self.group)
            if self.tile_graph is None or self.tile_graph.key != key:
                self.tile_graph = None  # Invalidate when KV address/capacity changes.
                self.tile_graph = TileGraph(self,shape,indices.shape[1],kv,layout,scale)
            self.tile_graph.update_layout(layout)
            with stage("attention"):
                for start in range(0,q.shape[0],TILE):
                    result = self.tile_graph.replay(q[start:start+TILE],indices[start:start+TILE])
                    out[start:start+result.shape[0]].copy_(result)
            return out
        q_stage = torch.zeros((TILE,*q.shape[1:]),dtype=q.dtype,device=q.device)
        ids_stage = indices.new_full((TILE,indices.shape[1]), -1)
        with stage("attention"):
            for start in range(0,q.shape[0],TILE):
                count = min(TILE,q.shape[0]-start)
                q_stage.view(torch.uint8).zero_()
                ids_stage.fill_(-1)
                q_stage[:count].copy_(q[start:start+count])
                ids_stage[:count].copy_(indices[start:start+count])
                result = self.tile(q_stage,ids_stage,kv,layout,scale)
                out[start:start+count].copy_(result[:count])
        return out
