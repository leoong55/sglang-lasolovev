"""Map allocator-widened top-k IDs into a sorted local KV layout and compact."""
import torch
import triton
import triton.language as tl


@triton.jit
def _compact(IDs, Local, Out, Lengths, N: tl.constexpr, K: tl.constexpr,
             STRIDE: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slots = tl.arange(0,BLOCK)
    ids = tl.load(IDs + row*STRIDE + slots, slots<K, other=-1)
    lo = tl.full((BLOCK,),0,tl.int32)
    hi = tl.full((BLOCK,),N,tl.int32)
    for _ in range(STEPS):
        mid = (lo+hi)//2
        value = tl.load(Local+mid, mid<N, other=2147483647)
        right = (lo<hi) & (value<ids)
        hi = tl.where((lo<hi) & ~right,mid,hi)
        lo = tl.where(right,mid+1,lo)
    value = tl.load(Local+lo,lo<N,other=2147483647)
    valid = (slots<K) & (ids>=0) & (ids<2147483647) & (lo<N) & (value==ids)
    destination = tl.cumsum(valid.to(tl.int32),0)-1
    tl.store(Out+row*K+destination,lo,valid)
    tl.store(Lengths+row,tl.sum(valid.to(tl.int32),0))


def compact_topk_cuda(ids,local):
    rows,k = ids.shape
    out = torch.full_like(ids,-1,dtype=torch.int32)
    lengths = torch.empty(rows,dtype=torch.int32,device=ids.device)
    _compact[(rows,)](ids,local,out,lengths,N=local.numel(),K=k,
                      STRIDE=ids.stride(0),STEPS=local.numel().bit_length()+1,
                      BLOCK=triton.next_power_of_2(k))
    return out,lengths
