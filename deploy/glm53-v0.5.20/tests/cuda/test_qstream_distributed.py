"""torchrun --standalone --nproc-per-node=4 -m pytest -q this_file.py

Exercises the actual Q8 partial kernel and token reduce-scatter. Every rank runs
identical tests. Native four-GPU execution is required; CPU checks are separate.
"""
import importlib.util
import os
from pathlib import Path
import sys
import pytest
import torch
import torch.distributed as dist

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or int(os.getenv('WORLD_SIZE','1'))!=4,reason='torchrun on four SM90 GPUs required')

class Group:
    world_size=4
    def all_gather_into_tensor(self,out,x): dist.all_gather_into_tensor(out,x)
    def all_gather(self,x,dim=0):
        out=x.new_empty((x.shape[0]*4,*x.shape[1:]));self.all_gather_into_tensor(out,x);return out
    def reduce_scatter_along_dim(self,x,dim=0):
        assert dim==0
        out=x.new_empty((x.shape[0]//4,*x.shape[1:]));dist.reduce_scatter_tensor(out,x);return out

@pytest.mark.parametrize('empty_rank',[False,True])
def test_native_q8_partition_and_token_reduction(empty_rank):
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')
    from sglang.srt.layers.dcp.glm53_qstream import QStreamAttention,LocalLayout,TILE,SENTINEL
    from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import sparse_mla_q8kv8_prefill_fwd
    rank=dist.get_rank()
    torch.manual_seed(95)
    keys=torch.randn(257,1,576,device='cuda').to(torch.float8_e4m3fn)
    wide=torch.arange(257,device='cuda',dtype=torch.int32)*4
    owners=torch.arange(257,device='cuda',dtype=torch.int32)%(3 if empty_rank else 4)
    wide+=owners+4096
    selected=torch.where(owners==rank)[0]
    ids=wide[selected]
    if ids.numel()==0: ids=wide.new_tensor([SENTINEL])
    local=keys.new_zeros((ids.numel()+2048,1,576))
    if selected.numel(): local[:selected.numel()].copy_(keys[selected])
    torch.manual_seed(125+rank)
    q=torch.randn(TILE,64,576,device='cuda').to(torch.float8_e4m3fn)
    indices=torch.full((TILE,2048),-1,device='cuda',dtype=torch.int32)
    indices[:,:257]=wide[None,:]
    indices[0].fill_(-1)
    runner=QStreamAttention(Group());runner.identity=torch.ones(1,device='cuda')
    actual=runner.tile(q,indices,local,LocalLayout(ids,ids),576**-.5)
    full=keys.new_zeros((257+2048,1,576));full[:257].copy_(keys)
    positions=torch.full_like(indices,-1);positions[:,:257]=torch.arange(257,device='cuda');positions[0].fill_(-1)
    expected,_,_=sparse_mla_q8kv8_prefill_fwd(q,full,positions[:,None],576**-.5,runner.identity,runner.identity)
    # Existing native Q8 tolerance, unchanged.
    torch.testing.assert_close(actual,expected,atol=.08,rtol=.08)
    for row in (1,255):
        scores=(q[row].float()@keys[:,0].float().T)*(576**-.5)
        reference=scores.softmax(-1)@keys[:,0,:512].float()
        torch.testing.assert_close(actual[row].float(),reference,atol=.08,rtol=.08)
    assert torch.isfinite(actual).all()
    assert (actual[0]==0).all()
