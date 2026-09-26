"""Run in the pinned SM90 image: pytest -q tests/cuda/test_compact_logits.py."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM90 CUDA required")

@pytest.mark.parametrize('q_rows', [4, 37, 256])
def test_request_relative_logits_and_topk(q_rows):
    import deep_gemm
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('SM90 only')
    torch.manual_seed(713)
    sizes = [256, 513, 1024]
    q = torch.randn(q_rows, 32, 128, device='cuda').to(torch.float8_e4m3fn)
    kv = torch.randn(sum(sizes), 128, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.ones(sum(sizes), device='cuda')
    weights = torch.randn(q_rows, 32, device='cuda')
    req = torch.arange(q_rows, device='cuda') % len(sizes)
    starts = torch.tensor([0, 256, 769], dtype=torch.int32, device='cuda')[req]
    lengths = torch.tensor(sizes, dtype=torch.int32, device='cuda')[req]
    lengths = torch.minimum(lengths, 1 + (torch.arange(q_rows,device='cuda')*47).int())
    ends = starts + lengths
    args = (q, (kv, scales), weights, starts, ends)
    old = deep_gemm.fp8_mqa_logits(*args, clean_logits=False)
    new = deep_gemm.fp8_mqa_logits(*args, clean_logits=False, max_seqlen_k=max(sizes))
    cols = torch.arange(max(sizes),device='cuda')[None,:]
    mask = cols < lengths[:,None]
    old_local = old.gather(1,(starts[:,None]+cols).clamp_max(old.shape[1]-1))
    # Same kernel arithmetic; changing the store location must be bit-exact.
    torch.testing.assert_close(new[mask],old_local[mask],rtol=0,atol=0)
    a = new.masked_fill(~mask,-torch.inf).topk(64,dim=-1).indices
    b = old_local.masked_fill(~mask,-torch.inf).topk(64,dim=-1).indices
    torch.testing.assert_close(a,b,rtol=0,atol=0)
