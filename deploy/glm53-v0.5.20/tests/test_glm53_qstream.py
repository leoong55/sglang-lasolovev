import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location('qstream',ROOT/'python/sglang/srt/layers/dcp/glm53_qstream.py')
m=importlib.util.module_from_spec(s)
sys.modules[s.name]=m
s.loader.exec_module(m)

class QStreamTest(unittest.TestCase):
    def test_nonidentity_page_ids_compact_and_empty(self):
        # Two remapped pages; owner is encoded by the allocator read ID.
        local=torch.tensor([256,260,1024,1028,m.SENTINEL],dtype=torch.int32)
        idx=torch.tensor([[1028,-1,257,256,1024,261],[-1]*6],dtype=torch.int32)
        out,lens=m.compact_topk(idx,local)
        self.assertEqual(out.tolist(),[[3,0,2,-1,-1,-1],[-1]*6])
        self.assertEqual(lens.tolist(),[3,0])
        out,lens=m.compact_topk(idx,torch.tensor([m.SENTINEL],dtype=torch.int32))
        self.assertTrue((out == -1).all())
        self.assertTrue((lens == 0).all())

    def test_base2_reduction_matches_unsharded_softmax(self):
        torch.manual_seed(43)
        scores=torch.randn(7,3,31)*5
        values=torch.randn(31,9)
        scores[1,:,:]=-torch.inf
        mask=torch.zeros(4,31,dtype=torch.bool)
        mask[0,:10]=True;mask[1,10:17]=True;mask[2,17:]=True # rank3 empty
        outputs=[];lses=[]
        for rank in range(4):
            local=scores.masked_fill(~mask[rank],-torch.inf)
            probs=torch.nan_to_num(local.softmax(-1))
            outputs.append(probs@values)
            lses.append(torch.logsumexp(local,-1)/torch.log(torch.tensor(2.)))
        all_lse=torch.stack(lses)
        actual=sum(m.correction_base2(o,l,all_lse) for o,l in zip(outputs,lses))
        expected=torch.nan_to_num(scores.softmax(-1))@values
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        self.assertTrue(torch.isfinite(actual).all())
