"""Portable real-dispatch regression for -1 aliasing the last EP expert."""
import __future__,ast,collections,unittest
from pathlib import Path
from types import SimpleNamespace as N
SOURCE=Path(__file__).resolve().parents[3]/'python/sglang/srt/layers/moe/token_dispatcher/standard.py'
TREE=ast.parse(SOURCE.read_text())
class Vec(list):
    def __getitem__(self,index):
        if isinstance(index,Vec):return Vec(super(Vec,self).__getitem__(int(i)) for i in index)
        return super().__getitem__(index)
    def __ge__(self,other):return Vec(x>=other for x in self)
def where(mask,a,b):return Vec(a[i] if yes else b for i,yes in enumerate(mask))
TopK=collections.namedtuple('TopK','topk_weights topk_ids router_logits')
class TestMoEPaddingIDs(unittest.TestCase):
    def run_dispatch(self,rank,ids,enabled):
        namespace=dict(torch=N(where=where),should_use_flashinfer_cutlass_moe_fp4_allgather=lambda:False,TopKOutputChecker=N(format_is_standard=lambda _:True),_PRESERVE_MOE_PAD_IDS=enabled,_MASK_DP_PAD_MOE=False,StandardDispatchOutput=lambda **kw:N(**kw))
        helper=next((x for x in TREE.body if isinstance(x,ast.FunctionDef) and x.name=='_map_experts_preserving_padding'),None)
        if helper:
            helper.decorator_list=[];exec(compile(ast.Module(body=[helper],type_ignores=[]),str(SOURCE),'exec'),namespace)
        cls=next(x for x in TREE.body if isinstance(x,ast.ClassDef) and x.name=='StandardDispatcher');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='dispatch');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec',flags=__future__.annotations.compiler_flag),namespace)
        mapping=Vec(i-rank*32 if rank*32<=i<(rank+1)*32 else -1 for i in range(256));obj=N(moe_ep_size=8,skip_local_expert_mapping=False,local_expert_mapping=mapping,use_aiter_moe_runner=False);topk=TopK(object(),Vec(ids),None);result=namespace['dispatch'](obj,object(),topk).topk_output;self.assertIs(result.topk_weights,topk.topk_weights);return result.topk_ids
    def test_padded_and_valid_ids_on_every_rank(self):
        for rank in range(8):
            for ids in [[0,31,32,224,255,-1],[-1,-1]]:
                with self.subTest(rank=rank,ids=ids):
                    expected=[i-rank*32 if rank*32<=i<(rank+1)*32 else -1 for i in ids]
                    self.assertEqual(self.run_dispatch(rank,ids,True),expected)
    def test_control_switch_preserves_legacy_and_positive_ids(self):
        self.assertEqual(self.run_dispatch(7,[-1],False),[31])
        for rank in range(8):self.assertEqual(self.run_dispatch(rank,[0,31,32,224,255],False),self.run_dispatch(rank,[0,31,32,224,255],True))
if __name__=='__main__':unittest.main()
