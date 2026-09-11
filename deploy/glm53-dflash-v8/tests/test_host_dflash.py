"""Exercise actual BCG and acceptance functions with CPU tensors."""
import ast
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang"


def extract(path, names, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class DFlashIntegrationTest(unittest.TestCase):
    def test_prefill_graph_gathers_packed_and_list_aux_features(self):
        seen = []
        def gather(tensor, *args):
            seen.append(tensor.clone())
            return torch.cat([tensor + rank * 100 for rank in range(8)], dim=0)
        ns = extract("srt/layers/cp/bcg.py", {"execute_prefill_cp_bcg", "_slice_output_rows"}, {
            "torch":torch, "PPProxyTensors":type("Proxy", (), {}),
            "cp_gather_after_forward":gather,
        })
        local = torch.arange(6).view(2, 3).float()
        aux = torch.arange(12).view(2, 6).float()
        model = NS(capture_aux_hidden_states=True,pp_group=NS(is_last_rank=True),lm_head=object())
        model.logits_processor = lambda ids, hidden, head, batch, features:(hidden,features)
        runner = NS(prefill_cp_bcg_input=NS(live_local_tokens=2),
                    model_runner=NS(model=model,server_args=None),
                    _prefill_forward_context=lambda *a,**k:nullcontext())
        modules={
            "sglang.srt.model_executor.runner.shape_key":NS(ShapeKey=lambda **kw:kw),
            "sglang.srt.layers.cp.glm53_bcg":NS(supports=lambda _:False),
        }
        batch=NS(input_ids=torch.arange(16))
        with patch.dict(sys.modules,modules), patch.object(torch.cuda,"current_stream",return_value=None):
            for features in (aux, [aux[:,:3],aux[:,3:]]):
                runner.backend=NS(replay=lambda *a,**k:(local,features))
                hidden,global_features=ns["execute_prefill_cp_bcg"](runner,batch,batch,16,16)
                self.assertEqual(hidden.shape[0],16)
                if isinstance(global_features,list):
                    self.assertEqual([x.shape[0] for x in global_features],[16,16])
                else:
                    self.assertEqual(global_features.shape,(16,6))
        self.assertEqual(len(seen),5)

    def test_accept_reject_and_bonus_keep_committed_prefix_only(self):
        ns=extract("srt/speculative/dflash_worker_v2.py",{"_commit_accept"},{"torch":torch})
        candidates=torch.tensor([[9,10,11,12],[8,20,21,22],[7,30,31,32]])
        out,lengths=ns["_commit_accept"](candidates,torch.tensor([0,1,3]),torch.tensor([99,98,97]))
        self.assertEqual(lengths.tolist(),[1,2,4])
        expected=[[99],[20,98],[30,31,32,97]]
        for row,n in enumerate(lengths.tolist()):
            self.assertEqual(out[row,:n].tolist(),expected[row])
        # Accepted target-input rows occupy the original widened virtual IDs.
        # Rejected rows and the bonus (whose target KV does not exist yet) must
        # not be added as extra committed cache locations.
        virtual=torch.tensor([[256,257,258,259],[512,513,514,515],[768,769,770,771]])
        for row,n in enumerate(lengths.tolist()):
            ownership=[[] for _ in range(4)]
            for loc in virtual[row,:n].tolist(): ownership[loc%4].append(loc//4)
            restored=sorted(local*4+rank for rank,locs in enumerate(ownership) for local in locs)
            self.assertEqual(restored,virtual[row,:n].tolist())

    def test_verify_host_lengths_restore_even_when_metadata_fails(self):
        source=ROOT/"srt/speculative/dflash_worker_v2.py"
        tree=ast.parse(source.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="forward_batch_generation")
        block=next(n for n in ast.walk(method) if isinstance(n,ast.Try)
                   and any(isinstance(x,ast.Attribute) and x.attr=="prepare_for_verify" for x in ast.walk(n)))
        batch=NS(seq_lens_cpu="expanded",seq_lens_sum=999)
        def fail(*args): raise RuntimeError("metadata failed")
        ns={"batch":batch,"self":NS(target_worker=object()),"verify_input":NS(prepare_for_verify=fail),
            "seq_lens_cpu_backup":"committed","seq_lens_sum_backup":42}
        with self.assertRaisesRegex(RuntimeError,"metadata failed"):
            exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),"exec"),ns)
        self.assertEqual((batch.seq_lens_cpu,batch.seq_lens_sum),("committed",42))


if __name__ == "__main__": unittest.main()
