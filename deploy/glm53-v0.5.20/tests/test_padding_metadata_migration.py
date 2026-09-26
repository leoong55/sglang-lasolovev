"""Regression for the first 0.5.20 CP/DCP prefill capture failure."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

@pytest.mark.parametrize('scattered,global_count,extend_count,expected',[
    (False,3,6,3), (True,7,2,2), (False,None,2,8), (False,8,2,8),
])
def test_padding_uses_0520_host_count_without_removed_field(scattered,global_count,extend_count,expected):
    path=ROOT/'python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_zero_dsa_dcp_padding')
    namespace={'torch':torch,'ForwardBatch':N,'get_attn_tp_context':lambda:N(input_scattered=scattered)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    # Deliberately no legacy num_token_non_padded_cpu attribute.
    batch=N(global_num_token_non_padded_cpu=global_count,extend_num_tokens=extend_count)
    values=torch.ones(8,2,4)
    namespace[fn.name](values,batch)
    assert torch.all(values[:expected]==1)
    assert torch.all(values[expected:]==0)
