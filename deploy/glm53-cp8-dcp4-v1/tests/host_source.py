"""Load actual host functions via AST without importing GPU dependencies.

Only host logic is exercised. Collectives, quantization and Triton launches are
mocked in tests; the separate upstream integration test imports real modules.
"""

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[3]))


def extract(path, names, class_name=None, namespace=None):
    source = ROOT / "python/sglang" / path
    parsed = ast.parse(source.read_text())
    nodes = parsed.body
    if class_name:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(body) == len(names), (path, names)
    module = types.ModuleType(path)
    module.__dict__.update(torch=torch, **(namespace or {}))
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)] + body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    return module


def stub(name):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(stub(parent), child, module)
    return sys.modules[name]


stub("sglang.kernels.ops.attention.dsa.quant_k_cache").quantize_k_cache_separate = None
stub("sgl_kernel.flash_mla").get_mla_metadata = None
dcp_comm = extract("srt/layers/dcp/comm.py", {
    "_all_gather_dcp_kv_cache", "all_gather_kv_cache_for_dcp",
    "all_gather_kv_cache_for_mla_extend",
}, namespace={"get_parallel": lambda: None})
dcp_planner = extract("srt/layers/dcp/planner.py", {
    "prepare_decode_context_parallel_metadata",
}, namespace={"get_parallel": lambda: None, "get_device": lambda: None,
              "create_dcp_kv_indices": None, "DecodeContextParallelMetadata": SimpleNamespace})
backend = extract("srt/layers/attention/dsa_backend.py", {
    "_build_dcp_prefill_page_table", "_flashmla_q_head_bucket", "_compute_flashmla_metadata",
}, class_name="DeepseekSparseAttnBackend", namespace={"DSAFlashMLAMetadata": SimpleNamespace})
DeepseekSparseAttnBackend = type("DeepseekSparseAttnBackend", (), {
    name: backend.__dict__[name] for name in (
        "_build_dcp_prefill_page_table", "_flashmla_q_head_bucket", "_compute_flashmla_metadata"
    )
})
lse = extract("kernels/ops/attention/dcp_kernels.py", {"_lse_weighted_combine_cpu"})
