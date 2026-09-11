"""Compile CP/DCP bridge kernels for H200 without a GPU or model weights.

Run against the installed image by default, or --package-root python/sglang.
This is a compiler gate, not CUDA execution or full-model validation.
"""

import argparse
import importlib.util
import inspect
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def load_source(package_root, relative):
    path = Path(package_root) / relative
    spec = importlib.util.spec_from_file_location("glm53_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_kernel(fn, signature, constants):
    # Triton 3.1 uses positional signature/constant keys; newer versions use
    # named signatures and the constexprs argument. Inspect the installed API.
    if "constexprs" in inspect.signature(ASTSource).parameters:
        source = ASTSource(fn, signature=signature, constexprs=constants)
    else:
        source = ASTSource(
            fn,
            signature={fn.arg_names.index(k): v for k, v in signature.items()},
            constants={fn.arg_names.index(k): v for k, v in constants.items()},
        )
    result = triton.compile(source, target=GPUTarget("cuda", 90, 32))
    if not result.asm.get("ptx") or not result.asm.get("cubin"):
        raise RuntimeError(f"No SM90 PTX/cubin produced for {fn.__name__}")
    return result


def cases(package_root):
    split = load_source(package_root, "kernels/ops/attention/dsa/cp_split.py")
    dcp = load_source(package_root, "kernels/ops/attention/dcp_kernels.py")
    transform = load_source(package_root, "kernels/ops/attention/dsa/transform_index.py")
    for dtype in ("i32", "i64"):
        for requests in (1, 32, 64, 96):
            for rank in (0, 7):
                yield (
                    split.dsa_cp_round_robin_split_q_seqs_kernel,
                    dict(in_seqs_ptr="*" + dtype, out_seqs_ptr="*" + dtype, bs_idx_ptr="*i32"),
                    dict(tokens=requests, cp_size=8, cp_rank=rank),
                )
        for name, constants in (
            ("create_dcp_kv_indices", dict(dcp_world_size=4)),
            ("update_kv_lens_and_indices", dict(dcp_rank=3, dcp_world_size=4, BLOCK_SIZE=512)),
            ("create_mla_kv_page_table_for_dcp", dict(
                req_to_token_stride=32768, block_table_stride=128,
                PHYSICAL_PAGE_SIZE=64, DCP_SIZE=4, DCP_RANK=3, PAGES_PER_BLOCK=32,
            )),
            ("create_triton_kv_indices_for_dcp_triton", dict(
                req_to_token_ptr_stride=32768, dcp_size=4, dcp_rank=3,
            )),
        ):
            fn = getattr(dcp, name)
            signature = {arg: "*" + dtype for arg in fn.arg_names if arg not in constants}
            if "extend_prefix_lens_sum" in signature:
                signature["extend_prefix_lens_sum"] = dtype
            yield fn, signature, constants
        for expanded, causal, block_q, size in ((False, False, 4, 1), (True, True, 1, 4)):
            fn = transform.transform_index_page_table_prefill_kernel
            constants = dict(
                page_table_stride_0=32768, page_table_stride_1=1,
                topk_indices_stride_0=2048, topk_indices_stride_1=1,
                result_stride_0=2048, result_stride_1=1,
                PAGE_TABLE_IS_EXPANDED=expanded, HAS_CAUSAL_LENS=causal,
                PAGE_TABLE_WIDTH=32768, TOPK=2048, BLOCK_Q=block_q,
                BLOCK_TOPK=256, dcp_size=size, dcp_rank=size - 1,
            )
            signature = {arg: "*i32" for arg in fn.arg_names if arg not in constants}
            signature["page_table_ptr"] = "*" + dtype
            signature["cu_seqlens_q_ptr"] = "*" + dtype
            yield fn, signature, constants
        yield transform.transform_index_page_table_decode_kernel, dict(
            page_table_ptr="*" + dtype, topk_indices_ptr="*i32", result_ptr="*i32",
        ), dict(page_size=1, page_table_row_stride=32768, dcp_size=4, dcp_rank=3)

    for natural_lse in (False, True):
        fn = dcp._correct_attn_cp_out_kernel
        constants = dict(HEAD_DIM=512, N_ROUNDED=4, IS_LSE_BASE_ON_E=natural_lse)
        signature = {arg: "i64" for arg in fn.arg_names if arg not in constants}
        signature.update(outputs_ptr="*bf16", new_output_ptr="*bf16", lses_ptr="*fp32", vlse_ptr="*fp32")
        yield fn, signature, constants
        fn = dcp._dcp_lse_combine_kernel
        constants = dict(N=4, HEAD_DIM=512, IS_BASE_E=natural_lse, RETURN_LSE=True)
        signature = {arg: "i64" for arg in fn.arg_names if arg not in constants}
        signature.update(recv_output_ptr="*bf16", recv_lse_ptr="*fp32", out_ptr="*bf16", out_lse_ptr="*fp32")
        yield fn, signature, constants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    root = args.package_root
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            parser.error("Cannot find installed sglang; pass --package-root")
        root = Path(next(iter(spec.submodule_search_locations)))
    print(f"Compiler preflight: torch={torch.__version__}, triton={triton.__version__}, target=sm90", flush=True)
    count = 0
    for fn, signature, constants in cases(root):
        compile_kernel(fn, signature, constants)
        count += 1
        print(f"PASS {count}: {fn.__name__} {constants}", flush=True)
    print(f"Compiled {count} SM90 specializations; GPU execution remains untested.", flush=True)


if __name__ == "__main__":
    main()
