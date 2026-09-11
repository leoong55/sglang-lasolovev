"""Explicit CP8/DCP4 chunk profile; never replace arguments supplied by YAML."""

import argparse
import json
import os
import sys
from pathlib import Path

from baseline_profile import validate
from install import install, package_root

SUPPORTED_CHUNKS = (8192, 16384, 32768)


def configure(argv):
    seen = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option in seen:
            raise ValueError(f"Duplicate option {option}; provide one explicit value")
        seen.add(option)
    return list(argv)


def check_profile(argv):
    validate(argv)
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--chunked-prefill-size", required=True, type=int, choices=SUPPORTED_CHUNKS)
    p.add_argument("--max-running-requests", required=True, type=int)
    p.add_argument("--cuda-graph-backend-decode", required=True)
    p.add_argument("--cuda-graph-max-bs-decode", required=True, type=int)
    p.add_argument("--disable-shared-experts-fusion", action="store_true", required=True)
    p.add_argument("--dsa-prefill-backend", required=True)
    p.add_argument("--moe-a2a-backend", required=True)
    p.add_argument("--cuda-graph-backend-prefill", required=True)
    p.add_argument("--cuda-graph-bs-prefill", required=True, nargs="+", type=int)
    p.add_argument("--cuda-graph-max-bs-prefill", required=True, type=int)
    args, _ = p.parse_known_args(argv)
    buckets = args.cuda_graph_bs_prefill
    if (
        buckets != sorted(set(buckets))
        or any(size not in SUPPORTED_CHUNKS for size in buckets)
        or max(buckets) != args.chunked_prefill_size
        or args.cuda_graph_max_bs_prefill != args.chunked_prefill_size
    ):
        p.error("Prefill buckets must be a sorted unique subset of 8192/16384/32768; their maximum and --cuda-graph-max-bs-prefill must equal --chunked-prefill-size")
    if (
        args.max_running_requests != 32
        or args.cuda_graph_backend_decode != "full"
        or args.cuda_graph_max_bs_decode != 32
        or args.dsa_prefill_backend != "flashmla_sparse_q8"
        or args.moe_a2a_backend != "none"
        or args.cuda_graph_backend_prefill != "breakable"
    ):
        p.error("Requires max-running32, full decode graphs/max32, Q8 prefill, A2A none and breakable prefill graphs")
    forbidden = {
        "--enable-two-batch-overlap", "--enable-single-batch-overlap", "--enable-eplb",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    }
    if any(x.split("=", 1)[0] in forbidden for x in argv):
        p.error("Remove overlap/EPLB/legacy graph override flags for this experiment")
    if os.environ.get("SGLANG_ENABLE_CP_V2") != "1":
        p.error("SGLANG_ENABLE_CP_V2=1 is required")
    p_spec = argparse.ArgumentParser(allow_abbrev=False)
    p_spec.add_argument("--speculative-algorithm", required=True, choices=["DFLASH"])
    p_spec.add_argument("--speculative-draft-model-path", required=True)
    p_spec.add_argument("--speculative-draft-model-quantization", required=True, choices=["unquant"])
    p_spec.add_argument("--speculative-draft-attention-backend", required=True, choices=["fa4"])
    spec, _ = p_spec.parse_known_args(argv)
    if "GLM-5.3-Flash" in spec.speculative_draft_model_path:
        p_spec.error("Use the full GLM-5.3 draft, not GLM-5.3-Flash")
    p_cache = argparse.ArgumentParser(allow_abbrev=False)
    p_cache.add_argument("--enable-hierarchical-cache", action="store_true", required=True)
    p_cache.add_argument("--hicache-write-policy", required=True, choices=["write_through"])
    p_cache.add_argument("--hicache-mem-layout", required=True, choices=["layer_first"])
    p_cache.add_argument("--hicache-io-backend", required=True, choices=["direct"])
    p_cache.add_argument("--hicache-host-memory-mode", required=True, choices=["cache"])
    p_cache.parse_known_args(argv)
    return args


if __name__ == "__main__":
    argv = configure(sys.argv[1:])
    install(package_root(), Path(__file__).resolve().parent, verify_only=True)
    profile = check_profile(argv)
    os.environ.pop("SGLANG_GLM53_DEEPEP_PREFILL", None)
    os.environ.pop("SGLANG_GLM53_DEEPEP_OPT", None)
    os.environ["SGLANG_GLM53_PREFILL_BCG"] = "1"
    os.environ["SGLANG_GLM53_DFLASH_DCP"] = "1"
    os.environ["SGLANG_GLM53_HICACHE_DCP"] = "1"
    os.environ["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    print("glm53-hicache-v9: global buckets=" + str(profile.cuda_graph_bs_prefill)
          + "; local rows=" + str([n // 8 for n in profile.cuda_graph_bs_prefill]), flush=True)
    print("glm53-hicache-v9: effective argv=" + json.dumps(argv), flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *argv])
