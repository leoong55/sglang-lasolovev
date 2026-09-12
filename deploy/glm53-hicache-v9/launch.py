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
    p.add_argument("--cuda-graph-backend-decode", required=True, choices=["full", "disabled"])
    p.add_argument("--cuda-graph-max-bs-decode", type=int)
    p.add_argument("--cuda-graph-bs-decode", nargs="+", type=int)
    p.add_argument("--disable-shared-experts-fusion", action="store_true", required=True)
    p.add_argument("--dsa-prefill-backend", required=True)
    p.add_argument("--moe-a2a-backend", required=True)
    p.add_argument("--cuda-graph-backend-prefill", required=True, choices=["breakable", "disabled"])
    p.add_argument("--cuda-graph-bs-prefill", nargs="+", type=int)
    p.add_argument("--cuda-graph-max-bs-prefill", type=int)
    p.add_argument("--speculative-algorithm", choices=["DFLASH"])
    p.add_argument("--glm53-draft-cache-window", type=int, choices=[0, 2048], default=0)
    p.add_argument("--speculative-draft-model-path")
    p.add_argument("--speculative-draft-model-quantization", choices=["unquant"])
    p.add_argument("--speculative-draft-attention-backend", choices=["fa4"])
    p.add_argument("--enable-hierarchical-cache", action="store_true")
    p.add_argument("--hicache-write-policy", choices=["write_through"])
    p.add_argument("--hicache-mem-layout", choices=["layer_first"])
    p.add_argument("--hicache-io-backend", choices=["direct"])
    p.add_argument("--hicache-host-memory-mode", choices=["cache"])
    args, _ = p.parse_known_args(argv)
    buckets = args.cuda_graph_bs_prefill
    if args.cuda_graph_backend_prefill == "breakable" and (
        not buckets
        or buckets != sorted(set(buckets))
        or any(size not in SUPPORTED_CHUNKS for size in buckets)
        or max(buckets) != args.chunked_prefill_size
        or args.cuda_graph_max_bs_prefill != args.chunked_prefill_size
    ):
        p.error("Prefill buckets must be a sorted unique subset of 8192/16384/32768; their maximum and --cuda-graph-max-bs-prefill must equal --chunked-prefill-size")
    if (
        args.max_running_requests <= 0
        or args.dsa_prefill_backend != "flashmla_sparse_q8"
        or args.moe_a2a_backend != "none"
    ):
        p.error("Requires positive max-running-requests, Q8 prefill and A2A none")
    if args.cuda_graph_backend_decode == "full":
        if args.cuda_graph_max_bs_decode is None or args.cuda_graph_max_bs_decode <= 0:
            p.error("Full decode graphs require a positive --cuda-graph-max-bs-decode")
        decode_buckets = args.cuda_graph_bs_decode
        if decode_buckets is not None and (
            decode_buckets != sorted(set(decode_buckets))
            or any(n <= 0 for n in decode_buckets)
            or decode_buckets[-1] != args.cuda_graph_max_bs_decode
        ):
            p.error("Decode buckets must be sorted, unique, positive and end at --cuda-graph-max-bs-decode")
    forbidden = {
        "--enable-two-batch-overlap", "--enable-single-batch-overlap", "--enable-eplb",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-prefill-cuda-graph", "--disable-decode-cuda-graph", "--cuda-graph-config",
    }
    if any(x.split("=", 1)[0] in forbidden for x in argv):
        p.error("Remove overlap/EPLB flags; select graphs explicitly with --cuda-graph-backend-prefill/decode, including disabled")
    if os.environ.get("SGLANG_ENABLE_CP_V2") != "1":
        p.error("SGLANG_ENABLE_CP_V2=1 is required")
    if args.speculative_algorithm == "DFLASH":
        if not all((args.speculative_draft_model_path, args.speculative_draft_model_quantization,
                    args.speculative_draft_attention_backend)):
            p.error("DFLASH requires explicit draft model path, unquant quantization and fa4 attention")
        if "GLM-5.3-Flash" in args.speculative_draft_model_path:
            p.error("Use the full GLM-5.3 draft, not GLM-5.3-Flash")
    elif any(x.startswith("--speculative-") for x in argv):
        p.error("To disable speculation, remove all --speculative-* options and their values")
    if args.enable_hierarchical_cache and not all((
        args.hicache_write_policy, args.hicache_mem_layout,
        args.hicache_io_backend, args.hicache_host_memory_mode,
    )):
        p.error("HiCache requires explicit write_through, layer_first, direct and cache settings")
    if args.glm53_draft_cache_window and args.speculative_algorithm != "DFLASH":
        p.error("--glm53-draft-cache-window requires DFLASH")
    if args.glm53_draft_cache_window:
        unsupported = {"--enable-unified-memory", "--disaggregation-mode", "--enable-pdmux",
                       "--enable-memory-saver", "--speculative-draft-window-size"}
        if any(x.split("=", 1)[0] in unsupported for x in argv):
            p.error("Bounded draft requires the ordinary colocated pool; remove unified-memory, disaggregation, memory-saver and separate draft-window overrides")
    return args


def runtime_argv(argv):
    """Consume only the launcher-owned bounded-cache option."""
    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--glm53-draft-cache-window":
            i += 2
            continue
        if token.startswith("--glm53-draft-cache-window="):
            i += 1
            continue
        result.append(token)
        i += 1
    return result


def configure_runtime_env(profile):
    """Opt-in guards follow selected features; never enable a removed feature."""
    os.environ.pop("SGLANG_GLM53_DEEPEP_PREFILL", None)
    os.environ.pop("SGLANG_GLM53_DEEPEP_OPT", None)
    os.environ["SGLANG_GLM53_PREFILL_BCG"] = "1" if profile.cuda_graph_backend_prefill == "breakable" else "0"
    os.environ["SGLANG_GLM53_DFLASH_DCP"] = "1" if profile.speculative_algorithm == "DFLASH" else "0"
    os.environ["SGLANG_GLM53_HICACHE_DCP"] = "1" if profile.enable_hierarchical_cache else "0"
    os.environ["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    os.environ["SGLANG_GLM53_DRAFT_CACHE_WINDOW"] = str(profile.glm53_draft_cache_window)
    for name, enabled in (
        ("SGLANG_GLM53_HICACHE_INDEX_ELISION", profile.enable_hierarchical_cache),
        ("SGLANG_GLM53_BOUNDED_DRAFT_FASTPATH", bool(profile.glm53_draft_cache_window)),
    ):
        value = os.environ.get(name, "1")
        if value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        os.environ[name] = value if enabled else "0"


if __name__ == "__main__":
    argv = configure(sys.argv[1:])
    install(package_root(), Path(__file__).resolve().parent, verify_only=True)
    profile = check_profile(argv)
    configure_runtime_env(profile)
    argv = runtime_argv(argv)
    print(f"glm53: speculation={profile.speculative_algorithm or 'off'}; "
          f"hicache={profile.enable_hierarchical_cache}; max-running={profile.max_running_requests}; "
          f"bounded-draft-window={profile.glm53_draft_cache_window}; "
          f"hicache-index-elision={os.environ['SGLANG_GLM53_HICACHE_INDEX_ELISION']}; "
          f"bounded-draft-fastpath={os.environ['SGLANG_GLM53_BOUNDED_DRAFT_FASTPATH']}; "
          f"prefill-graphs={profile.cuda_graph_backend_prefill}; "
          f"decode-graphs={profile.cuda_graph_backend_decode}; "
          f"decode-max-bs={profile.cuda_graph_max_bs_decode}; "
          f"prefill-buckets={profile.cuda_graph_bs_prefill}", flush=True)
    print("glm53-hicache-v9: effective argv=" + json.dumps(argv), flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *argv])
