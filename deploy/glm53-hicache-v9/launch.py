"""Explicit GLM53 CP/DCP and native TP profiles; preserve runtime arguments."""

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
    p.add_argument("--glm53-profile", choices=["cp8-dcp4", "tp8"], default="cp8-dcp4")
    p.add_argument("--model-path")
    p.add_argument("--chunked-prefill-size", required=True, type=int, choices=SUPPORTED_CHUNKS)
    p.add_argument("--max-running-requests", required=True, type=int)
    p.add_argument("--cuda-graph-backend-decode", required=True, choices=["full", "disabled"])
    p.add_argument("--cuda-graph-max-bs-decode", type=int)
    p.add_argument("--cuda-graph-bs-decode", nargs="+", type=int)
    p.add_argument("--disable-shared-experts-fusion", action="store_true", required=True)
    p.add_argument("--dsa-prefill-backend", required=True)
    p.add_argument("--moe-a2a-backend", required=True)
    p.add_argument("--moe-runner-backend", choices=["auto", "cutlass", "humming"], default="auto")
    p.add_argument("--cuda-graph-backend-prefill", required=True, choices=["breakable", "disabled"])
    p.add_argument("--cuda-graph-bs-prefill", nargs="+", type=int)
    p.add_argument("--cuda-graph-max-bs-prefill", type=int)
    p.add_argument("--speculative-algorithm", choices=["DFLASH", "EAGLE"])
    p.add_argument("--speculative-num-steps", type=int)
    p.add_argument("--speculative-eagle-topk", type=int)
    p.add_argument("--speculative-num-draft-tokens", type=int)
    p.add_argument("--speculative-dflash-block-size", type=int)
    p.add_argument("--glm53-dflash-graph-policy", choices=["warn", "require"], default="warn")
    p.add_argument("--glm53-dflash-profile-steps", type=int, default=0)
    p.add_argument("--glm53-dflash-profile-min-bs", type=int, default=1)
    p.add_argument("--glm53-draft-cache-window", type=int, choices=[0, 2048], default=0)
    p.add_argument("--glm53-cp-decode-fusion", choices=["off", "attention"], default="off")
    p.add_argument("--flashinfer-allreduce-fusion-backend")
    p.add_argument("--speculative-draft-model-path")
    p.add_argument("--speculative-draft-model-quantization", choices=["unquant", "w4afp8"])
    p.add_argument("--speculative-draft-attention-backend", choices=["fa4", "dsa"])
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
        or args.dsa_prefill_backend not in (
            ("flashmla_sparse_q8",) if args.glm53_profile == "cp8-dcp4"
            else ("flashmla_sparse_q8", "flashmla_kv")
        )
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
    if args.glm53_profile == "cp8-dcp4" and os.environ.get("SGLANG_ENABLE_CP_V2") != "1":
        p.error("SGLANG_ENABLE_CP_V2=1 is required")
    if args.speculative_algorithm == "DFLASH":
        if args.glm53_profile != "cp8-dcp4":
            p.error("This DFLASH profile requires cp8-dcp4; the tp8 comparison supports EAGLE or speculation off")
        if not all((args.speculative_draft_model_path, args.speculative_draft_model_quantization,
                    args.speculative_draft_attention_backend)):
            p.error("DFLASH requires explicit draft model path, unquant quantization and fa4 attention")
        if args.speculative_draft_model_quantization != "unquant" or args.speculative_draft_attention_backend != "fa4":
            p.error("DFLASH requires unquant draft and fa4 draft attention")
        if "GLM-5.3-Flash" in args.speculative_draft_model_path:
            p.error("Use the full GLM-5.3 draft, not GLM-5.3-Flash")
        if args.speculative_num_steps not in (None, 1) or args.speculative_eagle_topk not in (None, 1):
            p.error("DFLASH uses one parallel block; speculative-num-steps and speculative-eagle-topk must be 1 or omitted")
        widths = [n for n in (args.speculative_num_draft_tokens, args.speculative_dflash_block_size) if n is not None]
        if any(n < 2 for n in widths) or len(set(widths)) > 1:
            p.error("DFLASH block widths must agree and be at least 2")
        if args.glm53_draft_cache_window and any(n not in (2, 4, 8) for n in widths):
            p.error("Bounded GLM53 draft supports block size 2, 4 or 8")
    elif args.speculative_algorithm == "EAGLE":
        if args.glm53_profile != "tp8":
            p.error("EAGLE/MTP requires --glm53-profile tp8 and DCP1; the CP8/DCP4 speculative path implements DFLASH only")
        if args.speculative_draft_model_quantization != "w4afp8" or args.speculative_draft_attention_backend != "dsa":
            p.error("Native GLM53 MTP requires explicit w4afp8 draft quantization and dsa attention")
        if args.speculative_eagle_topk != 1 or not args.speculative_num_steps or args.speculative_num_steps < 1:
            p.error("This native EAGLE comparison requires topk=1 and positive explicit speculative-num-steps")
        if args.speculative_num_draft_tokens != args.speculative_num_steps + 1:
            p.error("Native EAGLE topk=1 requires speculative-num-draft-tokens = speculative-num-steps + 1")
        if args.speculative_dflash_block_size is not None:
            p.error("Remove DFLASH block options for EAGLE")
        validate_mtp_checkpoint(args.speculative_draft_model_path or args.model_path)
    elif any(x.startswith("--speculative-") for x in argv):
        p.error("To disable speculation, remove all --speculative-* options and their values")
    if args.glm53_dflash_profile_steps < 0 or args.glm53_dflash_profile_min_bs < 1:
        p.error("DFLASH profile steps must be nonnegative and minimum batch size positive")
    if args.speculative_algorithm != "DFLASH" and (
        args.glm53_dflash_profile_steps or args.glm53_dflash_graph_policy != "warn"
    ):
        p.error("DFLASH diagnostic options require --speculative-algorithm DFLASH")
    if args.glm53_dflash_graph_policy == "require" and (
        args.cuda_graph_backend_decode != "full"
        or args.cuda_graph_max_bs_decode < args.max_running_requests
    ):
        p.error("Required DFLASH graphs must cover max-running-requests with full decode graphs")
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
    if args.glm53_cp_decode_fusion == "attention":
        if args.glm53_profile != "cp8-dcp4":
            p.error("CP attention fusion requires the cp8-dcp4 profile")
        if args.speculative_algorithm or args.cuda_graph_backend_decode != "full":
            p.error("CP attention fusion currently requires full decode graphs and speculation off")
        if max(args.max_running_requests, args.cuda_graph_max_bs_decode) > 2048:
            p.error("CP attention fusion supports at most 2048 decode rows")
        if args.flashinfer_allreduce_fusion_backend not in (None, "trtllm"):
            p.error("CP attention fusion requires the trtllm allreduce backend")
        conflicts = {"--enforce-disable-flashinfer-allreduce-fusion", "--enable-deterministic-inference"}
        if any(x.split("=", 1)[0] in conflicts for x in argv):
            p.error("Remove conflicting deterministic/disabled/global-allreduce options for the CP fusion A/B")
    if args.moe_runner_backend == "humming":
        if args.glm53_cp_decode_fusion != "off":
            p.error("Keep --glm53-cp-decode-fusion off for the W4 MoE comparison")
    return args


def validate_mtp_checkpoint(model_path):
    """Fail before GPU loading if a target-only checkpoint omits native MTP."""
    if not model_path:
        raise ValueError("Native EAGLE requires a local GLM53 checkpoint")
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    if (config.get("num_nextn_predict_layers", 0) < 1 or config.get("num_hidden_layers") != 78
            or "GlmMoeDsaForCausalLM" not in config.get("architectures", [])):
        raise ValueError("Native GLM53 EAGLE requires num_nextn_predict_layers >= 1 and 78 target layers")
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"Cannot verify MTP tensors without {index_path}; do not substitute the DFlash2 checkpoint")
    weights = json.loads(index_path.read_text())["weight_map"]
    prefix = "model.layers.78."
    required = ("eh_proj.", "enorm.", "hnorm.", "shared_head.norm.", "self_attn.", "mlp.")
    for part in required:
        matches = [name for name in weights if name.startswith(prefix + part)]
        if not matches:
            raise ValueError(f"Checkpoint is missing native MTP weights: {prefix}{part}*")
        for name in matches:
            if not (root / weights[name]).is_file():
                raise ValueError(f"Missing MTP shard: {weights[name]}")


def runtime_argv(argv):
    """Consume launcher-owned options; request workspace allocation for fusion."""
    result = []
    owned = {"--glm53-draft-cache-window", "--glm53-cp-decode-fusion", "--glm53-profile",
             "--glm53-dflash-graph-policy", "--glm53-dflash-profile-steps", "--glm53-dflash-profile-min-bs"}
    fusion = "off"
    i = 0
    while i < len(argv):
        token = argv[i]
        option = token.split("=", 1)[0]
        if option == "--glm53-cp-decode-fusion":
            fusion = token.split("=", 1)[1] if "=" in token else argv[i + 1]
        if token in owned:
            i += 2
            continue
        if option in owned and "=" in token:
            i += 1
            continue
        result.append(token)
        i += 1
    if fusion == "attention" and not any(x.split("=", 1)[0] == "--flashinfer-allreduce-fusion-backend" for x in result):
        result += ["--flashinfer-allreduce-fusion-backend", "trtllm"]
    return result


def configure_runtime_env(profile):
    """Opt-in guards follow selected features; never enable a removed feature."""
    os.environ.pop("SGLANG_GLM53_DEEPEP_PREFILL", None)
    os.environ.pop("SGLANG_GLM53_DEEPEP_OPT", None)
    cp = profile.glm53_profile == "cp8-dcp4"
    os.environ["SGLANG_ENABLE_CP_V2"] = "1" if cp else "0"
    os.environ["SGLANG_GLM53_PREFILL_BCG"] = "1" if cp and profile.cuda_graph_backend_prefill == "breakable" else "0"
    os.environ["SGLANG_GLM53_DFLASH_DCP"] = "1" if profile.speculative_algorithm == "DFLASH" else "0"
    os.environ["SGLANG_GLM53_HICACHE_DCP"] = "1" if cp and profile.enable_hierarchical_cache else "0"
    os.environ["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
    os.environ["SGLANG_GLM53_DRAFT_CACHE_WINDOW"] = str(profile.glm53_draft_cache_window)
    os.environ["SGLANG_GLM53_CP_DECODE_FUSION"] = "1" if profile.glm53_cp_decode_fusion == "attention" else "0"
    os.environ["SGLANG_GLM53_DFLASH_GRAPH_POLICY"] = profile.glm53_dflash_graph_policy
    os.environ["SGLANG_GLM53_DFLASH_PROFILE_STEPS"] = str(profile.glm53_dflash_profile_steps)
    os.environ["SGLANG_GLM53_DFLASH_PROFILE_MIN_BS"] = str(profile.glm53_dflash_profile_min_bs)
    for name, enabled in (
        ("SGLANG_GLM53_HUMMING_EP_AWARE", profile.moe_runner_backend == "humming"),
        ("SGLANG_GLM53_PREFILL_BCG_PADDING", cp and profile.cuda_graph_backend_prefill == "breakable"),
        ("SGLANG_GLM53_HICACHE_INDEX_ELISION", cp and profile.enable_hierarchical_cache),
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
    print(f"glm53: profile={profile.glm53_profile}; speculation={profile.speculative_algorithm or 'off'}; "
          f"moe-backend={'cutlass' if profile.moe_runner_backend == 'auto' else profile.moe_runner_backend}; "
          f"cp-decode-fusion={profile.glm53_cp_decode_fusion}; "
          f"hicache={profile.enable_hierarchical_cache}; max-running={profile.max_running_requests}; "
          f"bounded-draft-window={profile.glm53_draft_cache_window}; "
          f"hicache-index-elision={os.environ['SGLANG_GLM53_HICACHE_INDEX_ELISION']}; "
          f"bounded-draft-fastpath={os.environ['SGLANG_GLM53_BOUNDED_DRAFT_FASTPATH']}; "
          f"prefill-graphs={profile.cuda_graph_backend_prefill}; "
          f"decode-graphs={profile.cuda_graph_backend_decode}; "
          f"decode-max-bs={profile.cuda_graph_max_bs_decode}; "
          f"prefill-buckets={profile.cuda_graph_bs_prefill}; "
          f"humming-ep-aware={os.environ.get('SGLANG_GLM53_HUMMING_EP_AWARE', '1')}; "
          f"prefill-padding={os.environ.get('SGLANG_GLM53_PREFILL_BCG_PADDING', '1')}", flush=True)
    print("glm53-hicache-v9: effective argv=" + json.dumps(argv), flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *argv])
