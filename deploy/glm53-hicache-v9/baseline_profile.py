"""Validate the v3 launch profile, then exec SGLang as the container process."""

import argparse
import json
import os
import sys
from pathlib import Path

from install import install, package_root


def validate(argv, *, target_quantization="w4afp8"):
    if target_quantization not in ("w4afp8", "fp8"):
        raise ValueError("Unsupported GLM53 target quantization")
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--glm53-profile", choices=["cp8-dcp4", "tp8"], default="cp8-dcp4")
    for flag in (
        "model-path",
        "tp-size",
        "ep-size",
        "dp-size",
        "dcp-size",
        "dcp-comm-backend",
        "kv-cache-dtype",
        "page-size",
        "dsa-prefill-backend",
        "dsa-decode-backend",
    ):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--quantization", required=target_quantization == "w4afp8")
    p.add_argument("--cp-strategy")
    p.add_argument("--enable-prefill-cp", action="store_true")
    p.add_argument("--enable-cp-decode-attn-tp", action="store_true")
    args, remaining = p.parse_known_args(argv)
    expected = dict(
        tp_size="8",
        ep_size="8",
        dp_size="1",
        dcp_size="4" if args.glm53_profile == "cp8-dcp4" else "1",
        dcp_comm_backend="ag_rs",
        kv_cache_dtype="fp8_e4m3",
        page_size="64",
        dsa_decode_backend="flashmla_kv",
    )
    if args.glm53_profile == "cp8-dcp4":
        if not (args.enable_prefill_cp and args.enable_cp_decode_attn_tp and args.cp_strategy == "interleave"):
            p.error("cp8-dcp4 requires prefill CP, interleave and CP decode attention TP")
    elif args.enable_prefill_cp or args.enable_cp_decode_attn_tp or args.cp_strategy:
        p.error("tp8 requires removing --enable-prefill-cp, --cp-strategy and --enable-cp-decode-attn-tp")
    for key, value in expected.items():
        if getattr(args, key) != value:
            p.error(f"{args.glm53_profile} profile requires --{key.replace('_', '-')} {value}")
    if args.dsa_prefill_backend not in ("flashmla_sparse_q8", "flashmla_kv"):
        p.error("v3 prefill must be flashmla_sparse_q8 or flashmla_kv (A/B reference)")
    for token in remaining:
        option = token.split("=", 1)[0]
        if option in ("--enable-hisparse", "--enable-lmcache", "--hicache-storage-backend"):
            p.error("This profile supports L1/L2 HiCache only")
    config_path = Path(args.model_path) / "config.json"
    config = json.loads(config_path.read_text())
    if target_quantization == "fp8":
        quant = config.get("quantization_config") or {}
        if (
            args.quantization not in (None, "fp8")
            or quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
            or quant.get("activation_scheme") != "dynamic"
            or config.get("architectures") != ["GlmMoeDsaForCausalLM"]
        ):
            p.error("FP8 profile requires the full GLM53 serialized FP8 checkpoint, dynamic activations and 128x128 weight blocks; omit --quantization or use fp8")
    elif args.quantization != "w4afp8":
        p.error("W4 profile requires --quantization w4afp8")
    expected_shape = dict(num_hidden_layers=78, kv_lora_rank=512, qk_rope_head_dim=64)
    for key, value in expected_shape.items():
        if config.get(key) != value:
            p.error(
                f"Expected full GLM-5.3 model at {config_path}: {key} must be {value}, got {config.get(key)!r}"
            )
    print(
        f"glm53: full GLM-5.3; profile={args.glm53_profile}; TP8 EP8 DP1 DCP{args.dcp_size}; FP8 KV; prefill={args.dsa_prefill_backend}; decode=flashmla_kv",
        flush=True,
    )


if __name__ == "__main__":
    install(package_root(), Path(__file__).resolve().parent, verify_only=True)
    validate(sys.argv[1:])
    os.execv(
        sys.executable, [sys.executable, "-m", "sglang.launch_server", *sys.argv[1:]]
    )
