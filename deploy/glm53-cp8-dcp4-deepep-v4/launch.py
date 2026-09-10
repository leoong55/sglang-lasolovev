"""Validate the v4 launch profile, then exec SGLang as the container process."""

import argparse
import json
import os
import sys
from pathlib import Path

from install import install, package_root


def validate(argv):
    p = argparse.ArgumentParser(allow_abbrev=False)
    for flag in (
        "model-path",
        "tp-size",
        "ep-size",
        "dp-size",
        "dcp-size",
        "cp-strategy",
        "dcp-comm-backend",
        "kv-cache-dtype",
        "page-size",
        "quantization",
        "dsa-prefill-backend",
        "dsa-decode-backend",
    ):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--enable-prefill-cp", action="store_true", required=True)
    p.add_argument("--enable-cp-decode-attn-tp", action="store_true", required=True)
    for flag, default in (
        ("moe-a2a-backend", "none"),
        ("chunked-prefill-size", "8192"),
        ("nnodes", "1"),
        ("pp-size", "1"),
        ("cuda-graph-backend-prefill", "disabled"),
    ):
        p.add_argument("--" + flag, default=default)
    args, remaining = p.parse_known_args(argv)
    expected = dict(
        tp_size="8",
        ep_size="8",
        dp_size="1",
        dcp_size="4",
        cp_strategy="interleave",
        dcp_comm_backend="ag_rs",
        kv_cache_dtype="fp8_e4m3",
        page_size="64",
        quantization="w4afp8",
        dsa_decode_backend="flashmla_kv",
        moe_a2a_backend="none",
        chunked_prefill_size="8192",
        nnodes="1",
        pp_size="1",
        cuda_graph_backend_prefill="disabled",
    )
    for key, value in expected.items():
        if getattr(args, key) != value:
            p.error(f"v4 profile requires --{key.replace('_', '-')} {value}")
    if args.dsa_prefill_backend not in ("flashmla_sparse_q8", "flashmla_kv"):
        p.error("v4 prefill must be flashmla_sparse_q8 or flashmla_kv (A/B reference)")
    if os.environ.get("SGLANG_ENABLE_CP_V2") != "1":
        p.error("v4 requires SGLANG_ENABLE_CP_V2=1")
    forbidden = {
        "--enable-two-batch-overlap",
        "--enable-single-batch-overlap",
        "--enable-eplb",
        "--enable-torch-compile",
        "--enable-symm-mem",
        "--enable-dp-lm-head",
    }
    for token in remaining:
        option = token.split("=", 1)[0]
        if (
            option.startswith("--speculative")
            or option.startswith("--hicache")
            or option in ("--enable-hierarchical-cache", "--enable-hisparse")
            or option in forbidden
        ):
            p.error(f"v4 profile does not support {option}")
    config_path = Path(args.model_path) / "config.json"
    config = json.loads(config_path.read_text())
    expected_shape = dict(num_hidden_layers=78, kv_lora_rank=512, qk_rope_head_dim=64)
    for key, value in expected_shape.items():
        if config.get(key) != value:
            p.error(
                f"Expected full GLM-5.3 model at {config_path}: {key} must be {value}, got {config.get(key)!r}"
            )
    mode = os.environ.get("SGLANG_GLM53_DEEPEP_PREFILL", "0")
    if mode not in ("0", "1"):
        p.error("SGLANG_GLM53_DEEPEP_PREFILL must be 0 or 1")
    print(
        f"glm53-cp8-dcp4-deepep-v4: DeepEP prefill={mode}; full GLM-5.3; TP8 EP8 CP8 DCP4 DP1; FP8 KV; prefill={args.dsa_prefill_backend}; decode=flashmla_kv; no HiCache/spec",
        flush=True,
    )


if __name__ == "__main__":
    install(package_root(), Path(__file__).resolve().parent, verify_only=True)
    validate(sys.argv[1:])
    os.execv(
        sys.executable, [sys.executable, "-m", "sglang.launch_server", *sys.argv[1:]]
    )
