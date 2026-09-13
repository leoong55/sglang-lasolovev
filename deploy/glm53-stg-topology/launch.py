"""Launch one fixed, auditable STG topology; no arbitrary SGLang argument passthrough."""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

BASE_COMMIT = "0bcd822377da7b5718e674eaf9c870d349424dd1"
PROFILES = {
    "pp2": {"tp": 4, "pp": 2, "dp": 1, "ep": 4},
    "dpa2": {"tp": 8, "pp": 1, "dp": 2, "ep": 8},
    "dpa4": {"tp": 8, "pp": 1, "dp": 4, "ep": 8},
    "dpa8": {"tp": 8, "pp": 1, "dp": 8, "ep": 8},
}


# These are opt-in experiment overrides, not ordinary NCCL/device/cache settings.
# Reject even a disabled inherited value, so the launch record has one source of truth.
def conflicting_env(key):
    if not key.startswith("SGLANG_"):
        return False
    suffix = key.removeprefix("SGLANG_")
    parts = set(suffix.split("_"))
    return bool(
        parts & {"CP", "DCP", "DFLASH", "DSPARK", "SPECULATIVE", "SPEC", "DEEPEP"}
        or suffix.startswith(("PP_", "HICACHE_", "DEEP_EP_"))
        or "CONTEXT_PARALLEL" in suffix
        or suffix in {"EXTRA_ARGS", "SERVER_ARGS", "LAUNCH_ARGS"}
    )


def make_launch(profile, model_path, hicache=False, port=8080, environ=None):
    env = dict(os.environ if environ is None else environ)
    conflicts = sorted(key for key in env if conflicting_env(key))
    if conflicts:
        raise ValueError(
            "Inherited experiment overrides are forbidden: " + ", ".join(conflicts)
        )
    if profile not in PROFILES:
        raise ValueError(f"Unknown topology: {profile}")
    if not Path(model_path).is_absolute():
        raise ValueError("Model path must be an absolute local PVC path")
    p = PROFILES[profile]
    local_limit = 24 if profile == "pp2" else 48 // p["dp"]
    graphs = {
        "decode": {
            "backend": "full",
            "max_bs": local_limit,
            "bs": list(range(1, local_limit + 1)),
        },
        "prefill": {"backend": "disabled"},
    }
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--served-model-name",
        "GLM-5.3",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp-size",
        str(p["tp"]),
        "--pp-size",
        str(p["pp"]),
        "--dp-size",
        str(p["dp"]),
        "--ep-size",
        str(p["ep"]),
        "--attn-cp-size",
        "1",
        "--dcp-size",
        "1",
        "--moe-dp-size",
        "1",
        "--moe-a2a-backend",
        "none",
        "--mem-fraction-static",
        "0.80",
        "--max-running-requests",
        "48",
        "--chunked-prefill-size",
        "16384",
        "--context-length",
        "98304",
        "--page-size",
        "64",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--quantization",
        "w4afp8",
        "--disable-shared-experts-fusion",
        "--trust-remote-code",
        "--attention-backend",
        "dsa",
        "--dsa-prefill-backend",
        "flashmla_kv",
        "--dsa-decode-backend",
        "flashmla_kv",
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--cuda-graph-config",
        json.dumps(graphs, separators=(",", ":")),
        "--random-seed",
        "0",
        "--enable-metrics",
        "--enable-cache-report",
        "--enable-metrics-for-all-schedulers",
    ]
    if profile == "pp2":
        command += ["--pp-max-micro-batch-size", "24", "--pp-async-batch-depth", "0"]
        env["SGLANG_PP_LAYER_PARTITION"] = "39,39"
    else:
        command += ["--enable-dp-attention", "--load-balance-method", "round_robin"]
    if hicache:
        command += [
            "--enable-hierarchical-cache",
            "--hicache-size",
            "32",
            "--hicache-host-memory-mode",
            "cache",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "direct",
            "--hicache-mem-layout",
            "layer_first",
        ]
    # A stdlib-only import hook observes PP state in the multiprocessing children.
    bundle = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = bundle + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["GLM53_PP_OBSERVER"] = "1" if profile == "pp2" else "0"
    env["PYTHONUNBUFFERED"] = "1"
    return command, env


def validate_model(model_path):
    config = json.loads((Path(model_path) / "config.json").read_text())
    if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        raise ValueError("Expected full GlmMoeDsaForCausalLM checkpoint")
    if config.get("num_hidden_layers") != 78:
        raise ValueError("Expected 78 layers; PP39/39 is only valid for the full model")
    if config.get("quantization_config", {}).get("quant_method") != "w4afp8":
        raise ValueError("Expected checkpoint quantization_config.quant_method=w4afp8")


def validate_revision(bundle, environ=None, required=True):
    env = os.environ if environ is None else environ
    path = Path(bundle) / "image-revision"
    if not path.is_file():
        if required:
            raise ValueError("Image has no baked source revision")
        return None
    actual = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", actual):
        raise ValueError("Baked image source revision must be a full commit SHA")
    expected = env.get("SOURCE_COMMIT")
    if required and not expected:
        raise ValueError(
            "SOURCE_COMMIT must declare the expected image source revision"
        )
    if expected is not None and expected != actual:
        raise ValueError(
            f"Image source revision mismatch: baked={actual}, expected={expected}"
        )
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hicache", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    command, env = make_launch(args.profile, args.model_path, args.hicache, args.port)
    actual_revision = validate_revision(
        Path(__file__).resolve().parent, required=not args.print_command
    )
    record = {
        "base_commit": BASE_COMMIT,
        "profile": args.profile,
        "hicache": args.hicache,
        "image_source_commit": actual_revision,
        "expected_source_commit": os.environ.get("SOURCE_COMMIT"),
        "command": command,
        "pp_layer_partition": env.get("SGLANG_PP_LAYER_PARTITION"),
        "observer": env["GLM53_PP_OBSERVER"] == "1",
        "expected_local_running_limit": (
            24 if args.profile == "pp2" else 48 // PROFILES[args.profile]["dp"]
        ),
        "expected_local_prefill_chunk": 16384 // PROFILES[args.profile]["dp"],
        "hicache_anchor_gb_per_rank": 32 if args.hicache else 0,
        "hicache_dsa_indexer_memory_additional": args.hicache,
    }
    print("GLM53_LAUNCH " + json.dumps(record, sort_keys=True), flush=True)
    if args.print_command:
        print(shlex.join(command))
        return
    from install import package_root, verify

    verify(package_root(), Path(__file__).resolve().parent)
    validate_model(args.model_path)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
