"""FP8-only entry point; reuse the v9.14 CP/DCP feature validation."""

import json
import os
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
LEGACY = (
    KIT.parent
    if (KIT.parent / "base-files.json").is_file()
    else KIT.parent / "glm53-hicache-v9"
)
sys.path.insert(0, str(LEGACY))

from install import install, package_root
from launch import check_profile, configure, configure_runtime_env, runtime_argv


def prepare(argv):
    argv = configure(argv)
    profile = check_profile(argv, target_quantization="fp8")
    if profile.glm53_profile != "cp8-dcp4":
        raise ValueError("This FP8 image profile requires CP8/DCP4")
    if profile.glm53_cp_decode_fusion != "off":
        raise ValueError("This FP8 candidate keeps the custom CP decode fusion off")
    configure_runtime_env(profile)
    return profile, runtime_argv(argv)


def main():
    install(package_root(), LEGACY, verify_only=True)
    profile, argv = prepare(sys.argv[1:])
    print(
        "glm53-fp8-cp8-dcp4-v1: serialized block-FP8 target; "
        f"moe-backend-requested={profile.moe_runner_backend}; "
        "native FP8 auto + CUDA + A2A none selects Triton; "
        f"shared-experts-disable-requested={profile.disable_shared_experts_fusion}; "
        "shared-expert fusion is decided by the native model loader; "
        f"speculation={profile.speculative_algorithm or 'off'}; "
        f"hicache={profile.enable_hierarchical_cache}; "
        f"max-running={profile.max_running_requests}; "
        f"prefill-graphs={profile.cuda_graph_backend_prefill}; "
        f"decode-graphs={profile.cuda_graph_backend_decode}; "
        f"bounded-draft-window={profile.glm53_draft_cache_window}",
        flush=True,
    )
    print("glm53-fp8: effective argv=" + json.dumps(argv), flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *argv])


if __name__ == "__main__":
    main()
