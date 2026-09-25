"""Explicit experiment profile layered on the v3 launch arguments."""

import argparse
import json
import os
import sys
from pathlib import Path

from baseline_profile import validate
from install import install, package_root

PROFILE = "prefill-graph"
SETTINGS = {
    "--moe-a2a-backend": ["none"],
    "--deepep-mode": ["auto"],
    "--cuda-graph-backend-prefill": ["breakable"],
}
SETTINGS.update(
    {"--cuda-graph-bs-prefill": ["8192"], "--cuda-graph-max-bs-prefill": ["8192"]}
)


def configure(argv):
    # Remove complete occurrences, including list-valued options, before adding
    # the visible profile. Never retain duplicate argparse values from the YAML.
    out = []
    i = 0
    while i < len(argv):
        option = argv[i].split("=", 1)[0]
        if option in SETTINGS:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
        else:
            out.append(argv[i])
            i += 1
    for key, values in SETTINGS.items():
        out.extend([key, *values])
    return out


def check_profile(argv):
    validate(argv)
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--chunked-prefill-size", required=True, type=int)
    p.add_argument("--max-running-requests", required=True, type=int)
    p.add_argument("--cuda-graph-backend-decode", required=True)
    p.add_argument("--cuda-graph-max-bs-decode", required=True, type=int)
    p.add_argument(
        "--disable-shared-experts-fusion", action="store_true", required=True
    )
    p.add_argument("--dsa-prefill-backend", required=True)
    args, _ = p.parse_known_args(argv)
    if (
        args.chunked_prefill_size != 8192
        or args.max_running_requests != 32
        or args.cuda_graph_backend_decode != "full"
        or args.cuda_graph_max_bs_decode != 32
        or args.dsa_prefill_backend != "flashmla_sparse_q8"
    ):
        p.error(
            "Experiment requires v3: chunk8192, max-running32, full decode graphs/max32, Q8 prefill"
        )
    forbidden = {
        "--enable-two-batch-overlap",
        "--enable-single-batch-overlap",
        "--enable-eplb",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    }
    if any(x.split("=", 1)[0] in forbidden for x in argv):
        p.error("Remove overlap/EPLB/legacy graph override flags for this experiment")
    if os.environ.get("SGLANG_ENABLE_CP_V2") != "1":
        p.error("SGLANG_ENABLE_CP_V2=1 is required")


if __name__ == "__main__":
    argv = configure(sys.argv[1:])
    install(package_root(), Path(__file__).resolve().parent, verify_only=True)
    check_profile(argv)
    os.environ.pop("SGLANG_GLM53_DEEPEP_PREFILL", None)
    os.environ["SGLANG_GLM53_PREFILL_BCG"] = "1"
    print("glm53-" + PROFILE + "-v5: effective argv=" + json.dumps(argv), flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *argv])
