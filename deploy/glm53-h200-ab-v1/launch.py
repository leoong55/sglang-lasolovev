"""Launch the independent native TP/PP experiment with explicit arguments."""

import json
import os
import runpy
import sys


def main():
    argv = sys.argv[1:]
    # Experimental native profiles are deliberately explicit; no topology is rewritten.
    print("h200-ab native argv: " + json.dumps(argv), flush=True)
    print(
        "h200-ab scheduler switches: "
        + json.dumps(
            {
                k: os.environ.get(k, "0")
                for k in (
                    "SGLANG_ENABLE_H200_ADMIT_FULL_NEED",
                    "SGLANG_H200_SHORT_BYPASS_TOKENS",
                    "SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL",
                    "SGLANG_ENABLE_H200_SKIP_NOT_FITTING",
                )
            }
        ),
        flush=True,
    )
    sys.argv = ["sglang.launch_server", *argv]
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
