"""Pass native arguments through; retain the existing launcher for v9.14 profiles."""

import json
import os
import runpy
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    kit = Path(__file__).resolve().parent
    if any(x.split("=", 1)[0] == "--glm53-profile" for x in argv):
        runpy.run_path(str(kit.parent / "launch.py"), run_name="__main__")
        return
    # Experimental native profiles are deliberately explicit; no topology is rewritten.
    print("h200-ab native argv: " + json.dumps(argv), flush=True)
    print(
        "h200-ab scheduler switches: "
        + json.dumps(
            {
                k: os.environ.get(k, "0")
                for k in (
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
