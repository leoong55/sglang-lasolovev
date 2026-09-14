"""Verify the image and pass every serving argument unchanged to SGLang."""

import json
import os
import sys
from pathlib import Path

from install import install, package_root

if __name__ == "__main__":
    bundle = Path(__file__).resolve().parent
    install(package_root(), bundle, verify_only=True)
    revision = json.loads((bundle / "REVISION.json").read_text())
    print("GLM53_PP2_IMAGE " + json.dumps(revision, sort_keys=True), flush=True)
    os.execv(
        sys.executable,
        [sys.executable, "-m", "sglang.launch_server", *sys.argv[1:]],
    )
