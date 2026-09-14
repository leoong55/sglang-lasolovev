"""Record image source and dependencies; building is not GPU validation."""

import json
import os
import re
from importlib.metadata import version
from pathlib import Path

root = Path(__file__).resolve().parent
source = os.environ.get("SOURCE_COMMIT", "unknown")
if not re.fullmatch(r"[0-9a-f]{40}", source):
    raise ValueError("Build with --build-arg SOURCE_COMMIT=$(git rev-parse HEAD)")
manifest = json.loads((root / "manifest.json").read_text())
record = {
    "profile": manifest["profile"],
    "source_commit": source,
    "base_commit": manifest["base_commit"],
    "pp2_branch_commit": manifest["pp2_branch_commit"],
    "humming_source_commit": manifest["humming_source_commit"],
    "humming_kernels": version("humming-kernels"),
    "torch": version("torch"),
    "gpu_validated": False,
}
(root / "REVISION.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, sort_keys=True), flush=True)
