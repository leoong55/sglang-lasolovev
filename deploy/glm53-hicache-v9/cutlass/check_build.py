"""Verify the requested image flavor and extension ABI without running CUDA."""
import argparse
import json
from pathlib import Path

from build import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", choices=("0", "1"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    info = json.loads((root / "BUILD.json").read_text())
    enabled = args.expected == "1"
    if info["enabled"] is not enabled:
        raise RuntimeError("Image CUTLASS build flavor does not match the requested tag")
    library = root / "glm53_cutlass.so"
    if enabled:
        if info["variants"] != 7 or file_sha256(library) != info["library_sha256"]:
            raise RuntimeError("Incomplete or changed CUTLASS library")
        import torch
        torch.ops.load_library(str(library))
        assert hasattr(torch.ops.glm53_cutlass, "w4a8_mm")
    elif library.exists():
        raise RuntimeError("The Humming-only image contains an unexpected tuning library")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
