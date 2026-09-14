"""Render the existing eight-GPU Deployment with an explicit image reference."""

import argparse
import re
from pathlib import Path


def render(image):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", image):
        raise ValueError("Expected one Docker image reference")
    if ":" not in image.rsplit("/", 1)[-1]:
        raise ValueError("Image reference must have an explicit tag or digest")
    text = Path(__file__).with_name("deployment.template.yaml").read_text()
    return text.replace("REPLACE_WITH_PP2_HUMMING_IMAGE", image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(render(args.image), end="")
