"""Render the isolated C40 FP8 candidate with an immutable image reference."""

import argparse
from pathlib import Path

import yaml


def render(image):
    docs = list(
        yaml.safe_load_all(
            (Path(__file__).resolve().parent / "deployment.template.yaml").read_text()
        )
    )
    docs[0]["spec"]["template"]["spec"]["containers"][0]["image"] = image
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        parser.error("Use the published image digest")
    print(yaml.safe_dump_all(render(args.image), sort_keys=False), end="")
