"""Generate a guarded decode-graph experiment and rollback from a live Deployment.

Only writes local JSON patches. Does not run kubectl or modify the cluster.
Usage: python3 make_decode_graph_patch.py deployment.json --output-dir graph-a1
"""

import argparse
import json
from pathlib import Path


def scalar(args, option):
    values = []
    for i, token in enumerate(args):
        if token == option:
            if i + 1 == len(args) or args[i + 1].startswith("--"):
                raise ValueError(f"Missing value for {option}")
            values.append(args[i + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {option}, got {values}")
    return values[0]


def graph_args(args):
    for option, value in {
        "--tp-size": "8", "--ep-size": "8", "--dp-size": "1",
        "--dcp-size": "4", "--cp-strategy": "interleave",
        "--dcp-comm-backend": "ag_rs",
        "--cuda-graph-backend-decode": "disabled",
        "--cuda-graph-backend-prefill": "disabled",
    }.items():
        if scalar(args, option) != value:
            raise ValueError(f"This experiment expects {option} {value}")
    if not 1 <= int(scalar(args, "--max-running-requests")) <= 32:
        raise ValueError("A1 expects max-running-requests <= 32; keep admission unchanged")
    forbidden = {
        "--cuda-graph-config", "--disable-cuda-graph",
        "--disable-decode-cuda-graph", "--disable-cuda-graph-padding",
        "--cuda-graph-max-bs", "--cuda-graph-bs",
    }
    if any(token.split("=", 1)[0] in forbidden for token in args):
        raise ValueError("Conflicting graph flags: use the v1 per-phase configuration")
    remove = {
        "--cuda-graph-backend-decode", "--cuda-graph-max-bs-decode",
        "--cuda-graph-bs-decode",
    }
    result = []
    i = 0
    while i < len(args):
        token = args[i]
        option = token.split("=", 1)[0]
        i += 1
        if option not in remove:
            result.append(token)
        elif "=" not in token:
            if i == len(args) or args[i].startswith("--"):
                raise ValueError(f"Missing value for {option}")
            i += 1
            if option == "--cuda-graph-bs-decode":
                while i < len(args) and not args[i].startswith("--"):
                    i += 1
    return result + [
        "--cuda-graph-backend-decode", "full",
        "--cuda-graph-max-bs-decode", "32",
        "--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32",
    ]


def make_patches(deployment):
    if (deployment.get("kind"), deployment["metadata"].get("namespace"),
            deployment["metadata"].get("name")) != (
            "Deployment", "inf-glm53", "sglang-glm53-dcp4"):
        raise ValueError("Expected Deployment inf-glm53/sglang-glm53-dcp4")
    containers = deployment["spec"]["template"]["spec"]["containers"]
    indices = [i for i, c in enumerate(containers) if c["name"] == "sglang"]
    if len(indices) != 1:
        raise ValueError("Expected exactly one container named sglang")
    i = indices[0]
    container = containers[i]
    old = container["args"]
    new = graph_args(old)
    root = f"/spec/template/spec/containers/{i}"
    guard = [
        {"op": "test", "path": "/metadata/uid", "value": deployment["metadata"]["uid"]},
        {"op": "test", "path": root + "/name", "value": "sglang"},
        {"op": "test", "path": root + "/image", "value": container["image"]},
    ]

    def patch(before, after):
        return guard + [
            {"op": "test", "path": root + "/args", "value": before},
            {"op": "replace", "path": root + "/args", "value": after},
        ]

    return patch(old, new), patch(new, old)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deployment", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("graph-a1"))
    ns = p.parse_args()
    try:
        forward, rollback = make_patches(json.loads(ns.deployment.read_text()))
        ns.output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in (("enable.json", forward), ("rollback.json", rollback)):
            (ns.output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    print(f"Created {ns.output_dir}/enable.json and rollback.json; cluster unchanged")


if __name__ == "__main__":
    main()
