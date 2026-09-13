"""Render a synthetic CPU client validation Job; never access the cluster."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import render

ROOT = Path(__file__).resolve().parent
SCRIPTS = ("benchmark.py", "dataset_catalog.py", "export.py", "client_harness.py")


def validate_inputs(run_id, node, tooling_commit):
    if len(run_id) > 45 or not re.fullmatch(
        r"client-validation-[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", run_id
    ):
        raise ValueError(
            "run-id must be a unique client-validation-* DNS label, at most45 characters"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", tooling_commit):
        raise ValueError("tooling-commit must be a full lowercase40-character Git SHA")
    if len(node) > 253 or not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part)
        for part in node.split(".")
    ):
        raise ValueError(
            "node must be the privately supplied verified Kubernetes node name"
        )


def verify_snapshot(tooling_commit):
    """The declared revision must contain exactly the sources being rendered."""
    for name in (*SCRIPTS, "render_client_validation.py", "render.py"):
        result = subprocess.run(
            ["git", "show", f"{tooling_commit}:deploy/glm53-stg-topology/{name}"],
            cwd=ROOT.parent.parent,
            capture_output=True,
            check=False,
        )
        if result.returncode or result.stdout != (ROOT / name).read_bytes():
            raise ValueError(
                f"{name} does not match tooling-commit; render from its committed checkout"
            )


def build(run_id, node, tooling_commit):
    """Pure rendering; the CLI separately verifies the committed source snapshot."""
    validate_inputs(run_id, node, tooling_commit)
    data = {name: (ROOT / name).read_text() for name in SCRIPTS}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    metadata = render.meta("glm53-client-validation-" + digest[:20], "scripts")
    metadata["labels"].update(
        {"glm53-purpose": "validation", "glm53-synthetic": "true"}
    )
    metadata["annotations"] = {"glm53-script-sha256": digest}
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata,
        "immutable": True,
        "data": data,
    }
    job = render.job(
        run_id,
        node,
        [
            "/scripts/client_harness.py",
            "--benchmark-file",
            "/scripts/benchmark.py",
            "--tokenizer",
            "/model",
            "--results-dir",
            "/results/" + run_id,
            "--timeout",
            "3600",
        ],
        role="validation",
    )
    job["spec"]["activeDeadlineSeconds"] = 7200
    for metadata in (job["metadata"], job["spec"]["template"]["metadata"]):
        metadata["labels"].update(
            {"glm53-purpose": "validation", "glm53-synthetic": "true"}
        )
        metadata["annotations"] = {
            "glm53-script-sha256": digest,
            "glm53-tooling-commit": tooling_commit,
            "glm53-validation-scope": "synthetic-client-protocol-only-no-model-performance",
        }
    pod = job["spec"]["template"]["spec"]
    for volume in pod["volumes"]:
        if volume["name"] == "scripts":
            volume["configMap"]["name"] = configmap["metadata"]["name"]
    pod["containers"][0]["env"] += [
        {"name": "SCRIPT_CONFIG_SHA256", "value": digest},
        {"name": "TOOLING_COMMIT", "value": tooling_commit},
        {"name": "BENCHMARK_PURPOSE", "value": "validation"},
    ]
    return {"apiVersion": "v1", "kind": "List", "items": [configmap, job]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--tooling-commit", required=True)
    args = parser.parse_args()
    try:
        validate_inputs(args.run_id, args.node, args.tooling_commit)
        verify_snapshot(args.tooling_commit)
        document = build(args.run_id, args.node, args.tooling_commit)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
