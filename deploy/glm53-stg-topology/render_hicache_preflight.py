"""Render an immutable, CPU-only HiCache RAM gate; never contact the cluster."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import hicache_preflight as gate
import render

ROOT = Path(__file__).resolve().parent


def verify_snapshot(tooling_commit):
    gate.require(
        re.fullmatch(r"[0-9a-f]{40}", tooling_commit), "tooling_commit_must_be_full_sha"
    )
    for name in (
        "hicache_preflight.py",
        "render_hicache_preflight.py",
        "render.py",
        "analyze_results.py",
    ):
        result = subprocess.run(
            ["git", "show", f"{tooling_commit}:deploy/glm53-stg-topology/{name}"],
            cwd=ROOT.parent.parent,
            capture_output=True,
            check=False,
        )
        gate.require(
            result.returncode == 0 and result.stdout == (ROOT / name).read_bytes(),
            "tooling_source_snapshot_mismatch_" + name,
        )
    for name, expected in gate.SOURCE_SHA256.items():
        path = "python/sglang/srt/mem_cache/" + name
        result = subprocess.run(
            ["git", "show", f"{gate.UPSTREAM_COMMIT}:{path}"],
            cwd=ROOT.parent.parent,
            capture_output=True,
            check=False,
        )
        gate.require(
            result.returncode == 0
            and gate.digest(result.stdout) == expected
            and gate.digest((ROOT.parent.parent / path).read_bytes()) == expected,
            "reviewed_allocator_source_mismatch_" + name,
        )


def serving_plan(raw, node, tooling_commit, baseline):
    document = json.loads(raw)
    objects = (
        document if isinstance(document, list) else document.get("items", [document])
    )
    deployments = [obj for obj in objects if obj.get("kind") == "Deployment"]
    gate.require(len(deployments) == 1, "exactly_one_serving_deployment_required")
    deployment = deployments[0]
    gate.require(
        deployment["metadata"].get("name") == render.SERVICE
        and deployment["metadata"].get("namespace") == render.NAMESPACE,
        "unexpected_serving_target",
    )
    pod = deployment["spec"]["template"]
    profile = pod["metadata"]["labels"].get("glm53-profile")
    gate.require(
        profile in gate.PROFILES
        and pod["metadata"]["labels"].get("glm53-hicache") == "true",
        "planned_hicache_profile_missing",
    )
    gate.require(
        pod["spec"].get("affinity") == render.affinity(node),
        "planned_node_affinity_mismatch",
    )
    gate.require(
        not pod["spec"].get("nodeName") or pod["spec"]["nodeName"] == node,
        "planned_node_name_mismatch",
    )
    containers = pod["spec"]["containers"]
    gate.require(
        len(containers) == 1 and containers[0]["name"] == "sglang",
        "unexpected_serving_container",
    )
    container = containers[0]
    gate.require(
        container.get("command") == ["python3", "/scripts/supervise.py"],
        "planned_serving_entrypoint_changed",
    )
    gate.require(
        container.get("args")
        == ["--profile", profile, "--model-path", "/model", "--hicache"],
        "planned_launcher_flags_changed",
    )
    gate.require(
        container.get("resources") == render.resources(64, "640Gi", True),
        "planned_serving_resources_changed",
    )
    env = {item["name"]: item.get("value") for item in container["env"]}
    source, image = env.get("SOURCE_COMMIT", ""), container["image"]
    gate.require(re.fullmatch(r"[0-9a-f]{40}", source), "planned_source_commit_missing")
    gate.require(
        re.fullmatch(r"ghcr\.io/leoong55/sglang-lasolovev@sha256:[0-9a-f]{64}", image)
        and env.get("SERVING_IMAGE") == image,
        "planned_serving_image_unpinned",
    )
    gate.require(
        re.fullmatch(r"[0-9a-f]{40}", tooling_commit), "tooling_commit_must_be_full_sha"
    )
    return {
        "profile": profile,
        "node_sha256": gate.digest(node.encode()),
        "source_commit": source,
        "serving_image_digest": image,
        "model_config_sha256": baseline["model_config_sha256"],
        "tooling_commit": tooling_commit,
        "upstream_commit": gate.UPSTREAM_COMMIT,
        "upstream_source_sha256": gate.SOURCE_SHA256,
        "serving_manifest_sha256": gate.digest(raw),
        "serving_limit_bytes": gate.SERVING_LIMIT,
        "operational_margin_bytes": gate.OPERATIONAL_MARGIN,
    }


def verify_image_source_contract(plan):
    """Require the serving revision's installer to verify every allocator input."""
    result = subprocess.run(
        [
            "git",
            "show",
            f"{plan['source_commit']}:deploy/glm53-stg-topology/manifest.json",
        ],
        cwd=ROOT.parent.parent,
        capture_output=True,
        check=False,
    )
    gate.require(result.returncode == 0, "serving_revision_manifest_missing")
    manifest = json.loads(result.stdout)
    files = {item["path"]: item["sha256"] for item in manifest.get("files", [])}
    gate.require(
        manifest.get("base_commit") == gate.UPSTREAM_COMMIT
        and all(
            files.get("srt/mem_cache/" + name) == value
            for name, value in gate.SOURCE_SHA256.items()
        ),
        "serving_installer_does_not_verify_allocator_inputs",
    )


def build(run_id, node, tooling_commit, baseline, manifest_raw):
    gate.require(
        len(run_id) <= 45
        and re.fullmatch(r"hicache-preflight-[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", run_id),
        "run_id_must_be_unique_hicache_preflight_dns_label",
    )
    gate.require(
        len(node) <= 253
        and all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part)
            for part in node.split(".")
        ),
        "invalid_private_node_name",
    )
    plan = serving_plan(manifest_raw, node, tooling_commit, baseline)
    gate.require(
        baseline.get("status") == "verified"
        and baseline.get("purpose") == "hicache_ram_baseline_evidence",
        "verified_baseline_summary_required",
    )
    gate.require(
        baseline.get("helper_sha256")
        == gate.digest((ROOT / "hicache_preflight.py").read_bytes()),
        "baseline_helper_snapshot_mismatch",
    )
    gate.require(
        all(
            baseline.get(k) == plan[k]
            for k in (
                "node_sha256",
                "source_commit",
                "serving_image_digest",
                "model_config_sha256",
            )
        ),
        "baseline_plan_identity_mismatch",
    )
    data = {
        "hicache_preflight.py": (ROOT / "hicache_preflight.py").read_text(),
        "baseline.json": gate.encoded(baseline).decode(),
        "plan.json": gate.encoded(plan).decode(),
    }
    config_hash = gate.digest(gate.encoded(data))
    metadata = render.meta(
        "glm53-hicache-preflight-" + config_hash[:20], "ram-preflight"
    )
    metadata["annotations"] = {
        "glm53-script-sha256": config_hash,
        "glm53-tooling-commit": tooling_commit,
    }
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata,
        "immutable": True,
        "data": data,
    }
    job = render.job(
        run_id, node, ["/scripts/hicache_preflight.py", "check"], role="ram-preflight"
    )
    job["spec"]["activeDeadlineSeconds"] = 600
    spec = job["spec"]["template"]["spec"]
    spec["volumes"] = [v for v in spec["volumes"] if v["name"] != "results"]
    next(v for v in spec["volumes"] if v["name"] == "scripts")["configMap"]["name"] = (
        metadata["name"]
    )
    container = spec["containers"][0]
    container["volumeMounts"] = [
        v for v in container["volumeMounts"] if v["name"] != "results"
    ]
    container["resources"] = render.resources(1, "256Mi")
    container["env"] += [
        {"name": "SCRIPT_CONFIG_SHA256", "value": config_hash},
        {
            "name": "NODE_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
    ]
    for meta in (job["metadata"], job["spec"]["template"]["metadata"]):
        meta["labels"]["glm53-purpose"] = "hicache-ram-preflight"
        meta["annotations"] = dict(metadata["annotations"])
    return {"apiVersion": "v1", "kind": "List", "items": [cm, job]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--node", required=True, help="Verified node name, supplied privately"
    )
    parser.add_argument("--tooling-commit", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--serving-manifest",
        type=Path,
        required=True,
        help="Exact reviewed JSON manifest that will later be applied",
    )
    args = parser.parse_args()
    try:
        verify_snapshot(args.tooling_commit)
        baseline, _ = gate.read_json(args.baseline)
        gate.require(
            args.serving_manifest.stat().st_size <= gate.MAX_JSON_BYTES,
            "serving_manifest_size_limit",
        )
        document = build(
            args.run_id,
            args.node,
            args.tooling_commit,
            baseline,
            args.serving_manifest.read_bytes(),
        )
        verify_image_source_contract(
            json.loads(document["items"][0]["data"]["plan.json"])
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
