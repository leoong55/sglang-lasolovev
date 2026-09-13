"""Render a CPU-only, read-only results watcher for adaptive baseline screening."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import render


def objects(campaign, node):
    if not re.fullmatch(r"[a-z0-9-]{1,20}", campaign):
        raise ValueError("Invalid campaign")
    source = Path(__file__).with_name("watch_campaign.py").read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": render.meta("glm53-campaign-watch-" + digest[:16], "progress"),
        "data": {"watch_campaign.py": source},
    }
    job = render.job(
        campaign + "-progress",
        node,
        ["/scripts/watch_campaign.py", "--campaign", campaign],
        role="progress",
    )
    job["spec"]["activeDeadlineSeconds"] = 43260
    spec = job["spec"]["template"]["spec"]
    container = spec["containers"][0]
    container["resources"] = {
        "requests": {"cpu": "100m", "memory": "256Mi"},
        "limits": {"cpu": "200m", "memory": "256Mi"},
    }
    container["volumeMounts"] = [
        dict(v, readOnly=True)
        for v in container["volumeMounts"]
        if v["name"] != "model"
    ]
    spec["volumes"] = [v for v in spec["volumes"] if v["name"] != "model"]
    for volume in spec["volumes"]:
        if volume["name"] == "results":
            volume["persistentVolumeClaim"]["readOnly"] = True
        elif volume["name"] == "scripts":
            volume["configMap"]["name"] = cm["metadata"]["name"]
    return {"apiVersion": "v1", "kind": "List", "items": [cm, job]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    print(json.dumps(objects(args.campaign, args.node), indent=2))
