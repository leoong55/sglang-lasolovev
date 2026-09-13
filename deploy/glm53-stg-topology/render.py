"""Render isolated Kubernetes resources. No cluster access and no credentials."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMESPACE = "inf-glm53"
PART = "glm53-stg-pp2-dpa"
BENCH_IMAGE = "vllm/vllm-openai@sha256:3a1e7f5904e1a1192a02aa0086ceaffc33985d7044c7bb25b3a43d61bdbe3ac0"
SERVICE = "glm53-topology-serving"


def meta(name, role):
    return {
        "name": name,
        "namespace": NAMESPACE,
        "labels": {"app.kubernetes.io/part-of": PART, "glm53-role": role},
    }


def affinity(node):
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchFields": [
                            {"key": "metadata.name", "operator": "In", "values": [node]}
                        ]
                    }
                ]
            }
        }
    }


def resources(cpu, memory, gpu=False):
    r = {"cpu": str(cpu), "memory": memory}
    if gpu:
        r.update(
            {
                "nvidia.com/gpu": "8",
                "nvidia.com/gpumem-percentage": "100",
                "nvidia.com/gpucores": "100",
            }
        )
    return {"requests": r.copy(), "limits": r.copy()}


def configmap():
    files = [
        "benchmark.py",
        "dataset_catalog.py",
        "export.py",
        "supervise.py",
        "preflight.py",
    ]
    data = {name: (ROOT / name).read_text() for name in files}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    metadata = meta("glm53-topology-scripts-" + digest[:20], "scripts")
    metadata["annotations"] = {"glm53-script-sha256": digest}
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata,
        "immutable": True,
        "data": data,
    }


def storage():
    return [
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": meta("glm53-topology-" + suffix, "storage"),
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": "longhorn",
                "resources": {"requests": {"storage": size}},
            },
        }
        for suffix, size in [("compile-cache", "100Gi"), ("results", "100Gi")]
    ]


def serving(profile, image, commit, node, hicache=False):
    if profile not in ("pp2", "dpa2", "dpa4", "dpa8"):
        raise ValueError("unsupported profile")
    if not re.fullmatch(
        r"ghcr\.io/leoong55/sglang-lasolovev@sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("serving image must be pinned to the experiment GHCR digest")
    selector = {"app": SERVICE, "app.kubernetes.io/part-of": PART}
    env = {
        "SOURCE_COMMIT": commit,
        "SERVING_IMAGE": image,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, sub in [
        ("SGLANG_CACHE_DIR", "sglang"),
        ("SGLANG_DG_CACHE_DIR", "deepgemm"),
        ("DG_JIT_CACHE_DIR", "deepgemm"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("HF_HOME", "hf"),
        ("XDG_CACHE_HOME", "xdg"),
    ]:
        env[name] = "/cache/" + commit[:12] + "/" + profile + "/" + sub
    container = {
        "name": "sglang",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "/scripts/supervise.py"],
        "args": [
            "--profile",
            profile,
            "--model-path",
            "/model",
            *(["--hicache"] if hicache else []),
        ],
        "env": [{"name": k, "value": v} for k, v in env.items()]
        + [
            {
                "name": "POD_UID",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
            }
        ],
        "resources": resources(64, "640Gi", True),
        "ports": [{"name": "http", "containerPort": 8080}],
        "startupProbe": {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 720,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 5,
            "timeoutSeconds": 3,
        },
        "volumeMounts": [
            {"name": "model", "mountPath": "/model", "readOnly": True},
            {"name": "cache", "mountPath": "/cache"},
            {"name": "scripts", "mountPath": "/scripts", "readOnly": True},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }
    spec = {
        "schedulerName": "hami-scheduler",
        "affinity": affinity(node),
        "automountServiceAccountToken": False,
        "terminationGracePeriodSeconds": 120,
        "containers": [container],
        "volumes": [
            {
                "name": "model",
                "persistentVolumeClaim": {
                    "claimName": "sglang-w4fp8-pvc",
                    "readOnly": True,
                },
            },
            {
                "name": "cache",
                "persistentVolumeClaim": {"claimName": "glm53-topology-compile-cache"},
            },
            {"name": "scripts", "configMap": {"name": configmap()["metadata"]["name"]}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
        ],
    }
    deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": meta(SERVICE, "serving"),
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": selector},
            "template": {
                "metadata": {
                    "labels": {
                        **selector,
                        "glm53-profile": profile,
                        "glm53-hicache": str(hicache).lower(),
                    },
                    "annotations": {"glm53-source-commit": commit},
                },
                "spec": spec,
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": meta(SERVICE, "serving"),
        "spec": {
            "type": "ClusterIP",
            "selector": selector,
            "ports": [{"name": "http", "port": 8080, "targetPort": "http"}],
        },
    }
    return [deploy, service]


def job(run_id, node, argv, role="benchmark"):
    if not re.fullmatch("[a-z0-9][a-z0-9-]{0,44}", run_id):
        raise ValueError("run-id must be a short DNS label")
    name = "glm53-" + run_id
    container = {
        "name": role,
        "image": BENCH_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python3", *argv],
        "env": [
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
            {"name": "HF_HUB_OFFLINE", "value": "1"},
            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
            {"name": "HF_HOME", "value": "/tmp/hf"},
            {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
        "resources": resources(4, "16Gi"),
        "volumeMounts": [
            {"name": "model", "mountPath": "/model", "readOnly": True},
            {"name": "results", "mountPath": "/results"},
            {"name": "scripts", "mountPath": "/scripts", "readOnly": True},
        ],
    }
    container["env"] += [
        {"name": "BENCHMARK_IMAGE", "value": BENCH_IMAGE},
        {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": meta(name, role),
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 43200 if role == "benchmark" else 1800,
            "template": {
                "metadata": {"labels": meta(name, role)["labels"]},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "affinity": affinity(node),
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "model",
                            "persistentVolumeClaim": {
                                "claimName": "sglang-w4fp8-pvc",
                                "readOnly": True,
                            },
                        },
                        {
                            "name": "results",
                            "persistentVolumeClaim": {
                                "claimName": "glm53-topology-results"
                            },
                        },
                        {
                            "name": "scripts",
                            "configMap": {"name": configmap()["metadata"]["name"]},
                        },
                    ],
                },
            },
        },
    }


def benchmark(run_id, profile, image, commit, node, repetitions=1, mode="suite"):
    return job(
        run_id,
        node,
        [
            "/scripts/benchmark.py",
            "--mode",
            mode,
            "--profile",
            profile,
            "--base-url",
            "http://" + SERVICE + "." + NAMESPACE + ".svc:8080",
            "--tokenizer",
            "/model",
            "--results-dir",
            "/results/" + run_id,
            "--repetitions",
            str(repetitions),
            "--source-commit",
            commit,
            "--image-digest",
            image,
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["base", "serving", "benchmark"], required=True)
    p.add_argument("--profile", choices=["pp2", "dpa2", "dpa4", "dpa8"], default="pp2")
    p.add_argument("--image")
    p.add_argument("--commit")
    p.add_argument("--node", help="Verified GPU node name, supplied locally")
    p.add_argument("--run-id")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--hicache", action="store_true")
    a = p.parse_args()
    if a.kind != "base" and not a.node:
        p.error("--node is required for workload placement")
    if a.kind == "base":
        items = [configmap(), *storage()]
    elif a.kind == "serving":
        items = serving(a.profile, a.image, a.commit, a.node, a.hicache)
    else:
        items = [
            benchmark(a.run_id, a.profile, a.image, a.commit, a.node, a.repetitions)
        ]
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2))


if __name__ == "__main__":
    main()
