"""Render isolated H200 lab profiles; print YAML without deploying anything."""

import argparse
import math
from pathlib import Path

import yaml

PROFILES = ("tp8-decode", "pp2-decode", "pp4-decode", "tp8-archive", "pp4-archive")


def render(
    *,
    image,
    profile,
    concurrency=80,
    weights="w4afp8",
    model_pvc=None,
    context=500000,
    mem_fraction=0.90,
    hicache_size=187,
    patches=False,
    admit=False,
    short_bypass=0,
    park=False,
    skip=False,
    name="sglang-glm53-h200-ab"
):
    if profile not in PROFILES or concurrency < 1 or context < 4096:
        raise ValueError("Invalid profile/concurrency/context")
    if not 0 < mem_fraction < 1 or short_bypass < 0 or hicache_size <= 0:
        raise ValueError("Invalid memory fraction or scheduler budget")
    if patches:
        admit = park = skip = True
        short_bypass = 512
    if weights == "fp8" and not model_pvc:
        raise ValueError("FP8 requires the actual --model-pvc name")
    obj = yaml.safe_load(
        (Path(__file__).parent / "deployment.template.yaml").read_text()
    )
    obj["metadata"]["name"] = name
    obj["spec"]["selector"]["matchLabels"]["app"] = name
    pod = obj["spec"]["template"]
    pod["metadata"]["labels"]["app"] = name
    pod["metadata"]["annotations"]["glm53-test-profile"] = profile
    spec = pod["spec"]
    c = spec["containers"][0]
    c["image"] = image
    c["command"] = ["python3", "/opt/glm53-h200-ab/h200/launch.py"]
    pp = 1 if profile.startswith("tp8-") else (2 if profile.startswith("pp2-") else 4)
    tp = 8 // pp
    micro = math.ceil(concurrency / pp)
    buckets = sorted(
        {
            n
            for n in (1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 64, 80, 96, 120, 128, 160)
            if n <= micro
        }
        | {micro}
    )
    model_path = "/mnt/model-pvc-fp8" if weights == "fp8" else "/mnt/model-pvc-w4fp8"
    args = [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--model-path",
        model_path,
        "--served-model-name",
        "alpha-fm",
        "--trust-remote-code",
        "--tp-size",
        str(tp),
        "--pp-size",
        str(pp),
        "--ep-size",
        str(tp),
        "--dp-size",
        "1",
        "--moe-a2a-backend",
        "none",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--dsa-prefill-backend",
        "tilelang",
        "--dsa-decode-backend",
        "tilelang",
        "--disable-shared-experts-fusion",
        "--page-size",
        "64",
        "--mem-fraction-static",
        str(mem_fraction),
        "--context-length",
        str(context),
        "--max-running-requests",
        str(concurrency),
        "--chunked-prefill-size",
        "4096",
        "--prefill-decode-interval",
        "0",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        str(micro),
        "--cuda-graph-bs-decode",
        *map(str, buckets),
        "--stream-interval",
        "1",
        "--random-seed",
        "12345",
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--enable-metrics",
        "--enable-cache-report",
        "--enforce-disable-flashinfer-allreduce-fusion",
    ]
    if weights == "w4afp8":
        args += ["--quantization", "w4afp8", "--moe-runner-backend", "humming"]
    # Official FP8 checkpoint carries its own quantization config; native backend selection.
    if pp > 1:
        args += ["--pp-max-micro-batch-size", str(micro)]
    if profile.endswith("-archive"):
        args += [
            "--enable-hierarchical-cache",
            "--hicache-size",
            str(hicache_size),
            "--hicache-write-policy",
            "write_back",
            "--hicache-io-backend",
            "direct",
            "--hicache-mem-layout",
            "layer_first",
            "--enable-mixed-chunk",
        ]
    c["args"] = args
    env = {x["name"]: x["value"] for x in c["env"]}
    env.update(
        SGLANG_GLM53_HUMMING_EP_AWARE="1",
        SGLANG_ENABLE_H200_ADMIT_FULL_NEED="1" if admit else "0",
        SGLANG_H200_SHORT_BYPASS_TOKENS=str(short_bypass),
        SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL="1" if park else "0",
        SGLANG_ENABLE_H200_SKIP_NOT_FITTING="1" if skip else "0",
    )
    if pp > 1:
        env["SGLANG_PP_LAYER_PARTITION"] = "39,39" if pp == 2 else "21,20,20,17"
    c["env"] = [{"name": k, "value": v} for k, v in env.items()]
    for mount in c["volumeMounts"]:
        if mount["name"] == "model":
            mount["mountPath"] = model_path
    for vol in spec["volumes"]:
        if vol["name"] == "model":
            vol["persistentVolumeClaim"]["claimName"] = model_pvc or "sglang-w4fp8-pvc"
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": "inf-glm53"},
        "spec": {
            "selector": {"app": name},
            "ports": [{"name": "http", "port": 8080, "targetPort": "http"}],
        },
    }
    return [obj, service]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--concurrency", type=int, default=80)
    p.add_argument("--context", type=int, default=500000)
    p.add_argument("--weights", choices=("w4afp8", "fp8"), default="w4afp8")
    p.add_argument("--model-pvc")
    p.add_argument("--mem-fraction", type=float, default=0.90)
    p.add_argument("--hicache-size", type=float, default=187)
    p.add_argument(
        "--patches", action="store_true", help="Enable all four scheduler patches"
    )
    p.add_argument("--admit", action="store_true")
    p.add_argument("--short-bypass", type=int, default=0)
    p.add_argument("--park", action="store_true")
    p.add_argument("--skip", action="store_true")
    p.add_argument("--name", default="sglang-glm53-h200-ab")
    a = p.parse_args()
    try:
        docs = render(**vars(a))
    except ValueError as e:
        p.error(str(e))
    print(yaml.safe_dump_all(docs, sort_keys=False), end="")


if __name__ == "__main__":
    main()
