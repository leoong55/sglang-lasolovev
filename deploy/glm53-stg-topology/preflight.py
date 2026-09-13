"""Read-only inventory of the explicitly mounted GLM model and allocated GPUs."""

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path


def command(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    return {
        "command": argv,
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


root = Path("/model")
report = {
    "model_path": str(root),
    "top_level_files": sorted(p.name for p in root.iterdir())[:100],
    "cpu_count": os.cpu_count(),
    "meminfo": Path("/proc/meminfo").read_text(),
    "gpu": command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    ),
    "topology": command(["nvidia-smi", "topo", "-m"]),
}
errors = []
try:
    raw = (root / "config.json").read_bytes()
    cfg = json.loads(raw)
    report["config_sha256"] = hashlib.sha256(raw).hexdigest()
    report["config"] = cfg
    if cfg.get("num_hidden_layers") != 78 or "GlmMoeDsaForCausalLM" not in cfg.get(
        "architectures", []
    ):
        errors.append("Expected full 78-layer GlmMoeDsaForCausalLM checkpoint")
    if cfg.get("quantization_config", {}).get("quant_method") != "w4afp8":
        errors.append("Expected w4afp8 quantization_config")
    index_raw = (root / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_raw)
    report["index_sha256"] = hashlib.sha256(index_raw).hexdigest()
    names = sorted(set(index["weight_map"].values()))
    shards = []
    for name in names:
        path = root / name
        if not path.is_file():
            errors.append("Missing shard: " + name)
            continue
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 0 < size <= 64 * 1024 * 1024:
                raise ValueError("Invalid safetensors header size: " + name)
            header_bytes = f.read(size)
            header = json.loads(header_bytes)
        expected = [k for k, v in index["weight_map"].items() if v == name]
        missing = set(expected) - set(header)
        if missing:
            errors.append("Missing indexed tensors: " + name)
        data_end = max(
            (v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
            default=0,
        )
        if path.stat().st_size < 8 + size + data_end:
            errors.append("Truncated shard: " + name)
        shards.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                "tensor_count": len(expected),
            }
        )
    report["shards"] = shards
    report["weight_bytes"] = sum(s["bytes"] for s in shards)
    report["tokenizer_files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
        and (
            p.name.startswith("tokenizer")
            or p.name
            in (
                "generation_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
            )
        )
    }
    if not (root / "tokenizer.json").is_file():
        errors.append("Missing tokenizer.json")
except Exception as exc:
    errors.append(type(exc).__name__ + ": " + str(exc))
gpu_lines = report["gpu"]["stdout"].strip().splitlines()
if (
    report["gpu"]["returncode"]
    or len(gpu_lines) != 8
    or not all("H200" in line for line in gpu_lines)
):
    errors.append("Expected exactly eight allocated H200 GPUs")
report["errors"] = errors
report["status"] = "FAIL" if errors else "PASS"
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(bool(errors))
