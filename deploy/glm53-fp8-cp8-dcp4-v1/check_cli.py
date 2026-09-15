"""Parse the actual manifest in the installed runtime, without loading weights."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from launch_fp8 import prepare
from render import render

from sglang.srt.server_args import ServerArgs

with tempfile.TemporaryDirectory() as directory:
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 78,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (Path(directory) / "config.json").write_text(json.dumps(config))
    container = render("validation")[0]["spec"]["template"]["spec"]["containers"][0]
    os.environ.update({entry["name"]: entry["value"] for entry in container["env"]})
    argv = container["args"]
    argv[argv.index("--model-path") + 1] = directory
    _, runtime = prepare(argv)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(runtime)
    assert parsed.quantization is None
    assert not parsed.disable_shared_experts_fusion
    assert parsed.moe_runner_backend == "auto"
    print(
        "Installed CLI accepts FP8 autodetection, native MoE default and shared-expert default"
    )
