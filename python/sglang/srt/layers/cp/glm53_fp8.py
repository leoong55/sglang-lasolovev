"""Serialized FP8 target contract for the v9.14 CP8/DCP4 integration.

The CP, DCP and draft helpers exchange activations/KV, not packed expert
weights. Keep their layout checks and admit only the native block-FP8 MoE
path whose default with A2A none is Triton. This is not an H200 validation.
"""


def supports_fp8_target(cfg, resolved, model):
    quant = getattr(model, "quantization_config", None)
    return (
        cfg.quantization in (None, "fp8")
        and isinstance(quant, dict)
        and quant.get("quant_method") == "fp8"
        and quant.get("weight_block_size") == [128, 128]
        and quant.get("activation_scheme") == "dynamic"
        and resolved.moe_runner_backend in ("auto", "triton")
        and resolved.moe_a2a_backend == "none"
        and getattr(model, "kv_lora_rank", None) == 512
        and getattr(model, "qk_rope_head_dim", None) == 64
    )
