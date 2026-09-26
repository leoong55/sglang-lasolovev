"""Validate launcher intent against the initialized target runner."""
import json
import logging
import os


def validate(runner):
    expected = os.environ.get("SGLANG_GLM53_EXPECTED_RUNTIME")
    if not expected or runner.is_draft_worker:
        return
    expected = json.loads(expected)
    from sglang.srt.runtime_context import get_schedule
    cfg = get_schedule()
    prefill = runner.prefill_cuda_graph_runner
    actual = dict(chunk=cfg.chunked_prefill_size, slots=runner.max_running_requests,
                  delay=cfg.min_free_slots_delay,
                  prefill_backend=getattr(runner.attn_backend,"dsa_prefill_impl",None),
                  graph=type(getattr(prefill,"backend",prefill)).__name__,
                  buckets=getattr(prefill,"capture_num_tokens",None),
                  effective_kv_tokens=runner.max_total_num_tokens,
                  indexer=cfg.glm53_dsa_indexer_mode, dcp_prefill=cfg.glm53_dcp_prefill_mode,
                  attention_graph=cfg.glm53_prefill_attention_graph,
                  hicache_sync=cfg.glm53_hicache_event_sync)
    for key in ("chunk","slots","delay","prefill_backend"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"GLM profile mismatch {key}: expected {expected[key]}, got {actual[key]}")
    if expected["graph"] == "breakable":
        if actual["graph"] != "BreakableCudaGraphBackend" or actual["buckets"] != expected["buckets"]:
            raise RuntimeError(f"GLM prefill graph mismatch: {actual}")
    elif actual["graph"] != "EagerRunner":
        raise RuntimeError(f"Expected eager prefill, got {actual['graph']}")
    logging.getLogger(__name__).info("glm53_effective_runtime %s",json.dumps(actual,sort_keys=True))
