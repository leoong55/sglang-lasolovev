"""Exercise the selected prefill shapes before HTTP readiness is published."""
import logging
import time

logger = logging.getLogger(__name__)


def warmup_requests(buckets):
    sizes = sorted(set(int(x) for x in buckets))
    if not sizes or sizes[0] <= 0:
        raise ValueError("Positive prefill warmup buckets required")
    # Distinct token IDs prevent accidental reuse between shapes. The extra
    # request exercises a continuation and DCP prefix path after a full chunk.
    return [
        {"input_ids": [i + 1] * n,
         "sampling_params": {"temperature": 0, "max_new_tokens": 1},
         "stream": False}
        for i, n in enumerate([*sizes, sizes[-1] + 256])
    ]


def run(url, buckets, *, headers, verify, timeout, post=None):
    if post is None:
        import requests
        post = requests.post
    started = time.perf_counter()
    for body in warmup_requests(buckets):
        result = post(url + "/generate", json=body, headers=headers, verify=verify, timeout=timeout)
        result.raise_for_status()
        info = result.json().get("meta_info", {})
        reason = info.get("finish_reason", {})
        if not isinstance(info, dict) or not isinstance(reason, dict) or reason.get("type") in (None, "abort"):
            raise RuntimeError("GLM53 prefill warmup did not finish generation")
        logger.info("glm53_prefill warmup shape=%d completed", len(body["input_ids"]))
    logger.info("glm53_prefill warmup_seconds=%.3f; all selected shapes ready", time.perf_counter() - started)
