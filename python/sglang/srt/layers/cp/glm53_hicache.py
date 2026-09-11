"""Explicit L1/L2 contract for the GLM53 DFlash2 experiment."""

import os


def supports_hicache_dflash(cfg):
    # Called during argument resolution, before resolved attention fields exist.
    # The DSA backend separately requires supports_dflash_dcp(model_runner.args).
    return (
        os.environ.get("SGLANG_GLM53_HICACHE_DCP") == "1"
        and os.environ.get("SGLANG_GLM53_DFLASH_DCP") == "1"
        and os.environ.get("SGLANG_ENABLE_UNIFIED_RADIX_TREE") == "1"
        and cfg.speculative_algorithm == "DFLASH"
        and cfg.enable_hierarchical_cache
        and cfg.dcp_size == 4
        and cfg.tp_size == 8
        and cfg.hicache_storage_backend is None
        and cfg.hicache_host_memory_mode == "cache"
        and cfg.hicache_write_policy == "write_through"
        and cfg.hicache_mem_layout == "layer_first"
        and cfg.hicache_io_backend == "direct"
        and not cfg.enable_lmcache
        and not cfg.enable_hisparse
    )
