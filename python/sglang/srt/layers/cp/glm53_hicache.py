"""Explicit GLM53 L1/L2 contract, with an optional DFlash2 sidecar."""

import os


def supports_hicache_layout(cfg):
    # Also used before resolved attention fields exist. Model/topology checks
    # belong to supports_hicache_cp_dcp at backend construction.
    return (
        os.environ.get("SGLANG_GLM53_HICACHE_DCP") == "1"
        and os.environ.get("SGLANG_ENABLE_UNIFIED_RADIX_TREE") == "1"
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


def supports_hicache_dflash(cfg):
    return (
        cfg.speculative_algorithm == "DFLASH"
        and os.environ.get("SGLANG_GLM53_DFLASH_DCP") == "1"
        and supports_hicache_layout(cfg)
    )


def supports_hicache_cp_dcp(server_args):
    """The target/indexer L2 path also works without a draft sidecar."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.layers.cp.glm53_dflash import supports_cp_dcp, supports_dflash_dcp

    cfg = resolving_view(server_args)
    return (
        supports_cp_dcp(server_args)
        and supports_hicache_layout(cfg)
        and (
            cfg.speculative_algorithm is None
            or supports_dflash_dcp(server_args)
        )
    )
