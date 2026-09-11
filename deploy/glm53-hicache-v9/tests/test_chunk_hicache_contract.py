import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"])
spec = importlib.util.spec_from_file_location("hicache_contract", ROOT / "python/sglang/srt/layers/cp/glm53_hicache.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class HiCacheContractTest(unittest.TestCase):
    def test_supported_profile_and_incompatible_modes(self):
        cfg = NS(speculative_algorithm="DFLASH", enable_hierarchical_cache=True,
                 dcp_size=4, tp_size=8, hicache_storage_backend=None,
                 hicache_host_memory_mode="cache", hicache_write_policy="write_through",
                 hicache_mem_layout="layer_first", hicache_io_backend="direct",
                 enable_lmcache=False, enable_hisparse=False)
        env = {key:"1" for key in ("SGLANG_GLM53_HICACHE_DCP", "SGLANG_GLM53_DFLASH_DCP", "SGLANG_ENABLE_UNIFIED_RADIX_TREE")}
        with patch.dict(os.environ, env):
            self.assertTrue(helper.supports_hicache_dflash(cfg))
            for name, value in dict(hicache_storage_backend="mooncake", hicache_write_policy="write_back",
                                    hicache_mem_layout="page_first", hicache_io_backend="kernel",
                                    hicache_host_memory_mode="buffer_only",enable_lmcache=True,
                                    enable_hisparse=True, speculative_algorithm="EAGLE",dcp_size=2).items():
                changed = NS(**(vars(cfg) | {name:value}))
                self.assertFalse(helper.supports_hicache_dflash(changed), name)
            with patch.dict(os.environ, {"SGLANG_ENABLE_UNIFIED_RADIX_TREE":"0"}):
                self.assertFalse(helper.supports_hicache_dflash(cfg))


if __name__ == "__main__":
    unittest.main()
