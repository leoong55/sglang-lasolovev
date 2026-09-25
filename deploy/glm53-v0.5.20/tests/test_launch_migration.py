"""Launcher regression cases to run in the later validation phase (no GPU)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import launch


class LaunchMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / "config.json").write_text(json.dumps(dict(num_hidden_layers=78, kv_lora_rank=512, qk_rope_head_dim=64)))
        self.argv = json.loads((KIT / "profile.json").read_text())["argv"]
        self.set_value("--model-path", self.tmp.name)

    def set_value(self, flag, value):
        self.argv[self.argv.index(flag) + 1] = value

    def test_eager_4096_is_available_without_cp_v2_flag(self):
        self.set_value("--chunked-prefill-size", "4096")
        self.set_value("--cuda-graph-backend-prefill", "disabled")
        with patch.dict(launch.os.environ, {}, clear=True):
            profile = launch.check_profile(self.argv)
            launch.configure_runtime_env(profile)
            self.assertEqual(profile.chunked_prefill_size, 4096)
            self.assertEqual(launch.os.environ["SGLANG_GLM53_PREFILL_BCG"], "0")
            self.assertEqual(launch.os.environ["SGLANG_GLM53_DFLASH_DCP"], "1")
            self.assertNotIn("SGLANG_ENABLE_CP_V2", launch.os.environ)

    def test_4096_does_not_silently_enable_unsupported_graph_bucket(self):
        for flag in ("--chunked-prefill-size", "--cuda-graph-max-bs-prefill", "--cuda-graph-bs-prefill"):
            self.set_value(flag, "4096")
        with self.assertRaises(SystemExit):
            launch.check_profile(self.argv)

    def test_supplied_profile_keeps_full_draft_and_8k_prefill(self):
        profile = launch.check_profile(self.argv)
        self.assertEqual(profile.glm53_draft_cache_window, 0)
        self.assertEqual(profile.chunked_prefill_size, 8192)
        self.assertEqual(profile.speculative_dflash_block_size, 8)
        self.assertEqual(profile.moe_runner_backend, "humming")
        self.assertEqual(profile.cuda_graph_bs_decode, [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40])


if __name__ == "__main__":
    unittest.main()
