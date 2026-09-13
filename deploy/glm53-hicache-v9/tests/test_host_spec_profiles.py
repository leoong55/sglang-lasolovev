"""Real launch validation, including the local MTP checkpoint preflight."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_host_controls import drop_options, replace_value
from test_host_launch import args_for, launch


class SpecProfilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)
        self.config = dict(num_hidden_layers=78, kv_lora_rank=512, qk_rope_head_dim=64,
                           num_nextn_predict_layers=1, architectures=["GlmMoeDsaForCausalLM"])
        (self.model / "config.json").write_text(json.dumps(self.config))
        (self.model / "mtp.safetensors").touch()
        self.weights = {f"model.layers.78.{part}.weight": "mtp.safetensors" for part in
                        ("eh_proj", "enorm", "hnorm", "shared_head.norm", "self_attn.q_a_proj", "mlp.experts.0.gate_proj")}
        (self.model / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=self.weights)))

    def argv(self, mode="DFLASH"):
        argv = args_for(16384, [8192, 16384]) + [
            "--model-path", str(self.model), "--tp-size", "8", "--ep-size", "8", "--dp-size", "1",
            "--dcp-size", "4", "--dcp-comm-backend", "ag_rs", "--quantization", "w4afp8",
            "--kv-cache-dtype", "fp8_e4m3", "--page-size", "64", "--dsa-decode-backend", "flashmla_kv",
            "--moe-runner-backend", "humming", "--mem-fraction-static", "0.80",
        ]
        argv = replace_value(argv, "--max-running-requests", 80)
        argv = replace_value(argv, "--cuda-graph-max-bs-decode", 80)
        argv += ["--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32", "48", "64", "80"]
        if mode == "DFLASH":
            return argv + ["--enable-prefill-cp", "--cp-strategy", "interleave", "--enable-cp-decode-attn-tp",
                           "--glm53-draft-cache-window", "2048", "--speculative-dflash-block-size", "8"]
        argv = drop_options(argv, ["--speculative-"])
        if mode == "off":
            return argv + ["--enable-prefill-cp", "--cp-strategy", "interleave", "--enable-cp-decode-attn-tp"]
        argv = replace_value(argv, "--dcp-size", 1)
        return argv + ["--glm53-profile", "tp8", "--speculative-algorithm", "EAGLE",
                       "--speculative-draft-model-quantization", "w4afp8", "--speculative-draft-attention-backend", "dsa",
                       "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]

    def check(self, argv):
        with patch.dict(os.environ, {"SGLANG_ENABLE_CP_V2": "1"}), contextlib.redirect_stdout(io.StringIO()):
            return launch.check_profile(launch.configure(argv))

    def test_three_modes_preserve_capacity_and_reset_feature_env(self):
        for mode in ("DFLASH", "EAGLE", "off"):
            argv = self.argv(mode)
            profile = self.check(argv)
            self.assertEqual(profile.max_running_requests, 80)
            self.assertEqual(profile.cuda_graph_max_bs_decode, 80)
            runtime = launch.runtime_argv(argv)
            self.assertFalse(any(x.startswith("--glm53-") for x in runtime))
            self.assertIn("0.80", runtime)
            with patch.dict(os.environ, {"SGLANG_GLM53_DFLASH_DCP": "1", "SGLANG_GLM53_DRAFT_CACHE_WINDOW": "2048"}):
                launch.configure_runtime_env(profile)
                self.assertEqual(os.environ["SGLANG_GLM53_DFLASH_DCP"], "1" if mode == "DFLASH" else "0")
                self.assertEqual(os.environ["SGLANG_GLM53_DRAFT_CACHE_WINDOW"], "2048" if mode == "DFLASH" else "0")
                self.assertEqual(os.environ["SGLANG_ENABLE_CP_V2"], "0" if mode == "EAGLE" else "1")
                if mode == "EAGLE":
                    self.assertEqual(os.environ["SGLANG_GLM53_HICACHE_DCP"], "0")

    def test_eagle_dcp_and_wrong_draft_do_not_load(self):
        bad = [replace_value(self.argv("EAGLE"), "--dcp-size", 4),
               replace_value(self.argv("DFLASH"), "--speculative-algorithm", "EAGLE"),
               replace_value(self.argv("EAGLE"), "--speculative-draft-attention-backend", "fa4"),
               replace_value(self.argv("EAGLE"), "--speculative-eagle-topk", 2),
               replace_value(self.argv("EAGLE"), "--speculative-num-draft-tokens", 8),
               replace_value(self.argv("DFLASH"), "--speculative-dflash-block-size", 4),
               self.argv("DFLASH") + ["--speculative-num-steps", "3"]]
        for argv in bad:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.check(argv)

    def test_missing_mtp_weights_or_shards_fail_before_gpu_load(self):
        launch.validate_mtp_checkpoint(self.model)
        del self.weights["model.layers.78.eh_proj.weight"]
        (self.model / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=self.weights)))
        with self.assertRaisesRegex(ValueError, "missing native MTP weights"):
            self.check(self.argv("EAGLE"))
        self.weights["model.layers.78.eh_proj.weight"] = "absent.safetensors"
        (self.model / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=self.weights)))
        with self.assertRaisesRegex(ValueError, "Missing MTP shard"):
            self.check(self.argv("EAGLE"))

    def test_required_graphs_cover_capacity_but_warn_keeps_eager_option(self):
        argv = self.argv("DFLASH") + ["--glm53-dflash-graph-policy", "require"]
        self.check(argv)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.check(replace_value(argv, "--max-running-requests", 96))
        self.check(replace_value(self.argv("DFLASH"), "--max-running-requests", 96))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.check(self.argv("off") + ["--glm53-dflash-profile-steps", "32"])


if __name__ == "__main__":
    unittest.main()
