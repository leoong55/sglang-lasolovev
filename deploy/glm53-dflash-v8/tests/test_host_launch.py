import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import launch


def args_for(chunk, buckets):
    return [
        "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", "incoai/GLM-5.3-DFlash2",
        "--speculative-draft-model-quantization", "unquant", "--speculative-draft-attention-backend", "fa4",
        "--chunked-prefill-size", str(chunk), "--max-running-requests", "32",
        "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "32",
        "--disable-shared-experts-fusion", "--dsa-prefill-backend", "flashmla_sparse_q8",
        "--moe-a2a-backend", "none", "--cuda-graph-backend-prefill", "breakable",
        "--cuda-graph-max-bs-prefill", str(chunk), "--cuda-graph-bs-prefill", *map(str, buckets),
    ]


class LaunchTest(unittest.TestCase):
    def check(self, argv):
        with patch.object(launch, "validate"), patch.dict(os.environ, {"SGLANG_ENABLE_CP_V2": "1"}):
            return launch.check_profile(launch.configure(argv))

    def test_all_chunks_and_explicit_subsets(self):
        for chunk, buckets in ((8192, [8192]), (16384, [8192, 16384]),
                               (16384, [16384]), (32768, [8192, 16384, 32768]),
                               (32768, [16384, 32768]), (32768, [32768])):
            argv = args_for(chunk, buckets)
            self.assertEqual(launch.configure(argv), argv)
            result = self.check(argv)
            self.assertEqual(result.cuda_graph_bs_prefill, buckets)

    def test_conflicts_and_unsupported_buckets_rejected(self):
        for chunk, buckets in ((16384, [8192]), (8192, [16384]), (32768, [8192, 24576, 32768]),
                               (16384, [16384, 8192]), (8192, [8192, 8192]), (12288, [12288])):
            with self.subTest(chunk=chunk, buckets=buckets), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.check(args_for(chunk, buckets))
        argv = args_for(16384, [16384])
        argv[argv.index("none")] = "deepep"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.check(argv)

    def test_duplicate_separate_or_equals_flag_rejected(self):
        for suffix in (["--chunked-prefill-size=32768"], ["--chunked-prefill-size", "32768"]):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                launch.configure(args_for(16384, [16384]) + suffix)

    def test_unrelated_memory_and_decode_options_preserved(self):
        argv = args_for(16384, [16384]) + ["--mem-fraction-static", "0.80", "--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32"]
        self.assertEqual(launch.configure(argv), argv)


if __name__ == "__main__":
    unittest.main()
