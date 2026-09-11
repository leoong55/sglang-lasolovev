"""Launcher feature toggles preserve operator values and stale-env opt-outs."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from test_host_launch import args_for, launch


def drop_options(argv, prefixes):
    result, skip = [], False
    for token in argv:
        if token.startswith("--"):
            skip = any(token.split("=", 1)[0].startswith(p) for p in prefixes)
        if not skip:
            result.append(token)
    return result


def replace_value(argv, flag, value):
    result = list(argv)
    result[result.index(flag) + 1] = str(value)
    return result


class ControlsTest(unittest.TestCase):
    def check(self, argv):
        with patch.object(launch, "validate"), patch.dict(os.environ, {"SGLANG_ENABLE_CP_V2": "1"}):
            return launch.check_profile(launch.configure(argv))

    def test_speculation_and_hicache_are_independent(self):
        for speculation in (False, True):
            for cache in (False, True):
                args = args_for(16384, [8192, 16384])
                if not speculation:
                    args = drop_options(args, ["--speculative-"])
                if not cache:
                    args = drop_options(args, ["--enable-hierarchical-cache"])
                result = self.check(args)
                self.assertEqual(result.speculative_algorithm, "DFLASH" if speculation else None)
                self.assertEqual(result.enable_hierarchical_cache, cache)
                self.assertEqual(launch.configure(args), args)
                with patch.dict(os.environ, {"SGLANG_GLM53_DFLASH_DCP": "1", "SGLANG_GLM53_HICACHE_DCP": "1"}):
                    launch.configure_runtime_env(result)
                    self.assertEqual(os.environ["SGLANG_GLM53_DFLASH_DCP"], str(int(speculation)))
                    self.assertEqual(os.environ["SGLANG_GLM53_HICACHE_DCP"], str(int(cache)))

    def test_running_limit_and_decode_capture_limit_are_independent(self):
        for maximum in (1, 32, 40, 48, 64, 96, 128):
            args = replace_value(args_for(16384, [16384]), "--max-running-requests", maximum)
            result = self.check(args)
            self.assertEqual(result.max_running_requests, maximum)
            self.assertEqual(result.cuda_graph_max_bs_decode, 32)
            args = replace_value(args, "--cuda-graph-max-bs-decode", maximum)
            buckets = sorted({n for n in (1, 2, 4, 8, 16, 32, 48, 64, 96, maximum) if n <= maximum})
            args += ["--cuda-graph-bs-decode", *map(str, buckets)]
            result = self.check(args)
            self.assertEqual(result.cuda_graph_bs_decode, buckets)

    def test_each_graph_phase_can_be_disabled(self):
        for phase in ("prefill", "decode"):
            args = replace_value(args_for(16384, [16384]), "--cuda-graph-backend-" + phase, "disabled")
            args = drop_options(args, ["--cuda-graph-bs-" + phase, "--cuda-graph-max-bs-" + phase])
            result = self.check(args)
            self.assertEqual(getattr(result, "cuda_graph_backend_" + phase), "disabled")
            with patch.dict(os.environ):
                launch.configure_runtime_env(result)
                self.assertEqual(os.environ["SGLANG_GLM53_PREFILL_BCG"], "0" if phase == "prefill" else "1")

    def test_incomplete_speculation_and_invalid_limits_fail_before_loading(self):
        bad = [
            drop_options(args_for(16384, [16384]), ["--speculative-algorithm"]),
            drop_options(args_for(16384, [16384]), ["--speculative-draft-model-path"]),
            replace_value(args_for(16384, [16384]), "--max-running-requests", 0),
            replace_value(args_for(16384, [16384]), "--cuda-graph-max-bs-decode", -1),
            args_for(16384, [16384]) + ["--cuda-graph-bs-decode", "1", "16"],
            args_for(16384, [16384]) + ["--cuda-graph-bs-decode", "1", "32", "32"],
            replace_value(args_for(16384, [16384]), "--hicache-write-policy", "write_back"),
        ]
        for args in bad:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.check(args)


if __name__ == "__main__":
    unittest.main()
