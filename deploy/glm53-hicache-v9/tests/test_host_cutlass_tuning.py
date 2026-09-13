"""Validate opt-in dispatch and schema; CUDA execution is a separate gate."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"])
spec = importlib.util.spec_from_file_location(
    "glm53_cutlass_test", ROOT / "python/sglang/srt/layers/moe/glm53_cutlass.py")
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)


def config():
    return dict(schema=1, geometry=[6144, 2048, 256, 32, 8], sm=90,
                pairs={"40": [1, 4], "48": [0, 2]})


class CutlassTuningTest(unittest.TestCase):
    def tearDown(self):
        tuning._benchmark_pair = None
        tuning.config.cache_clear()
        tuning.load_library.cache_clear()

    def test_shape_and_exact_batch_dispatch_no_prefill_extrapolation(self):
        with patch.object(tuning, "config", return_value=config()):
            self.assertEqual(tuning.select_pair(40, 6144, 2048, 32, 8, 8), (1, 4))
            for batch in (1, 32, 39, 41, 64, 8192, 16384):
                self.assertEqual(tuning.select_pair(batch, 6144, 2048, 32, 8, 8), (0, 0))
            for geometry in ((7168, 2048, 32, 8, 8), (6144, 2048, 256, 1, 8),
                             (6144, 2048, 32, 8, 4), (6144, 1024, 32, 8, 8)):
                self.assertEqual(tuning.select_pair(40, *geometry), (0, 0))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tuning.select_pair(40, 6144, 2048, 32, 8, 8), (0, 0))

    def test_reject_wrong_geometry_invalid_variants_and_prefill_tables(self):
        self.assertEqual(tuning.validate_config(config()), config())
        for field, value in (("schema", 2), ("sm", 100), ("geometry", [7168, 2048, 256, 32, 8]),
                             ("pairs", {"40": [8, 0]}), ("pairs", {"40": [True, 0]}),
                             ("pairs", {"40": [-1, 0]}), ("pairs", {"40": [1]}),
                             ("pairs", {"040": [1, 2]}), ("pairs", {"2048": [1, 2]})):
            data = copy.deepcopy(config()); data[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                tuning.validate_config(data)

    def test_kernel_arguments_keep_real_problem_sizes_and_scaling(self):
        # The final stock argument is only the dispatch topk. No tensor,
        # routing bound, scale or group size may be adjusted by the selector.
        args = tuple(object() for _ in range(11)) + (128, 8)
        calls = []
        fake = NS(cutlass_w4a8_moe_mm=lambda *a: calls.append(("stock", a)))
        with patch.dict(sys.modules, {"sgl_kernel": fake}), \
             patch.object(tuning, "load_library", return_value=lambda *a:calls.append(("tuned", a))):
            tuning.gemm(0, *args)
            tuning.gemm(4, *args)
        self.assertEqual(calls[0], ("stock", args))
        self.assertEqual(calls[1], ("tuned", args[:-1] + (4,)))

    def test_load_once_rejects_stale_device_or_torch_and_missing_file(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.json"
            data = config(); data.update(torch=str(torch.__version__), gpu="NVIDIA H200")
            path.write_text(json.dumps(data))
            with patch.dict(os.environ, {"SGLANG_GLM53_CUTLASS_CONFIG": str(path)}), \
                 patch.object(torch.cuda, "get_device_name", return_value="NVIDIA H200"), \
                 patch.object(tuning, "load_library") as load:
                self.assertEqual(tuning.config()["pairs"], data["pairs"])
                path.unlink()
                self.assertEqual(tuning.config()["pairs"], data["pairs"])
                self.assertEqual(load.call_count, 1)
                tuning.config.cache_clear()
                with self.assertRaises(FileNotFoundError):
                    tuning.config()
                for changes in (dict(gpu="H100"), dict(torch="different")):
                    path.write_text(json.dumps(dict(data, **changes)))
                    with self.assertRaises(ValueError):
                        tuning.config()


if __name__ == "__main__":
    unittest.main()
