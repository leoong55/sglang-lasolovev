"""CPU-only exact CP geometry and real replay selection; no CUDA claims."""
import ast
import importlib.util
import os
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
spec = importlib.util.spec_from_file_location("glm53_bcg_geometry", ROOT / "python/sglang/srt/layers/cp/glm53_bcg.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class GeometryTest(unittest.TestCase):
    def test_mixed_request_lengths_keep_global_interleave_order(self):
        rng = random.Random(42)
        for total in helper.CAPTURE_TOKEN_SIZES:
            for _ in range(20):
                cuts = [0] + sorted(rng.sample(range(1, total), 31)) + [total]
                lengths = [b-a for a,b in zip(cuts, cuts[1:])]
                self.assertEqual(helper.exact_local_rows(lengths), total // 8)
                shards = [list(range(rank, total, 8)) for rank in range(8)]
                restored = [shards[i % 8][i // 8] for i in range(total)]
                self.assertEqual(restored, list(range(total)))
                self.assertEqual(helper.exact_replay_bucket(total, lengths, [total]), total)

    def test_tails_never_reuse_larger_graph(self):
        for total in helper.CAPTURE_TOKEN_SIZES:
            for delta in (-1, 1, -64, 64):
                self.assertIsNone(helper.exact_replay_bucket(total+delta, [total+delta], helper.CAPTURE_TOKEN_SIZES))
            self.assertIsNone(helper.exact_replay_bucket(total, [total-1], [total]))
            self.assertIsNone(helper.exact_replay_bucket(total, [total], []))
        for lengths in (None, [], [0,8192], [-1,8193]):
            self.assertIsNone(helper.exact_local_rows(lengths))

    def test_capture_sizes_validate_without_silently_dropping_buckets(self):
        for sizes in ([8192], [16384], [8192,32768], [8192,16384,32768]):
            self.assertEqual(helper.validate_capture_sizes(sizes), sizes)
        for sizes in ([], [8192,8192], [16384,8192], [4096], [8192.0]):
            with self.assertRaises(ValueError):
                helper.validate_capture_sizes(sizes)

    def test_actual_cp_replay_selector_requires_matching_capture(self):
        source = ROOT / "python/sglang/srt/layers/cp/bcg.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PrefillCPBCGInput")
        method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="select_replay_bucket_for_batch")
        strategy = type("InterleaveCPStrategy", (), {"cp_size":8})()
        namespace={"get_cp_strategy":lambda:strategy, "Any":object, "Optional":__import__('typing').Optional}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        modules={"sglang.srt.layers.cp.glm53_bcg":helper,
                 "sglang.srt.layers.cp.interleave":types.SimpleNamespace(InterleaveCPStrategy=type(strategy))}
        owner=types.SimpleNamespace(bucket_local_tokens={8192:1024,16384:2048,32768:4096})
        with patch.dict(sys.modules,modules), patch.dict(os.environ,{"SGLANG_GLM53_PREFILL_BCG":"1"}):
            for total in helper.CAPTURE_TOKEN_SIZES:
                def call(n=total):
                    return namespace[method.name](owner,num_tokens=n,extend_seq_lens=[n],capture_num_tokens=[8192,16384,32768],max_padding_factor=4)
                self.assertEqual(call(),total)
                self.assertIsNone(call(total-1))
                owner.bucket_local_tokens[total]+=1
                self.assertIsNone(call())
                owner.bucket_local_tokens[total]-=1


if __name__ == "__main__":
    unittest.main()
