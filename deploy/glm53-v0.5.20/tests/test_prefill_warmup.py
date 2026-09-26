import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prefill_warmup", ROOT / "python/sglang/srt/observability/glm53_prefill_warmup.py")
warmup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warmup)


class WarmupTest(unittest.TestCase):
    def test_bucket_and_continuation_are_exercised_without_prefix_hits(self):
        bodies = warmup.warmup_requests([16384, 8192])
        self.assertEqual([len(b["input_ids"]) for b in bodies], [8192, 16384, 16640])
        self.assertEqual(len({b["input_ids"][0] for b in bodies}), 3)

    def test_generation_failure_prevents_readiness(self):
        response = Mock()
        response.json.return_value = {"meta_info": {"finish_reason": {"type": "abort"}}}
        post = Mock(return_value=response)
        with self.assertRaises(RuntimeError):
            warmup.run("http://localhost", [8192], headers={}, verify=True, timeout=1, post=post)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
