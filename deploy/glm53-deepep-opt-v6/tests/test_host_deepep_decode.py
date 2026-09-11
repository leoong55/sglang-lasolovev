"""Ownership and graph-padding checks, without CUDA or model weights."""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"])
spec = importlib.util.spec_from_file_location(
    "tested_decode_partition", ROOT / "python/sglang/srt/layers/cp/glm53_deepep.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class OwnershipTests(unittest.TestCase):
    def test_model_selects_partition_only_for_decode_and_idle(self):
        path = ROOT / "python/sglang/srt/models/deepseek_v2.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if getattr(n, "name", "") == "DeepseekV2MoE")
        method = next(n for n in cls.body if getattr(n, "name", "") == "forward_deepep")
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias("annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ns = {}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        seen = []
        model = NS(
            _cp_deepep_decode_partition=True,
            is_hash=False,
            _forward_deepep_impl=lambda h, fb, ids=None: seen.append("original"),
        )
        stub = NS(forward_partitioned_decode=lambda *a: seen.append("partitioned"))
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.glm53_deepep": stub}):
            for decode, idle, expected in [
                (True, False, "partitioned"),
                (False, True, "partitioned"),
                (False, False, "original"),
            ]:
                batch = NS(
                    forward_mode=NS(is_decode=lambda: decode, is_idle=lambda: idle)
                )
                ns["forward_deepep"](model, torch.ones(32, 4), batch)
                self.assertEqual(seen[-1], expected)
            model._cp_deepep_decode_partition = False
            batch = NS(forward_mode=NS(is_decode=lambda: True, is_idle=lambda: False))
            ns["forward_deepep"](model, torch.ones(32, 4), batch)
            self.assertEqual(seen[-1], "original")

    def test_all_batches_and_graph_padding_preserve_unique_tokens_and_order(self):
        for padded in (1, 2, 4, 8, 16, 32):
            ids = torch.arange(padded * 4).reshape(padded, 4).float()
            for raw in range(padded + 1):
                count = torch.tensor([raw], dtype=torch.int32)
                parts, valid_ids = [], []
                for rank in range(8):
                    local, n = helper.partition_decode_tokens(ids, count, rank, 8)
                    self.assertTrue(local.is_contiguous())
                    valid_ids.extend(local[: n.item(), 0].tolist())
                    parts.append(local * 3 + 7)
                self.assertEqual(valid_ids, ids[:raw, 0].tolist())
                self.assertEqual(len(valid_ids), len(set(valid_ids)))
                for rank in range(8):

                    def gather(out, local):
                        out.copy_(torch.cat(parts))

                    result = helper.gather_decode_tokens(parts[rank], padded, 8, gather)
                    torch.testing.assert_close(result[:raw], (ids * 3 + 7)[:raw])
                self.assertEqual(count.item(), raw)

    def test_live_counter_changes_without_changing_graph_shape(self):
        counter = torch.tensor([32], dtype=torch.int32)
        hidden = torch.ones(32, 4)
        for raw in (32, 31, 17, 1, 0, 32):
            counter.fill_(raw)
            counts = [
                helper.partition_decode_tokens(hidden, counter, r, 8)[1].item()
                for r in range(8)
            ]
            self.assertEqual(sum(counts), raw)
            self.assertTrue(all(0 <= n <= 4 for n in counts))

    def test_scope_restores_shared_batch_counter_on_error(self):
        original = torch.tensor([31], dtype=torch.int32)
        batch = NS(num_token_non_padded=original)

        def forward(hidden, fb):
            self.assertEqual(hidden.shape[0], 4)
            self.assertEqual(fb.num_token_non_padded.item(), 3)
            raise RuntimeError("expert failure")

        modules = {
            "sglang.srt.runtime_context": NS(
                get_parallel=lambda: NS(attn_cp_rank=7, attn_cp_size=8)
            ),
            "sglang.srt.distributed": NS(get_tp_group=lambda: NS()),
        }
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "expert failure"):
                helper.forward_partitioned_decode(
                    NS(_forward_deepep_impl=forward), torch.ones(32, 4), batch
                )
        self.assertIs(batch.num_token_non_padded, original)

    def test_unpadded_eager_and_empty_inputs(self):
        for batch in (0, 1, 7, 15, 32):
            x = torch.zeros(batch, 4)
            counts = [
                helper.partition_decode_tokens(x, None, rank, 8)[1].item()
                for rank in range(8)
            ]
            self.assertEqual(sum(counts), batch)


if __name__ == "__main__":
    unittest.main()
