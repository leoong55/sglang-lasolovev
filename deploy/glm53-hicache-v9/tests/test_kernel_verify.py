"""Actual Triton index transform versus an independent scalar causal reference.

TRITON_INTERPRET=1 runs on CPU; without it this test requires CUDA.
"""
import importlib.util
import os
import unittest
from pathlib import Path

import torch

ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
spec = importlib.util.spec_from_file_location("actual_transform", ROOT / "python/sglang/kernels/ops/attention/dsa/transform_index.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


class VerifyIndexTest(unittest.TestCase):
    def test_causal_owner_filter_across_pages_padding_and_graph_widths(self):
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        for width in (2, 8, 16):
            prefix = [0, 63, 256]
            rows = len(prefix) * width
            table_width = 320
            # Non-identity physical mapping crosses several logical/physical pages.
            table = torch.stack([(torch.arange(table_width) * 7 + i * 64 + 256) for i in range(len(prefix))])
            table = table.repeat_interleave(width, dim=0).to(device=device, dtype=torch.int32)
            limits = torch.tensor([p+j+1 for p in prefix for j in range(width)], dtype=torch.int32, device=device)
            choices = [-1, -4, 0, 1, 62, 63, 64, 65, 127, 128, 255, 256, 257, 271, 319, 320, 10000]
            topk = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
            topk[:, :len(choices)] = torch.tensor(choices, device=device)
            cpu_table = table.cpu().tolist()
            cpu_limits = limits.cpu().tolist()
            for rank in range(4):
                got = kernel.transform_index_page_table_prefill_fast(
                    table, topk, [1] * rows, page_table_is_expanded=True,
                    output_num_tokens=rows + 3, dcp_size=4, dcp_rank=rank,
                    causal_seq_lens=limits,
                ).cpu()
                expected = torch.full((rows + 3, 2048), -1, dtype=torch.int32)
                for row in range(rows):
                    for col, logical in enumerate(choices):
                        if 0 <= logical < min(cpu_limits[row], table_width):
                            virtual = cpu_table[row][logical]
                            if virtual % 4 == rank:
                                expected[row, col] = virtual // 4
                torch.testing.assert_close(
                    got, expected, msg=lambda msg: f"width={width}, rank={rank}, first rows={got[:, :18].tolist()}\n{msg}"
                )


if __name__ == "__main__":
    unittest.main()
