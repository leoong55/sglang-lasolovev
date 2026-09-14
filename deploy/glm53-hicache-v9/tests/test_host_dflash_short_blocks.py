"""Native DFlash2 block convolution boundaries, using CPU tensor execution."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F


ROOT = Path(os.environ['SGLANG_SOURCE_ROOT'])


class ShortBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / 'python/sglang/srt/models/dflash.py'
        tree = ast.parse(source.read_text())
        conv = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_grouped_conv')
        # Execute the native tensor function, not torch.compile's CPU backend.
        conv.decorator_list = []
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == 'DFlashDraftModel')
        setter = next(n for n in model.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'set_block_size')
        ns = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[conv, setter], type_ignores=[]), str(source), 'exec'), ns)
        cls.conv = staticmethod(ns['_grouped_conv'])
        cls.setter = staticmethod(ns['set_block_size'])

    def test_native_grouped_conv_does_not_mix_requests_for_short_blocks(self):
        generator = torch.Generator().manual_seed(42)
        for width in (2, 4, 8):
            with self.subTest(width=width):
                requests, groups, group_size, taps = 3, 2, 4, 2
                hidden = torch.randn(requests * width, groups * group_size, generator=generator)
                delta = torch.randn(requests * width, taps, groups, generator=generator)
                base = torch.randn(taps, groups, group_size, generator=generator)
                actual = self.conv(hidden, delta, base, width, groups, group_size, taps)
                expected = torch.zeros_like(hidden)
                for row in range(requests * width):
                    for tap in range(min(taps, row % width + 1)):
                        coeff = base[tap] + delta[row, tap, :, None]
                        expected[row] += coeff.flatten() * hidden[row - tap]
                torch.testing.assert_close(actual, expected)

    def test_native_setter_updates_all_attention_and_mlp_convolutions(self):
        model = NS(block_size=8, layers=[NS(attention_conv=NS(block_size=8),
                                           mlp_conv=NS(block_size=8)) for _ in range(6)])
        for width in (4, 2, 8):
            self.setter(model, width)
            self.assertEqual(model.block_size, width)
            for layer in model.layers:
                self.assertEqual(layer.attention_conv.block_size, width)
                self.assertEqual(layer.mlp_conv.block_size, width)


if __name__ == '__main__':
    unittest.main()
