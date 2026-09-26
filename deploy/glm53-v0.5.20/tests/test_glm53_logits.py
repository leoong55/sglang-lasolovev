import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
s = importlib.util.spec_from_file_location("logits", ROOT / "python/sglang/srt/layers/attention/dsa/glm53_logits.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

class LogitsBudgetTest(unittest.TestCase):
    def test_aligned_allocation_and_balanced_tail(self):
        for q in (1, 3, 4, 4097, 16384):
            for k in (17, 65535, 131072):
                for compact in (False, True):
                    budget = 64 << 20
                    tiles = m.balanced_tiles(q, k, budget, compact=compact)
                    self.assertEqual(tiles[0][0], 0)
                    self.assertEqual(tiles[-1][1], q)
                    sizes = [m.aligned(b-a,4) for a,b in tiles]
                    self.assertLessEqual(max(sizes)-min(sizes), 4)
                    for (a,b), size in zip(tiles,sizes):
                        self.assertEqual(a%4,0)
                        self.assertLessEqual(size*m.logits_stride(k,compact=compact)*4,budget)
                    for x,y in zip(tiles,tiles[1:]):
                        self.assertEqual(x[1],y[0])
    def test_width_and_reserve(self):
        self.assertEqual(m.logits_stride(1024,compact=False),1280)
        self.assertEqual(m.logits_stride(1024,compact=True),1024)
        self.assertEqual(m.workspace_reservation_bytes(1024),1<<30)
        self.assertEqual(m.workspace_reservation_bytes(1024,is_draft=True),0)
        with self.assertRaises(ValueError):
            m.balanced_tiles(4,131072,1<<20,compact=True)
