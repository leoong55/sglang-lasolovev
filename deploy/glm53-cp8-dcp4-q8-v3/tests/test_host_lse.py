import math
import unittest

import torch

from host_source import lse


class TestHostLSE(unittest.TestCase):
    def test_empty_shards_ignore_nonfinite_output(self):
        out = torch.full((4, 2, 3, 8), float("nan"))
        out[0].fill_(2)
        out[2].fill_(6)
        out[3].fill_(float("inf"))
        weights = torch.full((4, 2, 3), float("-inf"))
        weights[0].zero_()
        weights[2].zero_()
        for base_e in (True, False):
            actual = lse._lse_weighted_combine_cpu(out, weights, base_e)
            torch.testing.assert_close(actual, torch.full_like(actual, 4))

    def test_all_shards_empty(self):
        out = torch.full((4, 2, 3, 8), float("nan"))
        weights = torch.full((4, 2, 3), float("-inf"))
        for base_e in (True, False):
            actual = lse._lse_weighted_combine_cpu(out, weights, base_e)
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_natural_log_weighting(self):
        out = torch.tensor([2., 6.]).view(2, 1, 1, 1)
        weights = torch.tensor([0., math.log(3)]).view(2, 1, 1)
        actual = lse._lse_weighted_combine_cpu(out, weights, True)
        torch.testing.assert_close(actual, torch.tensor([[[5.]]]))


if __name__ == "__main__":
    unittest.main()
