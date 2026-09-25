"""Connect actual attention dispatch, LSE selection and numeric reduction."""

import math
import unittest
from types import SimpleNamespace

import torch
from host_source import extract, lse
from test_host_q8_prefill import Mode, make_backend


class LSEDispatchTest(unittest.TestCase):
    def setUp(self):
        self.backend = make_backend()
        self.source = extract(
            "srt/models/deepseek_common/attention_forward_methods/forward_mla.py",
            {"is_dsa_dcp_lse_base_on_e"},
            namespace={"get_attn_backend": lambda: self.backend},
        )

    def test_kernel_dispatch_and_lse_base_agree(self):
        for mode in Mode:
            for cp in (False, True):
                batch = SimpleNamespace(forward_mode=mode, attn_cp_metadata=object() if cp else None)
                expected = mode != Mode.EXTEND or not cp
                self.assertEqual(self.source.is_dsa_dcp_lse_base_on_e(batch), expected, (mode, cp))

    def test_short_prefill_matches_dense_attention(self):
        batch = SimpleNamespace(forward_mode=Mode.EXTEND, attn_cp_metadata=None)
        # Two nonempty DCP owners and two empty owners. Natural partition
        # functions are 1 and 3, so the exact weighted answer is 5, not 4.73.
        out = torch.tensor([2., 6., float("nan"), float("nan")]).view(4, 1, 1, 1)
        weights = torch.tensor([0., math.log(3), -float("inf"), -float("inf")]).view(4, 1, 1)
        actual = lse._lse_weighted_combine_cpu(out, weights, self.source.is_dsa_dcp_lse_base_on_e(batch))
        torch.testing.assert_close(actual, torch.tensor([[[5.]]]))


if __name__ == "__main__":
    unittest.main()
