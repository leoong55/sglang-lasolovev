"""Execute actual AG+RS correction and A2A combine kernels against softmax."""

import importlib.util
import math
import os
import unittest

import torch
from host_source import ROOT

spec = importlib.util.spec_from_file_location(
    "actual_dcp_kernels", ROOT / "python/sglang/kernels/ops/attention/dcp_kernels.py"
)
kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernels)


class ActualLSETest(unittest.TestCase):
    def test_ag_rs_correction_matches_a2a_and_dense_reference(self):
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        # Unequal weights, an empty owner with NaN output, and an all-empty row.
        out = torch.tensor([2., 6., float("nan"), 10.], device=device).view(4, 1, 1, 1).expand(4, 2, 3, 8).clone()
        out[:, 1].fill_(float("nan"))
        natural = torch.tensor([0., math.log(3), -float("inf"), math.log(2)], device=device).view(4, 1, 1).expand(4, 2, 3).clone()
        natural[:, 1].fill_(-float("inf"))
        expected = torch.zeros((2, 3, 8), device=device)
        expected[0].fill_(40. / 6.)
        for base_e in (False, True):
            weights = natural if base_e else natural / math.log(2)
            corrected = []
            for rank in range(4):
                dest = torch.empty((3, 2, 8), device=device)
                total_lse = torch.empty((2, 3), device=device)
                kernels._correct_attn_cp_out_kernel[(2, 3, 1)](
                    out[rank], dest, weights, total_lse,
                    *out[rank].stride(), *weights.stride(), *dest.stride(), rank,
                    HEAD_DIM=8, N_ROUNDED=4, IS_LSE_BASE_ON_E=base_e,
                )
                corrected.append(dest.transpose(0, 1))
            ag_rs = torch.stack(corrected).sum(0)
            combined = torch.empty_like(expected)
            combined_lse = torch.empty((2, 3), device=device)
            kernels._dcp_lse_combine_kernel[(2, 3)](
                out, weights, combined, combined_lse,
                *out.stride(), *weights.stride(), *combined.stride(),
                N=4, HEAD_DIM=8, IS_BASE_E=base_e, RETURN_LSE=True,
            )
            torch.testing.assert_close(ag_rs, expected)
            torch.testing.assert_close(combined, expected)
            torch.testing.assert_close(total_lse, combined_lse)


if __name__ == "__main__":
    unittest.main()
