"""CPU contract checks; DeepEP, CUDA and CUTLASS execution are mocked.

Exercise the actual new forward and the actual decoder-layer stage branch.
These do not substitute for hardware/numerical validation of GPU kernels.
"""

import importlib.util
import unittest
from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import torch
from host_source import ROOT, extract

path = ROOT / "python/sglang/srt/layers/moe/glm53_prefill_deepep.py"
spec = importlib.util.spec_from_file_location("prefill_deepep_tested", path)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
CombineInput = namedtuple("CombineInput", "hidden_states topk_ids topk_weights")


class TestLayout(unittest.TestCase):
    def test_all_ranks_interleave_reconstructs_global_tokens(self):
        for total in (0, 1, 7, 8, 9, 31, 8132, 8192):
            for size in (2, 4, 8):
                seen = []
                for rank in range(size):
                    n = runtime.local_token_count(
                        total, rank, size, (total + size - 1) // size
                    )
                    ids = list(range(rank, total, size))
                    self.assertEqual(n, len(ids))
                    seen.extend(ids)
                self.assertEqual(sorted(seen), list(range(total)))

    def test_invalid_layout_rejected(self):
        for args in (
            (-1, 0, 8, 0),
            (10, 8, 8, 2),
            (10, -1, 8, 2),
            (10, 0, 0, 2),
            (8192, 0, 8, 1023),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runtime.local_token_count(*args)


class TestSharedWeightLoading(unittest.TestCase):
    def model(self):
        return NS(
            named_parameters=lambda: iter(
                [
                    (
                        "model.layers.3.mlp.prefill_shared_experts.gate_up_proj.weight",
                        None,
                    ),
                    (
                        "model.layers.3.mlp.prefill_shared_experts.gate_up_proj.weight_scale_inv",
                        None,
                    ),
                    (
                        "model.layers.3.mlp.prefill_shared_experts.down_proj.weight",
                        None,
                    ),
                    (
                        "model.layers.3.mlp.prefill_shared_experts.down_proj.weight_scale_inv",
                        None,
                    ),
                ]
            )
        )

    def weights(self):
        return [
            (
                f"model.layers.3.mlp.shared_experts.{proj}.{kind}",
                torch.tensor([float(i)]),
            )
            for i, (proj, kind) in enumerate(
                (p, k)
                for p in ("gate_proj", "up_proj", "down_proj")
                for k in ("weight", "weight_scale_inv")
            )
        ]

    def test_gate_up_down_and_scales_are_loaded_to_both_without_alias(self):
        weights = self.weights()
        out = list(runtime.duplicate_shared_weights(iter(weights), self.model()))
        self.assertEqual(len(out), 12)
        for i, (name, tensor) in enumerate(weights):
            new_name, new_tensor = out[i * 2]
            old_name, old_tensor = out[i * 2 + 1]
            self.assertEqual(
                new_name,
                name.replace(".mlp.shared_experts.", ".mlp.prefill_shared_experts."),
            )
            self.assertEqual(old_name, name)
            self.assertIs(old_tensor, tensor)
            torch.testing.assert_close(new_tensor, tensor, rtol=0, atol=0)
            new_tensor.add_(1000)
            self.assertNotEqual(new_tensor.item(), old_tensor.item())

    def test_missing_weight_or_scale_fails(self):
        weights = self.weights()
        for missing in range(len(weights)):
            with (
                self.subTest(missing=missing),
                self.assertRaisesRegex(RuntimeError, "missing"),
            ):
                list(
                    runtime.duplicate_shared_weights(
                        weights[:missing] + weights[missing + 1 :], self.model()
                    )
                )

    def test_no_auxiliary_module_preserves_iterator(self):
        weights = [("model.layers.3.mlp.experts.1.gate_proj.weight", torch.ones(2))]
        out = list(
            runtime.duplicate_shared_weights(
                weights, NS(named_parameters=lambda: iter([]))
            )
        )
        self.assertEqual(out[0][0], weights[0][0])
        self.assertIs(out[0][1], weights[0][1])


class TestPrefillForward(unittest.TestCase):
    def run_forward(self, total, rank, width=4, physical=None):
        size = 8
        physical = physical if physical is not None else (total + size - 1) // size
        n = len(range(rank, total, size))
        x = torch.full((physical, width), float("nan"), dtype=torch.bfloat16)
        local = (
            torch.tensor(list(range(rank, total, size)), dtype=torch.long)[:, None]
            + torch.arange(width)[None, :]
        ).to(torch.bfloat16)
        x[:n].copy_(local)
        events = []

        def gate(h, **kwargs):
            events.append("gate")
            self.assertEqual(len(h), n)
            self.assertTrue(torch.isfinite(h).all())
            # Distinct token routes and weights, including multiple experts on a rank.
            return h[:, :1].float()

        def topk(h, logits):
            ids = (h[:, 0].long()[:, None] + torch.arange(2)[None, :]) % 16
            return NS(
                topk_ids=ids, topk_weights=torch.tensor([0.25, 0.75]).expand(n, -1)
            )

        def dispatch(h, top):
            events.append("dispatch")
            self.assertEqual(len(h), n)
            return NS(
                hidden_states=h, topk_ids=top.topk_ids, topk_weights=top.topk_weights
            )

        def compute(experts, dispatched):
            events.append("experts")
            # A small independent MoE: expert e multiplies by (e+1).
            return sum(
                (
                    dispatched.hidden_states.float()
                    * (dispatched.topk_ids[:, k : k + 1] + 1)
                    * dispatched.topk_weights[:, k : k + 1]
                )
                for k in range(2)
            ).to(torch.bfloat16)

        def combine(item):
            events.append("combine")
            return item.hidden_states

        def shared(h):
            events.append("shared")
            self.assertTrue(torch.isfinite(h).all())
            return h * 2

        top = MagicMock(side_effect=topk)
        top.empty_topk_output.return_value = NS(
            topk_ids=torch.empty((0, 2), dtype=torch.long),
            topk_weights=torch.empty((0, 2)),
        )
        moe = NS(
            gate=gate,
            topk=top,
            layer_id=3,
            routed_scaling_factor=2.5,
            experts=NS(quant_method=NS(apply_deepep_normal=compute)),
            prefill_shared_experts=shared,
        )
        driver = object.__new__(runtime.Glm53PrefillDeepEP)
        driver.cp_size, driver.cp_rank = size, rank
        driver.combine_input_type = CombineInput
        driver.dispatcher = NS(dispatch=dispatch, combine=combine)
        driver.log_first_forward = False
        out = driver.forward(moe, x, NS(attn_cp_metadata=NS(total_seq_lens=total)))
        expected = torch.zeros_like(x)
        for i, token in enumerate(range(rank, total, size)):
            e0, e1 = token % 16, (token + 1) % 16
            routed = (local[i].float() * (0.25 * (e0 + 1) + 0.75 * (e1 + 1))).to(
                torch.bfloat16
            )
            expected[i] = routed * 2.5 + local[i] * 2
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertEqual(
            events,
            (["gate"] if n else [])
            + ["dispatch", "experts", "combine"]
            + (["shared"] if n else []),
        )

    def test_routing_scale_shared_and_tail_padding(self):
        for total in (0, 3, 8, 19, 32):
            for rank in range(8):
                with self.subTest(total=total, rank=rank):
                    self.run_forward(total, rank)

    def test_extra_physical_padding_never_reaches_router(self):
        for rank in range(8):
            self.run_forward(19, rank, physical=8)

    def test_wrong_dtype_rejected_before_dispatch(self):
        driver = object.__new__(runtime.Glm53PrefillDeepEP)
        with self.assertRaisesRegex(ValueError, "BF16"):
            driver.forward(None, torch.zeros(1, 4), None)


class TestDecoderStageSelection(unittest.TestCase):
    def run_layer(self, cp_active, enabled, sparse=True):
        calls = []

        class MoE:
            def __init__(self):
                self.experts = NS(moe_runner_config=NS(inplace=True))
                self.prefill_deepep = (
                    NS(forward=lambda m, h, b: calls.append("deepep") or h + 100)
                    if enabled
                    else None
                )

            def __call__(self, h, *args):
                calls.append("standard_moe")
                return h + 20

        class MLP:
            def __call__(self, h, *args):
                calls.append("dense")
                return h + 30

        module = extract(
            "srt/models/deepseek_v2.py",
            {"forward"},
            class_name="DeepseekV2DecoderLayer",
            namespace={
                "DeepseekV2MoE": MoE,
                "DeepseekV2MLP": MLP,
                "nullcontext": nullcontext,
                "get_attn_tp_context": lambda: NS(clear_attn_inputs=lambda: None),
                "maybe_prefetch_next_full_attention_kv": lambda *a: None,
                "dsa_use_prefill_cp": lambda *a: cp_active,
                "get_forward": lambda: NS(scoped=lambda **kw: nullcontext()),
            },
        )

        class Attention:
            def maybe_use_decode_attn_tp(self, b):
                return nullcontext()

            def __call__(self, **kwargs):
                return kwargs["hidden_states"] + 1

        comm = NS(
            prepare_attn_and_capture_last_layer_outputs=lambda h, r, *a, **kw: (h, r),
            prepare_mlp=lambda h, r, b: (calls.append("gather_norm") or h + 2, r),
            should_fuse_mlp_allreduce_with_next_layer=lambda b: False,
            should_use_reduce_scatter=lambda b: cp_active,
            postprocess_layer=lambda h, r, b: (calls.append("postprocess") or h, r),
        )
        layer = NS(
            layer_communicator=comm,
            self_attn=Attention(),
            layer_scatter_modes=None,
            _resolve_gfx95_quant_format=lambda: "",
            mlp=MoE() if sparse else MLP(),
            dsa_enable_prefill_cp=True,
            mla_enable_prefill_cp=False,
            post_attention_layernorm=lambda h, r: (
                calls.append("local_norm") or h + 2,
                r,
            ),
        )
        original_residual = torch.ones(2, 4)
        out, residual, _ = module.forward(
            layer, None, torch.zeros(2, 4), NS(), original_residual, None
        )
        self.assertIs(residual, original_residual)
        return calls, out

    def test_decode_uses_identical_standard_path_with_feature_on_and_off(self):
        off_calls, off = self.run_layer(False, False)
        on_calls, on = self.run_layer(False, True)
        self.assertEqual(on_calls, off_calls)
        self.assertEqual(on_calls, ["gather_norm", "standard_moe", "postprocess"])
        torch.testing.assert_close(on, off, rtol=0, atol=0)

    def test_cp_prefill_bypasses_full_gather_and_post_reduce(self):
        calls, out = self.run_layer(True, True)
        self.assertEqual(calls, ["local_norm", "deepep"])
        torch.testing.assert_close(out, torch.full_like(out, 103), rtol=0, atol=0)

    def test_dense_layers_keep_original_path(self):
        calls, _ = self.run_layer(True, True, sparse=False)
        self.assertEqual(calls, ["gather_norm", "dense", "postprocess"])


if __name__ == "__main__":
    unittest.main()
