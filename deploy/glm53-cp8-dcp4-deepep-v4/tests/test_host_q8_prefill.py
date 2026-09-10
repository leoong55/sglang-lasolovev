"""Exercise real CP+DCP q8 dispatch with CPU reference slot translation.

CUDA top-k translation, FP8 repacking and the attention kernel are not executed.
The consumer checks the gathered layout against an independent request view.
"""

import enum
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from host_source import DeepseekSparseAttnBackend as Planner
from host_source import extract


class Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    VERIFY = 3
    DRAFT = 4

    def is_decode_or_idle(self):
        return self == Mode.DECODE

    def is_target_verify(self):
        return self == Mode.VERIFY

    def is_draft_extend_v2(self):
        return self == Mode.DRAFT


class Transform(enum.Enum):
    PAGED = 1
    RAGGED = 2


NAMES = {
    "_dsa_impl_for_batch",
    "get_topk_transform_method",
    "forward_extend",
    "_forward_dcp_sparse_q8",
    "get_device_int32_arange",
}
source = extract(
    "srt/layers/attention/dsa_backend.py",
    NAMES,
    class_name="DeepseekSparseAttnBackend",
    namespace={
        "ForwardMode": Mode,
        "TopkTransformMethod": Transform,
        "dsa_use_prefill_cp": lambda fb: fb.attn_cp_metadata is not None,
        "transform_index_page_table_prefill": None,
    },
)
Backend = type("Backend", (), {name: source.__dict__[name] for name in NAMES})


def make_backend():
    b = Backend()
    b.dcp_enabled = True
    b.dsa_prefill_impl = "flashmla_sparse_q8"
    b.dsa_decode_impl = "flashmla_kv"
    b.dsa_kv_cache_store_fp8 = True
    b.use_fused_topk = False
    b.use_mha = False
    b.hisparse_coordinator = None
    b.dcp_size, b.dcp_rank = 4, 2
    b._arange_buf = torch.arange(128, dtype=torch.int32)
    b.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=MagicMock(side_effect=AssertionError("Read local DCP shard"))
    )
    return b


class TestDCPQ8Prefill(unittest.TestCase):
    def test_batch_dispatch_and_small_non_cp_tail(self):
        b = make_backend()
        for mode in Mode:
            for cp in (False, True):
                with self.subTest(mode=mode, cp=cp):
                    fb = SimpleNamespace(
                        forward_mode=mode, attn_cp_metadata=object() if cp else None
                    )
                    expected = (
                        "flashmla_sparse_q8"
                        if mode == Mode.EXTEND and cp
                        else "flashmla_kv"
                    )
                    self.assertEqual(b._dsa_impl_for_batch(fb), expected)
        b.dcp_enabled = False
        self.assertEqual(
            b._dsa_impl_for_batch(
                SimpleNamespace(forward_mode=Mode.EXTEND, attn_cp_metadata=None)
            ),
            "flashmla_sparse_q8",
        )

    def test_dcp_uses_paged_mapping_and_non_dcp_keeps_ragged(self):
        b = make_backend()
        self.assertEqual(b.get_topk_transform_method(Mode.EXTEND), Transform.PAGED)
        b.dcp_enabled = False
        self.assertEqual(b.get_topk_transform_method(Mode.EXTEND), Transform.RAGGED)
        self.assertEqual(b.get_topk_transform_method(Mode.DECODE), Transform.PAGED)

    def _check_forward(self, prefix_lengths, current_lengths, selected_requests):
        # Independently construct [all prefixes][all current tokens]. Mark
        # each row by its request and logical position, so wrong layout or
        # request-major offsets produce different attention results.
        logical = [
            [10 * r + p + 1 for p in range(pre + cur)]
            for r, (pre, cur) in enumerate(zip(prefix_lengths, current_lengths))
        ]
        packed_order = [
            x for r, pre in enumerate(prefix_lengths) for x in logical[r][:pre]
        ]
        packed_order += [
            x for r, pre in enumerate(prefix_lengths) for x in logical[r][pre:]
        ]
        position = {value: i for i, value in enumerate(packed_order)}
        kv_indices = torch.tensor(
            [position[value] for row in logical for value in row], dtype=torch.int32
        )
        indptr = torch.tensor(
            [0] + list(__import__("itertools").accumulate(map(len, logical))),
            dtype=torch.int32,
        )
        dcp = SimpleNamespace(dcp_kv_indices=kv_indices, dcp_kv_indptr=indptr)
        page_table = Planner._build_dcp_prefill_page_table(
            Planner(),
            seq_lens=torch.tensor([len(logical[r]) for r in selected_requests]),
            dcp_meta=dcp,
            selected_batch_indices=selected_requests,
        )
        n = len(selected_requests)
        gathered = torch.zeros((64, 1, 656), dtype=torch.float8_e4m3fn)
        # Synthetic packed rows: only byte zero is used by the reference
        # consumer below; actual FP8 packing is tested separately in v1.
        gathered.view(torch.uint8)[: len(packed_order), 0, 0] = torch.tensor(
            packed_order, dtype=torch.uint8
        )
        dcp.dcp_kv_buffer = gathered
        fb = SimpleNamespace(
            forward_mode=Mode.EXTEND, attn_cp_metadata=object(), attn_dcp_metadata=dcp
        )
        b = make_backend()
        b.forward_metadata = SimpleNamespace(
            dcp_page_table_1=page_table,
            dsa_extend_seq_lens_list=[1] * n,
            cu_seqlens_q=torch.arange(n + 1),
        )
        topk = torch.tensor(
            [[len(logical[r]) - 1, 0, -1] for r in selected_requests], dtype=torch.int32
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            head_dim=576,
            v_head_dim=512,
            scaling=0.25,
            layer_id=7,
        )

        def translate(**kw):
            self.assertEqual((kw["dcp_size"], kw["dcp_rank"]), (1, 0))
            self.assertIs(kw["page_table"], page_table)
            return torch.stack(
                [
                    torch.tensor([int(page_table[i, j]) if j >= 0 else -1 for j in row])
                    for i, row in enumerate(kw["topk_indices"])
                ]
            ).to(torch.int32)

        def consume(**kw):
            self.assertIs(kw["paged_kv_cache"], gathered)
            torch.testing.assert_close(
                kw["page_table_1_flattened"], torch.arange(64, dtype=torch.int32)
            )
            self.assertIsNone(kw["kv_bf16"])
            self.assertEqual(kw["layer_id"], 7)
            values = gathered.view(torch.uint8)[:, 0, 0].float()
            result = []
            for row in kw["page_table_1"]:
                keys = values[row[row >= 0].long()]
                result.append((torch.softmax(keys * 0.25, dim=0) * keys).sum())
            return torch.stack(result)

        b._forward_flashmla_sparse_q8kv8 = MagicMock(side_effect=consume)
        with patch.object(source, "transform_index_page_table_prefill", translate):
            out = b.forward_extend(
                q=torch.zeros((n, 2, 512)),
                k=None,
                v=None,
                q_rope=torch.zeros((n, 2, 64)),
                layer=layer,
                forward_batch=fb,
                topk_indices=topk,
                save_kv_cache=False,
            )
        expected = []
        for r in selected_requests:
            keys = torch.tensor([logical[r][-1], logical[r][0]], dtype=torch.float32)
            expected.append((torch.softmax(keys * 0.25, dim=0) * keys).sum())
        torch.testing.assert_close(out, torch.stack(expected))
        b._forward_flashmla_sparse_q8kv8.assert_called_once()
        b.token_to_kv_pool.get_key_buffer.assert_not_called()

    def test_cached_prefix_and_cp_request_subset(self):
        self._check_forward([1, 2, 1], [3, 4, 5], [1, 2])

    def test_cold_prefill_and_cp_request_subset(self):
        self._check_forward([0, 0, 0], [5, 6, 7], [0, 2])

    def test_q8_adapter_rejects_wrong_kv_dtype(self):
        b = make_backend()
        with self.assertRaisesRegex(RuntimeError, "packed FP8"):
            b._forward_dcp_sparse_q8(None, None, torch.zeros((64, 1, 656)), None, None)


if __name__ == "__main__":
    unittest.main()
