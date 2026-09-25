"""Match actual DFlash pool shapes to its budget without allocating GPU memory."""

import ast
import logging
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def extract(path, names, namespace, cls=None):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    nodes = tree.body
    if cls is not None:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls)
    body = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names), names
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


class DraftPoolTest(unittest.TestCase):
    def fixture(self, *, tp=8, cp=8, dcp=4, total_heads=8, draft=True, dflash=True):
        parallel = NS(tp_size=tp, attn_tp_size=tp // cp, attn_dcp_size=dcp)
        self.parallel = parallel
        schedule = NS(page_size=64, prefill_only_disable_kv_cache=False)
        methods = extract(
            "configs/model_config.py", {"get_num_kv_heads"}, {}, "ModelConfig"
        )
        model_cls = type("Config", (), methods)
        config = model_cls()
        config.is_draft_model = draft
        config.get_total_num_kv_heads = lambda: total_heads
        config.head_dim = config.v_head_dim = 128
        config.hf_config = NS()
        methods = extract(
            "mem_cache/kv_cache_configurator.py",
            {"_build_mha_kv_pool", "_derive_pool_sizes", "loc_space_scale", "pool_page_size"},
            dict(
                get_parallel=lambda: parallel,
                get_schedule=lambda: schedule,
                get_exec=lambda: NS(features=NS(enable_memory_saver=False)),
                get_disagg=lambda: NS(enable_pdmux=False),
                get_spec=lambda: NS(speculative_algorithm="DFLASH" if dflash else "EAGLE"),
                is_deepseek_v4=lambda _: False,
                _PoolSizes=lambda **kw: NS(**kw),
                logger=logging.getLogger(__name__),
            ),
            "KVCacheConfigurator",
        )
        kvc = type("Configurator", (), methods)()
        kvc.model_config = config
        kvc.is_draft_worker = draft
        kvc.spec_algorithm = NS(is_dflash_family=lambda: dflash)
        kvc.kv_cache_dtype_str = "auto"
        kvc.kv_cache_dtype = "bf16"
        kvc.post_capture_kv_active = False
        kvc.layer_info = NS(num_effective_layers=6, start_layer=0, end_layer=6)
        kvc.device = "cuda"
        kvc.is_hybrid_swa = False
        kvc.spec_aux_config = NS(dflash_draft_cell_size_per_token=3072)
        return kvc

    def pool(self, kvc, tokens=493312):
        sizes = kvc._derive_pool_sizes(
            config=NS(
                max_total_num_tokens=tokens,
                max_running_requests=32,
                unified_total_bytes=None,
            )
        )

        def capture_pool(size, **kw):
            return NS(size=size, use_hnd=False, **kw)

        pool = kvc._build_mha_kv_pool(
            max_total_num_tokens=sizes.max_total_num_tokens,
            mha_pool_class=capture_pool,
        )
        shape = extract(
            "mem_cache/memory_pool.py", {"_kv_buffer_shapes"}, {}, "MHATokenToKVPool"
        )["_kv_buffer_shapes"]
        return pool, shape(pool)

    def test_uploaded_oom_geometry_matches_one_head_not_eight(self):
        kvc = self.fixture()
        # The old builder passed target attn_tp=1. Its tensor bytes rounded
        # to a 2 MiB CUDA allocation match the uploaded allocator OOM request.
        old_heads = kvc.model_config.get_num_kv_heads(1, 4)
        self.assertEqual(old_heads, 8)
        old_tensor_bytes = (493312 + 64) * 4 * old_heads * 128 * 2
        allocation_unit = 2 * 1024**2
        self.assertEqual(
            math.ceil(old_tensor_bytes / allocation_unit) * allocation_unit,
            4043309056,
        )
        pool, shapes = self.pool(kvc)
        self.assertEqual(pool.head_num, 1)
        self.assertEqual(shapes, ((1973504, 1, 128), (1973504, 1, 128)))
        self.assertEqual(sum(math.prod(s) * 2 for s in shapes) * 6, 6062604288)

    def test_allocated_bytes_match_existing_dflash_budget_including_padding(self):
        kvc = self.fixture()
        byte_fn = extract(
            "speculative/dflash_utils.py",
            {"dflash_draft_cell_size_per_token"},
            {"torch": NS(_utils=NS(_element_size=lambda dtype: 2))},
        )["dflash_draft_cell_size_per_token"]
        cost = byte_fn(
            draft_model_config=kvc.model_config,
            draft_num_layers=6,
            draft_kv_cache_dtype="bf16",
            tp_size=8,
        )
        self.assertEqual(cost, 3072)
        budget_fn = extract(
            "model_executor/pool_configurator.py",
            {"_dflash_draft_cell_size"},
            {"get_parallel": lambda: self.parallel},
        )["_dflash_draft_cell_size"]
        target = NS(
            is_draft_worker=False,
            spec_algorithm=kvc.spec_algorithm,
            spec_aux_config=NS(dflash_draft_cell_size_per_token=cost),
        )
        pool, shapes = self.pool(kvc)
        allocated = sum(math.prod(s) * 2 for s in shapes) * pool.layer_num
        self.assertEqual(allocated, (493312 + 64) * budget_fn(target))
        self.assertEqual(budget_fn(kvc), 0)

    def test_dflash_heads_follow_tp_while_slots_still_follow_dcp(self):
        for tp, cp, dcp in ((8, 8, 4), (8, 4, 2), (8, 1, 4), (2, 2, 2), (1, 1, 1)):
            for total_heads in (1, 8, 16):
                with self.subTest(tp=tp, cp=cp, dcp=dcp, heads=total_heads):
                    kvc = self.fixture(tp=tp, cp=cp, dcp=dcp, total_heads=total_heads)
                    pool, shapes = self.pool(kvc, tokens=256)
                    self.assertEqual(pool.head_num, max(1, total_heads // tp))
                    self.assertEqual(pool.size, 256 * dcp)
                    self.assertEqual(shapes[0][0], (256 + 64) * dcp)

    def test_target_and_other_draft_geometry_stays_on_existing_path(self):
        for draft, dflash in ((False, True), (False, False), (True, False)):
            kvc = self.fixture(tp=8, cp=2, dcp=2, draft=draft, dflash=dflash)
            pool = kvc._build_mha_kv_pool(
                max_total_num_tokens=256,
                mha_pool_class=lambda size, **kw: NS(**kw),
            )
            self.assertEqual(pool.head_num, kvc.model_config.get_num_kv_heads(4, 2))


if __name__ == "__main__":
    unittest.main()
