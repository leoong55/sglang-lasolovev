import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(os.environ.get("SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
spec = importlib.util.spec_from_file_location("dflash_contract", ROOT / "python/sglang/srt/layers/cp/glm53_dflash.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class VerifyContractTest(unittest.TestCase):
    def test_live_and_padded_graph_rows(self):
        for bs in (1, 2, 4, 8, 16, 32):
            for width in (2, 8, 16):
                batch = NS(batch_size=bs, input_ids=range(bs*width),
                           spec_info=NS(draft_token_num=width, custom_mask=None, ragged_verify_layout=None))
                helper.validate_verify_layout(batch, width)
                helper.validate_flashmla_rows(bs*width, bs*width, bs*width, bs*width+1)
                batch.input_ids = range(bs*width-1)
                with self.assertRaises(ValueError): helper.validate_verify_layout(batch, width)

    def test_tree_or_ragged_verify_cannot_enter_linear_path(self):
        for change in (dict(custom_mask=object()), dict(ragged_verify_layout=object()), dict(draft_token_num=8)):
            args=dict(draft_token_num=16, custom_mask=None, ragged_verify_layout=None)
            args.update(change)
            with self.assertRaises(ValueError):
                helper.validate_verify_layout(NS(batch_size=1,input_ids=range(16),spec_info=NS(**args)),16)
        for sizes in ((16,1,16,17),(16,16,1,17),(16,16,16,2)):
            with self.assertRaises(ValueError): helper.validate_flashmla_rows(*sizes)


if __name__ == "__main__": unittest.main()
