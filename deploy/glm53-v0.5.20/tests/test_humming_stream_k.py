"""Portable regression against the real Humming configuration method."""
import __future__
import ast
import copy
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as N

SOURCE = Path(os.environ.get('HUMMING_RUNNER_SOURCE', str(Path(__file__).resolve().parents[3] / 'python/sglang/srt/layers/moe/moe_runner/humming.py')))
TREE = ast.parse(SOURCE.read_text())
CORE = next(x for x in TREE.body if isinstance(x, ast.ClassDef) and x.name == 'HummingRunnerCore')
METHOD = next(x for x in CORE.body if isinstance(x, ast.FunctionDef) and x.name == 'get_humming_gemm_configs')

class TestHummingStreamK(unittest.TestCase):
    def setup_case(self, disabled=None):
        defaults = [(0, 2048, {'block_shape': [64, 128, 128], 'use_stream_k': True}), (2048, 100000, {'block_shape': [128, 128, 128], 'use_stream_k': True})]
        namespace = {'json': json, 'envs': N(SGLANG_HUMMING_USE_F16_ACCUM=N(get=lambda: False)), 'HummingMethod': N(get_default_tuning_configs=lambda **kw: defaults)}
        exec(compile(ast.Module(body=[METHOD], type_ignores=[]), str(SOURCE), 'exec', flags=__future__.annotations.compiler_flag), namespace)
        layer = N()
        if disabled is not None:
            layer._humming_disable_stream_k = disabled
        core = N(layer=layer, humming_gemm_configs={})
        return namespace['get_humming_gemm_configs'], core, defaults

    def test_off_preserves_tiles_scales_and_shared_defaults(self):
        method, core, defaults = self.setup_case(True)
        before = copy.deepcopy(defaults)
        result = method(core, N(value='indexed'))
        self.assertEqual(defaults, before)
        for name in ['w13_tuning_config', 'w2_tuning_config']:
            for original, changed in zip(defaults, result[name]):
                self.assertEqual(original[:2], changed[:2])
                self.assertFalse(changed[2]['use_stream_k'])
                self.assertEqual({k:v for k,v in original[2].items() if k!='use_stream_k'}, {k:v for k,v in changed[2].items() if k!='use_stream_k'})
            self.assertEqual(json.loads(result[name+'_str']), json.loads(json.dumps(result[name])))
        self.assertFalse(result['compute_config']['use_f16_accum'])
        self.assertIs(result, method(core, N(value='indexed')))

    def test_default_and_unrelated_layers_preserve_control(self):
        for disabled in [None, False]:
            method, core, defaults = self.setup_case(disabled)
            result = method(core, N(value='indexed'))
            self.assertEqual(result['w13_tuning_config'], defaults)
            self.assertEqual(result['w2_tuning_config'], defaults)

if __name__ == '__main__':
    unittest.main()
