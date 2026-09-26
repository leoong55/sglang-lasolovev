"""Portable regression: loaded physical weights must follow the logical router.

Execute the real normal forward method with tiny CPU stand-ins for GPU kernels.
This reproduces wrong expert selection with a fixed layout and online EPLB off.
CUDA Humming arithmetic and full-model numerical parity need separate gates.
"""
import __future__
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'python/sglang/srt/models/deepseek_v2.py'
TREE = ast.parse(SOURCE.read_text())


def method(name):
    return next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)


class TestFixedExpertPlacement(unittest.TestCase):
    def namespace(self, fixed, enabled):
        # Physical slot0 contains logical expert2, slot1 logical0, slot2 logical1.
        inverse = [1, 2, 0, 3]
        return dict(
            get_exec=lambda: N(moe=N(enable_eplb=enabled,
                                      init_expert_location='fixed.json' if fixed else 'trivial')),
            ExpertLocationDispatchInfo=N(init_new=lambda **_: inverse if fixed else None),
            _is_cuda=True, _is_musa=False, _is_xpu=False, _use_aiter=False,
            KTEPWrapperMethod=type('UnusedQuantMethod', (), {}),
            maybe_fuse_routed_scale_and_shared_add=lambda _, output, *args: output,
        )

    def test_normal_routes_to_the_loaded_logical_expert(self):
        for fixed, enabled, draft in [(True, False, False), (True, True, False),
                                       (False, False, False), (True, False, True)]:
            with self.subTest(fixed=fixed, online_eplb=enabled, draft=draft):
                namespace = self.namespace(fixed, enabled)
                fn = method('forward_normal')
                exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec',
                             flags=__future__.annotations.compiler_flag), namespace)
                weights = [30, 10, 20, 40] if fixed and not draft else [10, 20, 30, 40]

                class Experts:
                    moe_runner_config = N(inplace=True)
                    quant_method = N()

                    def __call__(self, hidden, topk, **kwargs):
                        return weights[topk]

                obj = N(layer_id=3, is_nextn=draft, experts=Experts(), tp_size=1,
                        _fuse_shared_experts_inside_sbo=False, _shared_expert_tp1=False,
                        _maybe_quant_moe_input_once=lambda _: None,
                        _forward_shared_experts=lambda *args, **kwargs: None,
                        gate=lambda *args: [1, 0, 0, 0], routed_scaling_factor=1,
                        topk=lambda _, logits, expert_location_dispatch_info=None, **kwargs:
                        expert_location_dispatch_info[0] if expert_location_dispatch_info is not None else 0)
                result = namespace['forward_normal'](obj, N(shape=(1, 4)))
                self.assertEqual(result, 10, 'logical expert0 must contribute once')

    def test_dual_stream_uses_the_same_fixed_layout_without_online_eplb(self):
        fn = method('forward_normal_dual_stream')
        assignment = next(node for node in fn.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'dispatch_info'
                                  for t in node.targets))
        for draft in (False, True):
            namespace = self.namespace(True, False)
            namespace['self'] = N(layer_id=3, is_nextn=draft)
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), 'exec'), namespace)
            self.assertEqual(namespace['dispatch_info'], None if draft else [1, 2, 0, 3])


if __name__ == '__main__':
    unittest.main()
