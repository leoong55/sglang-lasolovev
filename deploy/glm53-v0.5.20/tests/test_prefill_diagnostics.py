import ast
import importlib.util
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prefill_diag", ROOT / "python/sglang/srt/observability/glm53_prefill.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticsTest(unittest.TestCase):
    def test_scheduler_configures_diagnostics_from_parallel_state(self):
        # Execute the actual startup call with 0.5.20's scheduler shape: ranks
        # live only in ParallelState, not as attributes on Scheduler itself.
        tree = ast.parse((ROOT / "python/sglang/srt/managers/scheduler.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_get_new_batch_prefill_raw")
        call = next(n for n in method.body if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "diag.configure")
        code = compile(ast.Module(body=[call], type_ignores=[]), "scheduler.py", "exec")
        for tp, pp, export in [(0, 0, True), (7, 0, False), (0, 1, False)]:
            with self.subTest(tp=tp, pp=pp):
                probe = Mock()
                scheduler = SimpleNamespace(ps=SimpleNamespace(tp_rank=tp, pp_rank=pp))
                exec(code, {"self": scheduler, "diag": probe})
                probe.configure.assert_called_once_with(rank=tp, export=export)

    def tearDown(self):
        diag._pending.clear()
        diag.configure(rank=0, export=False)

    def test_disabled_decorator_preserves_original_function(self):
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_DIAGNOSTICS": "0"}):
            def fn():
                return 7
            self.assertIs(diag.traced("attention")(fn), fn)

    def test_events_are_never_waited_and_incomplete_pair_stays_queued(self):
        start, end = Mock(), Mock()
        end.query.return_value = False
        start.elapsed_time.return_value = 12
        diag._pending.append(("attention", start, end))
        with patch.object(diag, "_observe") as observe:
            diag.poll()
            observe.assert_not_called()
            self.assertEqual(len(diag._pending), 1)
            end.query.return_value = True
            diag.poll()
            observe.assert_called_once_with("attention", "cuda", .012)
        start.synchronize.assert_not_called()
        end.synchronize.assert_not_called()

    def test_only_designated_rank_exports_and_labels_are_bounded(self):
        metrics = (Mock(), Mock(), Mock())
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_DIAGNOSTICS": "1"}), patch.object(diag, "_get_metrics", return_value=metrics):
            diag.configure(rank=3, export=False)
            diag.reject("kv_capacity", rid="secret-request-id")
            metrics[0].labels.assert_not_called()
            diag.configure(rank=0, export=True)
            diag.reject("kv_capacity", rid="secret-request-id")
            metrics[0].labels.assert_called_once_with("kv_capacity")
            with self.assertRaises(ValueError):
                diag.reject("rid-unbounded")


if __name__ == "__main__":
    unittest.main()
