import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


launch = module("topology_launch", BUNDLE / "launch.py")
installer = module("topology_install", BUNDLE / "install.py")
observe = module("topology_observe", BUNDLE / "observe.py")


def value(command, option):
    return command[command.index(option) + 1]


class LauncherTests(unittest.TestCase):
    def test_matrix_uses_eight_gpus_and_disables_context_parallelism(self):
        for profile, expected in launch.PROFILES.items():
            with self.subTest(profile=profile):
                command, env = launch.make_launch(profile, "/models/full", environ={})
                self.assertEqual(
                    int(value(command, "--tp-size")) * int(value(command, "--pp-size")),
                    8,
                )
                self.assertEqual(value(command, "--ep-size"), str(expected["ep"]))
                self.assertEqual(value(command, "--dcp-size"), "1")
                self.assertEqual(value(command, "--attn-cp-size"), "1")
                self.assertEqual(value(command, "--moe-a2a-backend"), "none")
                self.assertNotIn("--speculative-algorithm", command)
                self.assertNotIn("--enable-prefill-cp", command)
                self.assertNotIn("--enable-hierarchical-cache", command)
                self.assertNotIn("--disable-radix-cache", command)
                self.assertEqual(value(command, "--max-running-requests"), "48")
                self.assertEqual(value(command, "--chunked-prefill-size"), "16384")
                self.assertEqual(value(command, "--quantization"), "w4afp8")
                self.assertIn("--disable-shared-experts-fusion", command)
                graphs = json.loads(value(command, "--cuda-graph-config"))
                self.assertEqual(graphs["prefill"]["backend"], "disabled")
                self.assertEqual(graphs["decode"]["backend"], "full")
                local_limit = 24 if profile == "pp2" else 48 // expected["dp"]
                self.assertEqual(
                    graphs["decode"]["bs"], list(range(1, local_limit + 1))
                )
                if profile == "pp2":
                    self.assertNotIn("--enable-dp-attention", command)
                    self.assertEqual(env["SGLANG_PP_LAYER_PARTITION"], "39,39")
                    self.assertEqual(value(command, "--pp-max-micro-batch-size"), "24")
                    self.assertEqual(value(command, "--pp-async-batch-depth"), "0")
                    self.assertEqual(env["GLM53_PP_OBSERVER"], "1")
                    self.assertEqual(env["GLM53_DPA_OBSERVER"], "0")
                else:
                    self.assertEqual(env["GLM53_DPA_OBSERVER"], "1")
                    self.assertEqual(env["GLM53_PP_OBSERVER"], "0")
                    self.assertIn("--enable-dp-attention", command)
                    self.assertEqual(
                        value(command, "--load-balance-method"), "round_robin"
                    )
                    self.assertNotIn("SGLANG_PP_LAYER_PARTITION", env)

    def test_hicache_is_explicit_and_has_no_storage_backend(self):
        for profile in launch.PROFILES:
            command, _ = launch.make_launch(
                profile, "/models/full", hicache=True, environ={}
            )
            self.assertEqual(value(command, "--hicache-size"), "32")
            self.assertEqual(value(command, "--hicache-write-policy"), "write_through")
            self.assertEqual(value(command, "--hicache-io-backend"), "direct")
            self.assertEqual(value(command, "--hicache-mem-layout"), "layer_first")
            self.assertNotIn("--hicache-storage-backend", command)

    def test_inherited_experiment_overrides_are_rejected(self):
        for key in [
            "SGLANG_CP_FUSION",
            "SGLANG_DCP_MODE",
            "SGLANG_ENABLE_CP_V2",
            "SGLANG_DFLASH_BLOCK_SIZE",
            "SGLANG_SPECULATIVE_ALGORITHM",
            "SGLANG_PP_LAYER_PARTITION",
            "SGLANG_PP_FULL_NEED",
            "SGLANG_HICACHE_RATIO",
            "SGLANG_EXTRA_ARGS",
            "SGLANG_ENABLE_DEEPEP",
        ]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Inherited"):
                launch.make_launch("pp2", "/models/full", environ={key: "0"})
        _, env = launch.make_launch(
            "pp2",
            "/models/full",
            environ={"NCCL_DEBUG": "WARN", "SGLANG_SET_CPU_AFFINITY": "0"},
        )
        self.assertEqual(env["NCCL_DEBUG"], "WARN")

    def test_no_unrecognized_options_against_pinned_server_args(self):
        tree = ast.parse((REPO / "python/sglang/srt/server_args.py").read_text())
        flags = {
            "--" + n.target.id.replace("_", "-")
            for n in ast.walk(tree)
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        flags.update(
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("--")
            and " " not in n.value
        )
        for profile in launch.PROFILES:
            command, _ = launch.make_launch(
                profile, "/models/full", hicache=True, environ={}
            )
            self.assertEqual(
                [
                    option
                    for option in command
                    if option.startswith("--") and option not in flags
                ],
                [],
            )

    def test_model_geometry_is_checked_before_allocating_gpu_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["GlmMoeDsaForCausalLM"],
                        "num_hidden_layers": 78,
                        "quantization_config": {"quant_method": "w4afp8"},
                    }
                )
            )
            launch.validate_model(directory)
            path.write_text(
                json.dumps(
                    {"architectures": ["GlmMoeDsaForCausalLM"], "num_hidden_layers": 45}
                )
            )
            with self.assertRaisesRegex(ValueError, "78 layers"):
                launch.validate_model(directory)


class SourceVerificationTests(unittest.TestCase):
    def test_baked_revision_is_required_and_compared_to_expected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no baked"):
                launch.validate_revision(directory, environ={"SOURCE_COMMIT": "a" * 40})
            self.assertIsNone(
                launch.validate_revision(directory, environ={}, required=False)
            )
            path = Path(directory) / "image-revision"
            path.write_text("a" * 40 + "\n")
            self.assertEqual(
                launch.validate_revision(
                    directory, environ={"SOURCE_COMMIT": "a" * 40}
                ),
                "a" * 40,
            )
            with self.assertRaisesRegex(ValueError, "SOURCE_COMMIT must"):
                launch.validate_revision(directory, environ={})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                launch.validate_revision(directory, environ={"SOURCE_COMMIT": "b" * 40})
            path.write_text("unknown\n")
            with self.assertRaisesRegex(ValueError, "full commit SHA"):
                launch.validate_revision(directory, environ={}, required=False)

    def test_pinned_checkout_verifies(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = installer.verify(REPO / "python/sglang", BUNDLE)
        self.assertEqual(manifest["base_commit"], launch.BASE_COMMIT)

    def test_mismatch_and_foreign_overlay_fail_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runtime.py"
            source.write_bytes(b"pinned\n")
            manifest = {
                "base_commit": "pinned",
                "files": [
                    {
                        "path": "runtime.py",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
                "forbidden_paths": ["unexpected.py"],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            with contextlib.redirect_stdout(io.StringIO()):
                installer.verify(root, root)
            source.write_bytes(b"foreign patch\n")
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                installer.verify(root, root)
            self.assertEqual(source.read_bytes(), b"foreign patch\n")
            source.write_bytes(b"pinned\n")
            (root / "unexpected.py").write_text("overlay")
            with self.assertRaisesRegex(RuntimeError, "unexpected experiment"):
                installer.verify(root, root)


class ObserverTests(unittest.TestCase):
    def dpa_scheduler(self):
        return SimpleNamespace(
            ps=SimpleNamespace(
                pp_size=1,
                pp_rank=0,
                dp_size=2,
                dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
            ),
            server_args=SimpleNamespace(
                pp_size=1,
                dp_size=2,
                enable_dp_attention=True,
                attn_cp_size=1,
                dcp_size=1,
                enable_prefill_cp=False,
                dwdp_size=1,
                elastic_ep_backend=None,
                speculative_algorithm=None,
                enable_two_batch_overlap=False,
                disaggregation_mode="null",
                moe_a2a_backend="none",
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            require_mlp_sync=True,
            waiting_queue=[],
        )

    def dpa_batch(self, step=15):
        def req(rid, **changes):
            return SimpleNamespace(
                **(
                    {
                        "rid": rid,
                        "finished_reason": None,
                        "is_retracted": False,
                        "output_ids": [1, 2],
                    }
                    | changes
                )
            )

        return SimpleNamespace(
            forward_iter=step,
            forward_mode=SimpleNamespace(name="DECODE"),
            reqs=[
                req("live"),
                req("finished", finished_reason="length"),
                req("retracted", is_retracted=True),
                req("queued"),
            ],
        )

    def test_dpa_runtime_conditions_reject_unsupported_sync_variants(self):
        scheduler = self.dpa_scheduler()
        self.assertTrue(all(observe.dpa_sync_conditions(scheduler, False).values()))
        self.assertFalse(all(observe.dpa_sync_conditions(scheduler, True).values()))
        for field, value in {
            "pp_size": 2,
            "dp_size": 1,
            "enable_dp_attention": False,
            "attn_cp_size": 2,
            "dcp_size": 2,
            "enable_prefill_cp": True,
            "dwdp_size": 8,
            "elastic_ep_backend": "elastic",
            "speculative_algorithm": "draft",
            "enable_two_batch_overlap": True,
            "disaggregation_mode": "prefill",
            "moe_a2a_backend": "deepep",
        }.items():
            with self.subTest(field=field):
                candidate = self.dpa_scheduler()
                setattr(candidate.server_args, field, value)
                self.assertFalse(
                    all(observe.dpa_sync_conditions(candidate, False).values())
                )
        scheduler.require_mlp_sync = False
        self.assertFalse(all(observe.dpa_sync_conditions(scheduler, False).values()))

    def test_dpa_batch_reads_only_live_survivors_and_emits_idle_zero(self):
        scheduler, batch = self.dpa_scheduler(), self.dpa_batch()
        scheduler.waiting_queue = [batch.reqs[-1]]
        conditions = observe.dpa_sync_conditions(scheduler, False)
        record = observe.dpa_snapshot(scheduler, batch, conditions)
        self.assertEqual(record["rids"], ["live"])
        self.assertEqual(record["forward_iter"], 15)
        self.assertEqual(record["output_lengths"], {"live": 2})
        batch.reqs = []
        batch.forward_mode.name = "IDLE"
        self.assertEqual(observe.dpa_snapshot(scheduler, batch, conditions)["count"], 0)
        conditions["scheduler_all_gather_enabled"] = False
        self.assertFalse(observe.dpa_snapshot(scheduler, batch, conditions)["valid"])

    def test_dpa_emission_cadence_representative_rank_and_failure_isolation(self):
        scheduler, batch = self.dpa_scheduler(), self.dpa_batch()
        dependency = SimpleNamespace(
            should_skip_scheduler_all_gather=lambda size: False
        )
        with patch.dict(
            sys.modules,
            {"sglang.srt.managers.scheduler_components.dp_attn": dependency},
        ), patch.object(observe.os, "write") as write:
            observe._emit_dpa(scheduler, batch)
            self.assertEqual(write.call_count, 1)
            record = json.loads(
                write.call_args.args[1].decode().split(observe.DPA_PREFIX)[1]
            )
            self.assertTrue(record["valid"])
            batch.forward_iter = 16
            observe._emit_dpa(scheduler, batch)
            self.assertEqual(write.call_count, 1)
            batch.forward_iter = 30
            scheduler.ps.attn_tp_rank = 1
            observe._emit_dpa(scheduler, batch)
            self.assertEqual(write.call_count, 1)
            scheduler.ps.attn_tp_rank = 0
            del scheduler.server_args.dwdp_size
            observe._emit_dpa(scheduler, batch)
            record = json.loads(
                write.call_args.args[1].decode().split(observe.DPA_PREFIX)[1]
            )
            self.assertFalse(record["valid"])

    def test_dpa_wrapper_runs_stock_handler_once_then_observes_and_preserves_errors(
        self,
    ):
        observed = []

        class Scheduler:
            def __init__(self):
                self.ps = SimpleNamespace(pp_size=1, dp_size=2)

            def process_batch_result(self, batch, result):
                observed.append("stock")
                batch.finished = True
                if result == "fail":
                    raise RuntimeError("stock failure")
                return result

        with patch.dict(os.environ, {"GLM53_DPA_OBSERVER": "1"}):
            observe.wrap_scheduler(Scheduler)
            observe.wrap_scheduler(Scheduler)
        scheduler, batch = Scheduler(), SimpleNamespace(finished=False)
        with patch.object(
            observe, "_emit_dpa", side_effect=lambda s, b: observed.append(b.finished)
        ):
            self.assertEqual(scheduler.process_batch_result(batch, "value"), "value")
        self.assertEqual(observed, ["stock", True])
        with patch.object(
            observe, "_emit_dpa", side_effect=RuntimeError("observer failure")
        ):
            self.assertEqual(scheduler.process_batch_result(batch, "value"), "value")
        with self.assertRaisesRegex(RuntimeError, "stock failure"):
            scheduler.process_batch_result(batch, "fail")

    def test_pp_union_deduplicates_and_excludes_waiting_finished_and_retracted(self):
        def req(rid, finished=None, retracted=False, output_length=4):
            return SimpleNamespace(
                rid=rid,
                finished_reason=finished,
                is_retracted=retracted,
                output_ids=[1] * output_length,
            )

        a, b, c = req("a"), req("b"), req("c")
        done, waiting, retracted = (
            req("done", "length"),
            req("waiting"),
            req("retracted", retracted=True),
        )

        def batch(*reqs):
            return SimpleNamespace(reqs=list(reqs))

        scheduler = SimpleNamespace(
            running_mbs=[batch(a, done), batch(b, waiting, retracted)],
            running_batch=batch(a),
            mbs=[batch(a, c), None],
            waiting_queue=[waiting],
            chunked_req=c,
        )
        observed = observe.snapshot(scheduler)
        self.assertEqual(observed["rids"], ["a", "b", "c"])
        self.assertEqual(observed["count"], 3)
        self.assertEqual(observed["running_count"], 2)
        self.assertEqual(observed["inflight_count"], 2)
        self.assertEqual(observed["chunked_rids"], ["c"])
        self.assertEqual(observed["stage"], 0)
        self.assertEqual(observed["tp"], 0)
        self.assertTrue(observed["valid"])
        self.assertEqual(observed["output_lengths"], {"a": 4, "b": 4, "c": 4})

    def test_uninitialized_loop_is_not_reported_as_zero_activity(self):
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            observe.snapshot(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
