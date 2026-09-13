import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
                else:
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
