import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import yaml

BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load("install")
renderer = load("render")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "sglang"
        self.bundle = Path(self.tmp.name) / "bundle"
        self.root.mkdir()
        (self.bundle / "runtime").mkdir(parents=True)
        self.manifest = {
            "profile": "test",
            "base_commit": "a" * 40,
            "forbidden_paths": ["other_patch.py"],
            "files": [],
        }
        for path, before, after in [
            ("changed.py", b"before", b"after"),
            ("native.py", b"native", b"native"),
            ("added.py", None, b"new"),
        ]:
            if before is not None:
                (self.root / path).write_bytes(before)
            if before != after:
                (self.bundle / "runtime" / path).write_bytes(after)
            self.manifest["files"].append(
                {
                    "path": path,
                    "base_sha256": None if before is None else sha(before),
                    "sha256": sha(after),
                }
            )
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))

    def run_install(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return installer.install(self.root, self.bundle, **kwargs)

    def test_install_is_idempotent_and_verify_requires_installed_files(self):
        with self.assertRaises(RuntimeError):
            self.run_install(verify_only=True)
        self.run_install()
        self.run_install()
        self.run_install(verify_only=True)
        self.assertEqual((self.root / "changed.py").read_bytes(), b"after")
        self.assertEqual((self.root / "added.py").read_bytes(), b"new")

    def test_wrong_native_base_fails_before_any_mutation(self):
        (self.root / "native.py").write_bytes(b"another runtime")
        with self.assertRaises(RuntimeError):
            self.run_install()
        self.assertEqual((self.root / "changed.py").read_bytes(), b"before")
        self.assertFalse((self.root / "added.py").exists())

    def test_bad_payload_fails_before_any_mutation(self):
        (self.bundle / "runtime/added.py").write_bytes(b"broken payload")
        with self.assertRaises(RuntimeError):
            self.run_install()
        self.assertEqual((self.root / "changed.py").read_bytes(), b"before")

    def test_other_experiment_overlay_is_rejected(self):
        (self.root / "other_patch.py").write_bytes(b"patch")
        with self.assertRaises(RuntimeError):
            self.run_install()


class PackageContractTest(unittest.TestCase):
    def test_source_manifest_matches_every_runtime_file(self):
        manifest = json.loads((BUNDLE / "manifest.json").read_text())
        modified = []
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                self.assertEqual(
                    installer.digest(REPO / "python/sglang" / item["path"]),
                    item["sha256"],
                )
            if item["sha256"] != item["base_sha256"]:
                modified.append(item["path"])
        self.assertEqual(
            set(modified),
            {
                "srt/layers/quantization/w4afp8.py",
                "srt/layers/quantization/w4afp8_humming.py",
                "srt/layers/moe/moe_runner/humming.py",
            },
        )

    def test_render_uses_existing_deployment_service_and_eight_gpus(self):
        image = "ghcr.io/owner/repo@sha256:" + "a" * 64
        deployment, service = yaml.safe_load_all(renderer.render(image))
        spec = deployment["spec"]
        c = spec["template"]["spec"]["containers"][0]
        self.assertEqual(c["image"], image)
        self.assertEqual(spec["replicas"], 1)
        self.assertEqual(spec["strategy"]["type"], "Recreate")
        self.assertEqual(c["resources"]["limits"]["nvidia.com/gpu"], "8")
        self.assertEqual(service["spec"]["selector"], spec["selector"]["matchLabels"])
        self.assertEqual(service["metadata"]["name"], "sglang-glm53-dcp4")
        args = c["args"]
        for key, value in {
            "--tp-size": "4",
            "--pp-size": "2",
            "--ep-size": "4",
            "--attn-cp-size": "1",
            "--dcp-size": "1",
            "--moe-runner-backend": "humming",
        }.items():
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertFalse(any(x.startswith("--speculative-") for x in args))
        self.assertNotIn("--enable-hierarchical-cache", args)
        self.assertNotIn("--enable-prefill-cp", args)
        self.assertNotIn("--disable-radix-cache", args)
        self.assertEqual(
            len([x for x in args if x.startswith("--")]),
            len(set(x for x in args if x.startswith("--"))),
        )
        env = {x["name"]: x["value"] for x in c["env"]}
        self.assertEqual(env["SGLANG_PP_LAYER_PARTITION"], "39,39")

    def test_graphs_cover_local_microbatches_not_global_request_limit(self):
        deployment = next(yaml.safe_load_all(renderer.render("repo:tag")))
        args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]

        def scalar(flag):
            return int(args[args.index(flag) + 1])

        def values(flag):
            out = []
            for x in args[args.index(flag) + 1 :]:
                if x.startswith("--"):
                    break
                out.append(int(x))
            return out

        self.assertEqual(scalar("--max-running-requests"), 48)
        micro = scalar("--pp-max-micro-batch-size")
        self.assertEqual(micro, 24)
        decode = values("--cuda-graph-bs-decode")
        self.assertEqual(decode, sorted(set(decode)))
        self.assertEqual(max(decode), scalar("--cuda-graph-max-bs-decode"))
        self.assertGreaterEqual(max(decode), micro)
        self.assertIn(20, decode)
        prefill = values("--cuda-graph-bs-prefill")
        self.assertEqual(prefill, sorted(set(prefill)))
        self.assertEqual(max(prefill), scalar("--chunked-prefill-size"))
        self.assertEqual(max(prefill), scalar("--cuda-graph-max-bs-prefill"))

    def test_image_argument_cannot_inject_yaml(self):
        for image in ["repo", "repo:tag\nother: value", "repo:tag # comment"]:
            with self.subTest(image=image), self.assertRaises(ValueError):
                renderer.render(image)

    def test_ep4_heuristic_uses_host_shapes_and_keeps_explicit_counts(self):
        path = REPO / "python/sglang/srt/layers/moe/moe_runner/humming.py"
        tree = ast.parse(path.read_text())
        cls = next(
            x
            for x in tree.body
            if isinstance(x, ast.ClassDef) and x.name == "HummingRunnerCore"
        )
        fn = next(
            x
            for x in cls.body
            if isinstance(x, ast.FunctionDef)
            and x.name == "estimate_local_valid_shape_m"
        )
        module = ast.parse("from __future__ import annotations\n")
        module.body.append(fn)
        namespace = {}
        exec(compile(module, str(path), "exec"), namespace)
        estimate = namespace[fn.name]
        core = NS(
            layer=NS(_humming_standard_ep_aware=True),
            num_experts=64,
            global_num_experts=256,
        )
        for tokens in [1, 16, 20, 24, 512, 16384]:
            topk = NS(nelement=lambda: tokens * 8)
            self.assertEqual(estimate(core, topk), tokens * 2)
            self.assertEqual(estimate(core, topk, expected_m=3), 192)
        core.layer._humming_standard_ep_aware = False
        self.assertEqual(estimate(core, NS(nelement=lambda: 160)), 160)


if __name__ == "__main__":
    unittest.main()
