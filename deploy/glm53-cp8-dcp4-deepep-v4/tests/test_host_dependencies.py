"""Regression: the packaged DeepEP checks CUDA devices during import."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, KIT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDependencyStages(unittest.TestCase):
    def test_build_avoids_package_init_and_runtime_keeps_driver_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "import_attempted"
            for name in ("deep_ep", "sgl_kernel"):
                package = root / name
                package.mkdir()
                (package / "__init__.py").write_text(
                    "from pathlib import Path\n"
                    f"Path({str(marker)!r}).write_text('imported')\n"
                    "raise ImportError('CUDA device unavailable: test prerequisite')\n"
                )
            env = dict(os.environ, PYTHONPATH=str(root))
            build = subprocess.run(
                [sys.executable, str(KIT / "check_dependencies.py"), "--build"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            self.assertFalse(marker.exists(), "build imported a GPU package")
            runtime = subprocess.run(
                [sys.executable, str(KIT / "check_dependencies.py"), "--runtime"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(runtime.returncode, 0)
            self.assertTrue(marker.exists())
            self.assertIn("CUDA device unavailable: test prerequisite", runtime.stderr)

    def test_build_still_rejects_missing_package(self):
        dependencies = load_file("dependencies_tested", "check_dependencies.py")
        with patch.object(
            dependencies,
            "find_spec",
            side_effect=lambda name: None if name == "deep_ep" else object(),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Missing installed packages: deep_ep"
            ):
                dependencies.check_build()


class TestLauncherDependencyGate(unittest.TestCase):
    def setUp(self):
        dependencies = load_file("dependencies_tested", "check_dependencies.py")
        installer = SimpleNamespace(
            install=lambda *a, **kw: None, package_root=lambda: KIT
        )
        with patch.dict(
            sys.modules, {"check_dependencies": dependencies, "install": installer}
        ):
            self.launch = load_file("launch_tested", "launch.py")

    def test_runtime_check_precedes_server_and_is_skipped_for_disabled_feature(self):
        for enabled in ("0", "1"):
            events = []
            with (
                self.subTest(enabled=enabled),
                patch.dict(os.environ, {"SGLANG_GLM53_DEEPEP_PREFILL": enabled}),
                patch.object(
                    self.launch,
                    "validate",
                    side_effect=lambda args: events.append("validate"),
                ),
                patch.object(
                    self.launch,
                    "check_runtime",
                    side_effect=lambda: events.append("runtime"),
                ),
                patch.object(
                    self.launch.os,
                    "execv",
                    side_effect=lambda *args: events.append("server"),
                ),
            ):
                self.launch.main()
            self.assertEqual(
                events,
                ["validate"] + (["runtime"] if enabled == "1" else []) + ["server"],
            )

    def test_dependency_failure_prevents_server_start(self):
        with (
            patch.dict(os.environ, {"SGLANG_GLM53_DEEPEP_PREFILL": "1"}),
            patch.object(self.launch, "validate"),
            patch.object(
                self.launch, "check_runtime", side_effect=ImportError("driver failure")
            ),
            patch.object(self.launch.os, "execv") as execute,
        ):
            with self.assertRaisesRegex(ImportError, "driver failure"):
                self.launch.main()
            execute.assert_not_called()
