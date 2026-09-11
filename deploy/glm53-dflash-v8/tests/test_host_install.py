import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestInstaller(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "installer_tested", Path(__file__).resolve().parents[1] / "install.py"
        )
        self.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.installer)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sglang"
        self.bundle = Path(self.temp.name) / "bundle"
        self.original = b"VALUE = 'original'\n"
        self.updated = b"VALUE = 'updated'\n"
        self.added = b"VALUE = 'new'\n"
        (self.root / "srt").mkdir(parents=True)
        (self.root / "srt/existing.py").write_bytes(self.original)
        rows = []
        for name, before, after in (
            ("added.py", None, self.added),
            ("existing.py", self.original, self.updated),
        ):
            path = f"python/sglang/srt/{name}"
            target = self.bundle / "overlay" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(after)
            rows.append(
                dict(
                    path=path,
                    base_sha256=self.installer.digest(before) if before else None,
                    patched_sha256=self.installer.digest(after),
                )
            )
        (self.bundle / "base-files.json").write_text(
            json.dumps(dict(base_commit="base", files=rows))
        )

    def test_new_file_install_verify_and_idempotence(self):
        for verify in (False, True, False):
            self.installer.install(self.root, self.bundle, verify_only=verify)
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), self.updated)
        self.assertEqual((self.root / "srt/added.py").read_bytes(), self.added)

    def test_mismatch_validates_all_before_any_mutation(self):
        (self.root / "srt/existing.py").write_bytes(b"other = 1\n")
        with self.assertRaisesRegex(RuntimeError, "Source mismatch"):
            self.installer.install(self.root, self.bundle)
        self.assertFalse((self.root / "srt/added.py").exists())
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), b"other = 1\n")

    def test_failed_compile_rolls_back_added_and_existing_files(self):
        with patch.object(
            self.installer.py_compile,
            "compile",
            side_effect=[None, RuntimeError("failure")],
        ):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                self.installer.install(self.root, self.bundle)
        self.assertFalse((self.root / "srt/added.py").exists())
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), self.original)

    def test_verify_only_never_installs(self):
        with self.assertRaisesRegex(RuntimeError, "Source mismatch"):
            self.installer.install(self.root, self.bundle, verify_only=True)
        self.assertFalse((self.root / "srt/added.py").exists())
        self.assertEqual((self.root / "srt/existing.py").read_bytes(), self.original)
