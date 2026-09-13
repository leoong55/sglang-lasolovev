import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def test_parts_reconstruct_stable_checksummed_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run1").mkdir()
            (root / "run1" / "result.json").write_text('{"valid":true}\n')
            argv = [
                sys.executable,
                str(ROOT / "export.py"),
                "--root",
                directory,
                "--run-id",
                "run1",
            ]
            manifest = json.loads(
                subprocess.check_output(argv, text=True).split(" ", 1)[1]
            )
            parts = []
            for i, expected in enumerate(manifest["parts"]):
                row = json.loads(
                    subprocess.check_output([*argv, "--part", str(i)], text=True).split(
                        " ", 1
                    )[1]
                )
                raw = base64.b64decode(row["data"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
                parts.append(raw)
            raw = b"".join(parts)
            self.assertEqual(len(raw), manifest["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["sha256"])

    def test_parent_path_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "export.py"),
                    "--root",
                    directory,
                    "--run-id",
                    "..",
                ],
                capture_output=True,
            )
            self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
