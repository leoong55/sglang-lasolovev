import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import launch


class LaunchTest(unittest.TestCase):
    def test_replaces_separate_equals_duplicates_and_list_flags(self):
        incoming = [
            "--model-path",
            "/weights",
            "--moe-a2a-backend=none",
            "--moe-a2a-backend",
            "deepep",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-bs-prefill",
            "1",
            "2",
            "4",
            "--mem-fraction-static",
            "0.85",
        ]
        effective = launch.configure(incoming)
        self.assertEqual(effective.count("--moe-a2a-backend"), 1)
        self.assertIn("/weights", effective)
        self.assertIn("0.85", effective)
        for key, values in launch.SETTINGS.items():
            i = effective.index(key)
            self.assertEqual(effective[i + 1 : i + 1 + len(values)], values)
        self.assertEqual(launch.configure(effective), effective)

    def test_decode_flags_preserved(self):
        args = [
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-bs-decode",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
        ]
        self.assertEqual(launch.configure(args)[: len(args)], args)


if __name__ == "__main__":
    unittest.main()
