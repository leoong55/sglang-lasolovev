import importlib.util
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
spec = importlib.util.spec_from_file_location("v6_launch", KIT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchTest(unittest.TestCase):
    def valid(self):
        return [v for key, values in launch.SETTINGS.items() for v in [key, *values]]

    def test_manifest_is_preserved(self):
        args = self.valid() + [
            "--model-path",
            "/weights",
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
        self.assertEqual(launch.configure(args), args)

    def test_rejects_conflicting_or_missing_flags(self):
        for args in (
            [],
            self.valid() + ["--moe-a2a-backend", "none"],
            [x if x != "deepep" else "none" for x in self.valid()],
        ):
            with self.assertRaises(ValueError):
                launch.configure(args)

    def test_equals_syntax_is_accepted_without_rewriting(self):
        args = [key + "=" + values[0] for key, values in launch.SETTINGS.items()]
        self.assertEqual(launch.configure(args), args)


if __name__ == "__main__":
    unittest.main()
