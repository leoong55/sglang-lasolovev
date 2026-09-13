import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import guard_campaign as guard
import render_campaign_watch
import watch_campaign


class ScreeningTests(unittest.TestCase):
    def row(self, **changes):
        return {
            "stage": "preparation/r01-long-cold",
            "first_measured_started_at": 100,
            "last_measured_finished_at": 900,
            "completed_measured": 300,
            "error_measured": 0,
            "malformed_records": 0,
            "verdict_present": True,
        } | changes

    def test_completed_fast_workload_does_not_age_into_a_timeout(self):
        self.assertIsNone(guard.budget_decision(self.row(), 3000, 900))
        self.assertIsNotNone(
            guard.budget_decision(self.row(verdict_present=False), 1001, 900)
        )

    def test_unusable_evidence_and_short_tests_do_not_trigger_long_budget(self):
        for changes in (
            {"first_measured_started_at": None},
            {"first_measured_started_at": 2000},
            {"malformed_records": 1},
            {"stage": "preparation/r01-short"},
        ):
            self.assertIsNone(guard.budget_decision(self.row(**changes), 3000, 900))

    def test_zero_completion_fallback_includes_grace_and_stops_at_finished_event(self):
        event = {
            "event": "benchmark_started",
            "kind": "long-cold",
            "directory": "/results/example-pp2-baseline/preparation/r01-long-cold",
        }
        log = "1970-01-01T00:00:00Z " + json.dumps(event) + "\n"
        self.assertIsNone(guard.no_completion_decision(log, [], 1080, 1080))
        self.assertIsNotNone(guard.no_completion_decision(log, [], 1081, 1080))
        self.assertIsNone(guard.no_completion_decision(log, [self.row()], 2000, 1080))
        log += (
            "1970-01-01T00:10:00Z "
            + json.dumps(
                {"event": "benchmark_finished", "directory": event["directory"]}
            )
            + "\n"
        )
        self.assertIsNone(guard.no_completion_decision(log, [], 2000, 1080))

    def test_progress_uses_measured_records_and_retains_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "example-pp2-baseline/preparation/r01-long-cold"
            directory.mkdir(parents=True)
            rows = [
                {"measured": False, "started_at": 1, "finished_at": 2},
                {
                    "measured": True,
                    "started_at": 10,
                    "finished_at": 20,
                    "status": 200,
                    "sse_done": True,
                },
                {
                    "measured": True,
                    "started_at": 11,
                    "finished_at": 21,
                    "status": 500,
                    "sse_done": False,
                },
            ]
            (directory / "requests.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            report = watch_campaign.snapshot(root, "example")["workloads"][0]
            self.assertEqual(report["first_measured_started_at"], 10)
            self.assertEqual(report["last_measured_finished_at"], 21)
            self.assertEqual(report["completed_measured"], 1)
            self.assertEqual(report["error_measured"], 1)

    def test_watcher_has_no_model_gpu_or_service_account_credentials(self):
        obj = render_campaign_watch.objects("example", "test-node")
        spec = obj["items"][1]["spec"]["template"]["spec"]
        self.assertFalse(spec["automountServiceAccountToken"])
        self.assertNotIn("model", {v["name"] for v in spec["volumes"]})
        container = spec["containers"][0]
        self.assertTrue(all(v["readOnly"] for v in container["volumeMounts"]))
        self.assertNotIn("nvidia.com/gpu", container["resources"]["limits"])

    def test_cleanup_requires_owned_terminal_or_suspended_idle_job(self):
        tree = ast.parse((ROOT / "cleanup_completed_jobs.py").read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "eligible"
        )
        namespace = {"owner": "glm53-stg-pp2-dpa"}
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), "cleanup_policy", "exec"
            ),
            namespace,
        )
        eligible = namespace["eligible"]
        job = {
            "metadata": {"labels": {"app.kubernetes.io/part-of": "glm53-stg-pp2-dpa"}},
            "spec": {},
            "status": {"conditions": [{"status": "True", "type": "Complete"}]},
        }
        self.assertTrue(eligible(job))
        self.assertFalse(eligible(job | {"status": {"active": 1}}))
        self.assertFalse(eligible(job | {"metadata": {"labels": {}}}))
        self.assertTrue(
            eligible(job | {"spec": {"suspend": True}, "status": {"active": 0}})
        )
        self.assertFalse(eligible(job | {"status": {}}))


if __name__ == "__main__":
    unittest.main()
