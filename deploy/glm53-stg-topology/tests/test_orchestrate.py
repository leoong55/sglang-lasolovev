import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import orchestrate


class RevisionTests(unittest.TestCase):
    def test_independent_tooling_is_allowed_but_runtime_drift_or_uncommitted_code_is_not(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            package = repo / "deploy/glm53-stg-topology"
            package.mkdir(parents=True)

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=repo, stderr=subprocess.DEVNULL, text=True
                ).strip()

            git("init")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            for name in orchestrate.SERVING_CRITICAL_FILES:
                (package / name).write_text("fixed serving\n")
            (package / "benchmark.py").write_text("client v1\n")
            git("add", ".")
            git("commit", "-m", "serving")
            serving = git("rev-parse", "HEAD")
            (package / "benchmark.py").write_text("client v2\n")
            git("add", ".")
            git("commit", "-m", "tooling")
            tooling = git("rev-parse", "HEAD")
            proof = orchestrate.verify_revisions(repo, serving, tooling)
            self.assertEqual(set(proof), set(orchestrate.SERVING_CRITICAL_FILES))
            (package / "benchmark.py").write_text("dirty\n")
            with self.assertRaisesRegex(RuntimeError, "tooling commit"):
                orchestrate.verify_revisions(repo, serving, tooling)
            git("restore", ".")
            (package / "extra.py").write_text("untracked\n")
            with self.assertRaisesRegex(RuntimeError, "Untracked"):
                orchestrate.verify_revisions(repo, serving, tooling)
            (package / "extra.py").unlink()
            (package / "launch.py").write_text("changed runtime\n")
            git("add", ".")
            git("commit", "-m", "runtime drift")
            with self.assertRaisesRegex(RuntimeError, "Serving-critical"):
                orchestrate.verify_revisions(repo, serving, git("rev-parse", "HEAD"))
            with self.assertRaises(ValueError):
                orchestrate.verify_revisions(repo, "--bad", tooling)


class PreparationTests(unittest.TestCase):
    def test_long_prefill_readiness_changes_only_api_probe_and_annotation(self):
        args = self.args(None)
        for profile in ("pp2", "dpa2", "dpa4", "dpa8"):
            for phase in ("baseline", "hicache"):
                expected = orchestrate.render.serving(
                    profile,
                    args.image,
                    args.commit,
                    args.node,
                    hicache=phase == "hicache",
                )
                actual = orchestrate.serving_objects(args, profile, phase)
                template = actual[0]["spec"]["template"]
                container = template["spec"]["containers"][0]
                self.assertEqual(
                    container["startupProbe"]["httpGet"]["path"], "/health"
                )
                self.assertEqual(
                    container["readinessProbe"]["httpGet"]["path"], "/model_info"
                )
                container["readinessProbe"]["httpGet"]["path"] = "/health"
                template["metadata"]["annotations"].pop("glm53-readiness-policy")
                self.assertEqual(actual, expected)

    def test_deleted_deployment_still_waits_for_terminating_gpu_pods(self):
        cluster = object.__new__(orchestrate.Cluster)
        cluster.call = Mock(return_value=SimpleNamespace(stdout=""))
        cluster.get = Mock(
            side_effect=[
                {"items": [{"metadata": {"name": "terminating"}}]},
                {"items": []},
            ]
        )
        with patch.object(orchestrate.time, "sleep") as sleep:
            cluster.stop()
        self.assertEqual(cluster.get.call_count, 2)
        sleep.assert_called_once_with(5)
        cluster.call.assert_called_once()

    def args(self, output):
        return SimpleNamespace(
            campaign="test",
            commit="a" * 40,
            tooling_commit="b" * 40,
            configmap_sha256="c" * 64,
            image="ghcr.io/leoong55/sglang-lasolovev@sha256:" + "d" * 64,
            node="fixture-node",
            skip_preparation=False,
            serving_critical_sha256={"launch.py": "e" * 64},
        )

    def test_jobs_separate_purpose_and_results_without_changing_workloads(self):
        args = self.args(None)
        for purpose, path in [
            ("admission", "/admission"),
            ("preparation", "/preparation"),
            ("measurement", ""),
        ]:
            obj = orchestrate.benchmark_job(
                args,
                "test-pp2-baseline",
                "pp2",
                purpose,
                3 if purpose == "measurement" else 1,
            )
            spec = obj["spec"]["template"]["spec"]
            c = spec["containers"][0]
            env = {x["name"]: x.get("value") for x in c["env"]}
            self.assertEqual(env["TOOLING_COMMIT"], args.tooling_commit)
            self.assertEqual(env["SCRIPT_CONFIG_SHA256"], args.configmap_sha256)
            self.assertEqual(env["BENCHMARK_PURPOSE"], purpose)
            self.assertIn("/results/test-pp2-baseline" + path, c["command"])
            self.assertEqual(
                next(x for x in c["volumeMounts"] if x["name"] == "compile-cache")[
                    "readOnly"
                ],
                True,
            )
            self.assertNotIn("nvidia.com/gpu", c["resources"]["limits"])

    def test_failed_preparation_prevents_measurement_and_releases_gpu(self):
        for fail_preparation in [False, True]:
            with self.subTest(
                fail_preparation=fail_preparation
            ), tempfile.TemporaryDirectory() as folder:
                args = self.args(folder)

                class Cluster:
                    output = Path(folder)

                    def __init__(self):
                        self.jobs, self.stops = [], 0

                    def stop(self):
                        self.stops += 1

                    def apply(self, objects):
                        self.jobs += [
                            x["metadata"]["name"] for x in objects if x["kind"] == "Job"
                        ]

                    def ready(self, evidence):
                        return {"metadata": {"uid": "fixture-pod"}}

                    def wait_job(self, name, evidence):
                        return not (fail_preparation and name.endswith("-prep")), ""

                    def collect(self, evidence):
                        pass

                    def abort_job(self, name):
                        pass

                    def export(self, run_id, node):
                        return self.output / "results" / run_id

                cluster = Cluster()
                with contextlib.redirect_stdout(io.StringIO()):
                    row = orchestrate.run_profile(cluster, args, "pp2", "baseline", 1)
                expected = [
                    "glm53-test-pp2-baseline-smoke",
                    "glm53-test-pp2-baseline-prep",
                ]
                if not fail_preparation:
                    expected.append("glm53-test-pp2-baseline")
                self.assertEqual(cluster.jobs, expected)
                self.assertEqual(cluster.stops, 2)
                self.assertEqual(
                    row["status"], "failed" if fail_preparation else "measured"
                )
                self.assertEqual(row["preparation_job_success"], not fail_preparation)


if __name__ == "__main__":
    unittest.main()
