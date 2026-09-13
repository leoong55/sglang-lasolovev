import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hicache_preflight as gate
import render
import render_hicache_preflight as renderer

CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "num_hidden_layers": 78,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "index_head_dim": 128,
    "quantization_config": {"quant_method": "w4afp8", "group_size": 128},
}
NOW = 1_800_000_000
NODE = "synthetic-node"
IMAGE = "ghcr.io/leoong55/sglang-lasolovev@sha256:" + "a" * 64
COMMIT = "b" * 40
TOOLING = "c" * 40


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def fixture():
    raw = gate.encoded(CONFIG)
    identity = {
        "node_sha256": gate.digest(NODE.encode()),
        "source_commit": COMMIT,
        "serving_image_digest": IMAGE,
        "model_config_sha256": gate.digest(raw),
    }
    baseline = {
        "schema_version": 1,
        "purpose": "hicache_ram_baseline_evidence",
        "status": "verified",
        "helper_sha256": gate.digest(Path(gate.__file__).read_bytes()),
        "required_profiles": ["pp2"],
        **identity,
        "profiles": [
            {
                "profile": "pp2",
                **identity,
                "observed_peak_bytes": 49 * gate.GIB,
                "last_current_bytes": 40 * gate.GIB,
                "sample_count": 500,
                "last_sample_at": NOW - 50,
                "kernel_peak_bytes": None,
                "stages": [
                    {"stage": stage, "sample_count": 10, "max_sample_gap_seconds": 2}
                    for stage in ("admission", "preparation", "measurement")
                ],
            }
        ],
    }
    manifest = gate.encoded(render.serving("pp2", IMAGE, COMMIT, NODE, hicache=True))
    plan = renderer.serving_plan(manifest, NODE, TOOLING, baseline)
    host = gate.host_snapshot(
        NODE, "MemTotal: 2147483648 kB\nMemAvailable: 1073741824 kB\n", NOW - 1
    )
    return baseline, plan, raw, host, manifest


class MemoryArithmetic(unittest.TestCase):
    def test_exact_anchor_indexer_and_extra_page(self):
        expected = {
            "pp2": (1250816, 6439530240, 307523254272),
            "dpa2": (625408, 6439859712, 307525890048),
        }
        for profile, (tokens, indexer, total) in expected.items():
            with self.subTest(profile=profile):
                value = gate.memory_formula(profile, CONFIG)
                self.assertEqual(value["host_tokens_per_rank"], tokens)
                self.assertEqual(value["anchor_bytes_per_rank"], 32000876544)
                self.assertEqual(value["indexer_allocation_bytes_per_rank"], indexer)
                self.assertEqual(value["total_cache_allocation_bytes"], total)
                self.assertEqual(
                    indexer - value["indexer_check_bytes_per_rank"],
                    64 * 132 * value["layers_per_rank"],
                )
        for profile in ("dpa4", "dpa8"):
            self.assertEqual(
                gate.memory_formula(profile, CONFIG),
                gate.memory_formula("dpa2", CONFIG),
            )

    def test_native_floor_matches_exhaustive_valid_interleavings(self):
        # Enumerate all rank sequences for small worlds, with each anchor before
        # that rank's indexer. The check and actual indexer differ deliberately.
        anchor, check, actual = 101, 19, 23
        for ranks in (1, 2, 3, 4):
            peaks = []

            def visit(progress, resident, required):
                if all(p == 2 for p in progress):
                    peaks.append(required)
                    return
                for rank, phase in enumerate(progress):
                    if phase < 2:
                        new = list(progress)
                        new[rank] += 1
                        requested = anchor if phase == 0 else check
                        allocated = anchor if phase == 0 else actual
                        visit(
                            new,
                            resident + allocated,
                            max(required, resident + ranks * requested),
                        )

            visit([0] * ranks, 0, 0)
            formula = max(
                (ranks - 1) * (anchor + actual) + ranks * anchor,
                ranks * anchor + (ranks - 1) * actual + ranks * check,
            )
            self.assertEqual(max(peaks), formula)

    def test_reject_changed_model_layout_or_quantization(self):
        for key, value in (
            ("num_hidden_layers", 79),
            ("index_head_dim", 256),
            ("architectures", ["DifferentModel"]),
            ("quantization_config", {"quant_method": "fp8", "group_size": 128}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.memory_formula("pp2", CONFIG | {key: value})


class GateCriteria(unittest.TestCase):
    def check(self, baseline=None, plan=None, raw=None, host=None):
        values = fixture()[:4]
        return gate.evaluate(
            *(
                old if new is None else new
                for old, new in zip(values, (baseline, plan, raw, host))
            ),
            NOW,
        )

    def test_separate_host_floor_and_cgroup_budget_and_exact_boundary(self):
        good = self.check()
        self.assertEqual(good["status"], "PASS")
        self.assertEqual(good["baseline_budget_bytes"], 49 * gate.GIB)
        self.assertGreater(
            good["conservative_native_admission"]["required_bytes"],
            good["configured_cgroup_budget"]["required_bytes"],
        )
        host = fixture()[3]
        host["mem_available_bytes"] = good["conservative_native_admission"][
            "required_bytes"
        ]
        self.assertEqual(self.check(host=host)["status"], "PASS")
        host["mem_available_bytes"] -= 1
        blocked = self.check(host=host)
        self.assertEqual(
            blocked["reasons"], ["conservative_native_host_floor_insufficient"]
        )
        self.assertGreater(blocked["configured_cgroup_budget"]["headroom_bytes"], 0)

    def test_cgroup_can_fail_when_host_has_capacity(self):
        baseline = fixture()[0]
        baseline["profiles"][0]["observed_peak_bytes"] = 350 * gate.GIB
        result = self.check(baseline=baseline)
        self.assertEqual(
            result["reasons"], ["configured_serving_cgroup_budget_insufficient"]
        )
        self.assertGreater(result["conservative_native_admission"]["headroom_bytes"], 0)

    def test_optional_kernel_peak_increases_budget_never_replaces_observed_label(self):
        baseline = fixture()[0]
        baseline["profiles"][0]["kernel_peak_bytes"] = 60 * gate.GIB
        result = self.check(baseline=baseline)
        self.assertEqual(result["observed_baseline_peak_bytes"], 49 * gate.GIB)
        self.assertEqual(result["baseline_budget_bytes"], 60 * gate.GIB)
        baseline["profiles"][0]["observed_peak_bytes"] = None
        self.assertEqual(self.check(baseline=baseline)["status"], "BLOCKED")

    def test_missing_stale_or_asymmetric_identity_fails_closed(self):
        mutations = [
            (0, lambda x: x["profiles"][0].pop("observed_peak_bytes")),
            (0, lambda x: x["profiles"][0].update(last_sample_at=NOW + 1)),
            (0, lambda x: x["profiles"][0].update(stages=[])),
            (0, lambda x: x.update(required_profiles=["pp2", "dpa2"])),
            (0, lambda x: x["profiles"][0].update(source_commit="d" * 40)),
            (1, lambda x: x.update(serving_limit_bytes=1024 * gate.GIB)),
            (1, lambda x: x.update(upstream_source_sha256={})),
            (1, lambda x: x.update(operational_margin_bytes=0)),
            (1, lambda x: x.update(serving_image_digest=IMAGE[:-1] + "f")),
            (3, lambda x: x.update(node_sha256="d" * 64)),
            (3, lambda x: x.update(sampled_at=NOW - 121)),
            (3, lambda x: x.update(sampled_at=NOW + 1)),
            (3, lambda x: x.pop("mem_available_bytes")),
        ]
        for index, change in mutations:
            values = list(fixture()[:4])
            change(values[index])
            with self.subTest(index=index, mutation=change):
                self.assertEqual(gate.evaluate(*values, NOW)["status"], "BLOCKED")
        self.assertEqual(self.check(raw=b"{}")["status"], "BLOCKED")

    def test_report_expiration_manifest_and_node_binding(self):
        report, manifest = self.check(), fixture()[4]
        gate.verify_report(report, manifest, NODE, NOW + 1)
        for raw, node, now in (
            (manifest + b" ", NODE, NOW),
            (manifest, "another-node", NOW),
            (manifest, NODE, NOW + 121),
        ):
            with self.assertRaises(ValueError):
                gate.verify_report(report, raw, node, now)

    def test_render_is_immutable_cpu_only_and_read_only(self):
        baseline, _, _, _, manifest = fixture()
        cm, job = renderer.build(
            "hicache-preflight-test", NODE, TOOLING, baseline, manifest
        )["items"]
        self.assertTrue(cm["immutable"])
        self.assertEqual(
            set(cm["data"]), {"hicache_preflight.py", "baseline.json", "plan.json"}
        )
        pod = job["spec"]["template"]["spec"]
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual({v["name"] for v in pod["volumes"]}, {"model", "scripts"})
        self.assertTrue(
            all(v["readOnly"] for v in pod["containers"][0]["volumeMounts"])
        )
        self.assertEqual(
            pod["containers"][0]["resources"], render.resources(1, "256Mi")
        )
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        self.assertNotIn(NODE, cm["data"]["baseline.json"])
        env = {e["name"]: e for e in pod["containers"][0]["env"]}
        self.assertEqual(
            env["SCRIPT_CONFIG_SHA256"]["value"], gate.digest(gate.encoded(cm["data"]))
        )
        self.assertEqual(
            env["NODE_NAME"]["valueFrom"]["fieldRef"]["fieldPath"], "spec.nodeName"
        )
        with self.assertRaises(ValueError):
            renderer.build(
                "hicache-preflight-test", "another-node", TOOLING, baseline, manifest
            )

    def test_source_verification_rejects_dirty_snapshot(self):
        with mock.patch.object(
            renderer.subprocess,
            "run",
            return_value=mock.Mock(returncode=0, stdout=b"unreviewed"),
        ):
            with self.assertRaisesRegex(ValueError, "tooling_source_snapshot_mismatch"):
                renderer.verify_snapshot(TOOLING)

    def test_image_installer_must_verify_every_allocator_input(self):
        manifest = {
            "base_commit": gate.UPSTREAM_COMMIT,
            "files": [
                {"path": "srt/mem_cache/" + key, "sha256": value}
                for key, value in gate.SOURCE_SHA256.items()
            ],
        }
        with mock.patch.object(renderer.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=gate.encoded(manifest))
            renderer.verify_image_source_contract(fixture()[1])
            manifest["files"].pop()
            run.return_value.stdout = gate.encoded(manifest)
            with self.assertRaisesRegex(ValueError, "does_not_verify_allocator_inputs"):
                renderer.verify_image_source_contract(fixture()[1])


class ObservedBaseline(unittest.TestCase):
    def test_max_across_stages_replays_and_optional_kernel_counter(self):
        samples = []
        for t in range(0, 101, 2):
            row = {
                "timestamp": NOW + t,
                "cgroup_memory_bytes": (99 if t == 44 else 49) * gate.GIB,
                "rows": ["SYNTHETIC_GPU_ID"],
            }
            samples.append(f"{iso(NOW + t)} GLM53_GPU {json.dumps(row)}\n")
        stages = [
            {"stage": name, "started_at": NOW + a, "finished_at": NOW + b}
            for name, a, b in (
                ("admission", 10, 20),
                ("preparation", 30, 50),
                ("measurement", 60, 80),
            )
        ]
        result = gate.summarize_samples(["".join(samples), "".join(samples)], stages)
        self.assertEqual(result["observed_peak_bytes"], 99 * gate.GIB)
        self.assertEqual(result["last_current_bytes"], 49 * gate.GIB)
        self.assertEqual(result["sample_count"], 51)
        self.assertIsNone(result["kernel_peak_bytes"])
        self.assertNotIn("SYNTHETIC_GPU_ID", json.dumps(result))
        with self.assertRaisesRegex(ValueError, "telemetry_gap"):
            gate.summarize_samples([samples[0] + samples[-1]], stages)
        missing = json.loads(samples[0].split("GLM53_GPU ")[1])
        missing["cgroup_memory_bytes"] = None
        with self.assertRaisesRegex(ValueError, "not_observable"):
            gate.summarize_samples(
                [f"{iso(NOW)} GLM53_GPU {json.dumps(missing)}\n"], stages
            )

    def test_real_follow_verifier_and_artifact_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence, results = (
                root / "synthetic-pp2-baseline",
                root / "results" / "synthetic-pp2-baseline",
            )
            evidence.mkdir()
            results.mkdir(parents=True)

            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = gate.encoded(value)
                path.write_bytes(raw)
                return gate.digest(raw)

            uid, container_id = "synthetic-pod", "synthetic-container"
            state = {
                "run_id": evidence.name,
                "profile": "pp2",
                "phase": "baseline",
                "status": "measured",
                "job_success": True,
                "preparation_job_success": True,
                "source_commit": COMMIT,
                "image": IMAGE,
                "node": NODE,
                "pod_uid": uid,
            }
            save(evidence / "run-state.json", state)
            spec = render.serving("pp2", IMAGE, COMMIT, NODE)[0]["spec"]["template"][
                "spec"
            ]
            spec["nodeName"] = NODE
            save(
                evidence / "glm53-topology-serving-synthetic.json",
                {
                    "kind": "Pod",
                    "metadata": {"uid": uid},
                    "spec": spec,
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "sglang",
                                "restartCount": 0,
                                "containerID": container_id,
                                "imageID": IMAGE,
                            }
                        ]
                    },
                },
            )
            raw = b"".join(
                f'{iso(NOW + t)} GLM53_GPU {json.dumps({"timestamp": NOW + t, "cgroup_memory_bytes": (80 if t == 42 else 49) * gate.GIB})}\n'.encode()
                for t in range(0, 101, 2)
            )
            (evidence / "server-follow.log").write_bytes(raw)
            source = (ROOT / "collect_follow_logs.py").read_bytes()
            (evidence / "server-follow.collector.py").write_bytes(source)
            metadata = {
                "schema_version": 1,
                "format": "kubectl-follow-timestamps-lf-v1",
                "capture_mode": "continuous_follow",
                "container": "sglang",
                "collector_sha256": gate.digest(source),
                "pod_uid": uid,
                "container_id": container_id,
                "restart_count": 0,
                "source_commit": COMMIT,
                "image_id": IMAGE,
                "log_sha256": gate.digest(raw),
                "log_bytes": len(raw),
                "collection_errors": 0,
                "segments": [
                    {
                        "segment_id": 0,
                        "byte_start": 0,
                        "byte_end": len(raw),
                        "first_cri_timestamp": NOW,
                        "last_cri_timestamp": NOW + 100,
                        "started_at": NOW,
                        "ended_at": NOW + 101,
                        "records": 51,
                        "unframed_records": 0,
                        "partial_line_bytes": 0,
                        "clock_regressions": 0,
                        "container_id": container_id,
                        "restart_count": 0,
                        "interruption": None,
                    }
                ],
            }
            save(evidence / "server-follow.meta.json", metadata)
            for stage, suffix, subdir, start, end in (
                ("admission", "-smoke", "admission", 10, 20),
                ("preparation", "-prep", "preparation", 30, 50),
                ("measurement", "", "", 60, 80),
            ):
                save(
                    evidence / f"glm53-{evidence.name}{suffix}.json",
                    {
                        "kind": "Job",
                        "status": {
                            "conditions": [{"type": "Complete", "status": "True"}],
                            "startTime": iso(NOW + start),
                            "completionTime": iso(NOW + end),
                        },
                    },
                )
                provenance = {
                    "purpose": stage,
                    "profile": "pp2",
                    "source_commit": COMMIT,
                    "serving_image_digest": IMAGE,
                }
                if stage != "admission":
                    provenance["dataset_catalog_sha256"] = save(
                        results / subdir / "dataset-catalog.json",
                        {
                            "model_metadata": {
                                "files": {
                                    "config.json": gate.digest(gate.encoded(CONFIG))
                                }
                            }
                        },
                    )
                save(results / subdir / "provenance.json", provenance)
            summary = gate.summarize([root], ["pp2"])
            self.assertEqual(
                summary["profiles"][0]["observed_peak_bytes"], 80 * gate.GIB
            )
            serialized = json.dumps(summary)
            for private in (NODE, uid, container_id, str(root)):
                self.assertNotIn(private, serialized)
            # A reconnect after the pod disappeared produces a closed empty
            # segment. It must not invalidate an earlier complete capture.
            original = copy.deepcopy(metadata)
            tail = dict(metadata["segments"][0])
            tail.update(
                segment_id=1,
                byte_start=len(raw),
                byte_end=len(raw),
                first_cri_timestamp=None,
                last_cri_timestamp=None,
                started_at=NOW + 102,
                ended_at=NOW + 103,
                records=0,
                interruption="stream_ended_after_pod_removal",
            )
            metadata["segments"].append(tail)
            save(evidence / "server-follow.meta.json", metadata)
            repeated = gate.summarize([root], ["pp2"])
            self.assertEqual(repeated["profiles"][0]["selected_follow_segments"], [0])
            self.assertEqual(
                repeated["profiles"][0]["observed_peak_bytes"], 80 * gate.GIB
            )

            # Two individually valid segments cannot be joined to establish
            # continuous coverage of the preparation window crossing them.
            metadata = copy.deepcopy(original)
            boundary = len(b"\n".join(raw.split(b"\n")[:25]) + b"\n")
            first = metadata["segments"][0]
            first.update(
                byte_end=boundary,
                last_cri_timestamp=NOW + 48,
                ended_at=NOW + 49,
                records=25,
            )
            second = dict(original["segments"][0])
            second.update(
                segment_id=1,
                byte_start=boundary,
                first_cri_timestamp=NOW + 50,
                started_at=NOW + 50,
                records=26,
            )
            metadata["segments"].append(second)
            save(evidence / "server-follow.meta.json", metadata)
            with self.assertRaisesRegex(ValueError, "stage_stream_not_continuous"):
                gate.summarize([root], ["pp2"])
            save(evidence / "server-follow.meta.json", original)
            (evidence / "server-follow.log").write_bytes(raw + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "log_not_verified"):
                gate.summarize([root], ["pp2"])


if __name__ == "__main__":
    unittest.main()
