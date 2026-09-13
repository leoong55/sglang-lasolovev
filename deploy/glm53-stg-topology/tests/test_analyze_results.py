import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyze_results", Path(__file__).parents[1] / "analyze_results.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def linefile(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(value) + "\n" for value in values))

    def pp(self, timestamp, n=40, progress=True, **changes):
        ids = ["secret-request-" + str(index) for index in range(n)]
        return {
            "timestamp": timestamp,
            "valid": True,
            "stage": 0,
            "tp": 0,
            "rids": ids,
            "output_lengths": {rid: timestamp if progress else 0 for rid in ids},
        } | changes

    def pp_log(self, records):
        return "\n".join(
            "2026-09-13T00:00:00Z GLM53_PP_ACTIVITY " + json.dumps(row)
            for row in records
        )

    def test_pp_sorted_unique_stage_and_actual_progress(self):
        records = [self.pp(timestamp) for timestamp in range(1, 34)]
        records += [self.pp(timestamp, 100, stage=1) for timestamp in range(1, 34)]
        records.append(records[0])
        points, windows, duration = analysis.pp_evidence(
            self.pp_log(reversed(records)), 1, 33
        )
        self.assertEqual(max(value for _, value in points), 40)
        self.assertEqual(analysis.longest_window(points), 32)
        self.assertEqual(windows, 32)
        self.assertEqual(duration, 32)

    def test_pp_admission_is_not_decode_progress_and_invalid_breaks_window(self):
        points, windows, duration = analysis.pp_evidence(
            self.pp_log([self.pp(i, progress=False) for i in range(35)]), 0, 34
        )
        self.assertEqual(analysis.longest_window(points), 34)
        self.assertEqual((windows, duration), (0, 0))
        records = [self.pp(i, valid=i != 17) for i in range(35)]
        points, _, duration = analysis.pp_evidence(self.pp_log(records), 0, 34)
        self.assertEqual(analysis.longest_window(points), 16)
        self.assertEqual(duration, 16)

    def test_pp_foreign_length_ids_and_gaps_never_prove_progress(self):
        records = [
            self.pp(i, n=1, output_lengths={str(k): i for k in range(40)})
            for i in range(35)
        ]
        _, windows, duration = analysis.pp_evidence(self.pp_log(records), 0, 34)
        self.assertEqual((windows, duration), (0, 0))
        self.assertEqual(analysis.longest_window([(0, 40), (4, 40), (8, 40)]), 0)
        self.assertEqual(analysis.longest_window([(0, 40), (0, None), (1, 40)]), 0)

    def test_pp_health_and_foreign_ids_cannot_fill_measured_c40(self):
        measured = {"secret-request-" + str(index) for index in range(39)}
        rows = []
        for timestamp in range(35):
            row = self.pp(timestamp, n=39)
            row["rids"] += ["HEALTH_CHECK_probe", "foreign-request"]
            row["output_lengths"].update(
                HEALTH_CHECK_probe=timestamp, **{"foreign-request": timestamp}
            )
            rows.append(row)
        points, windows, duration = analysis.pp_evidence(
            self.pp_log(rows), 0, 34, measured
        )
        self.assertEqual(max(value for _, value in points), 39)
        self.assertEqual((windows, duration), (0, 0))
        health_only = self.pp(0, n=39)
        health_only["rids"].append("HEALTH_CHECK_probe")
        health_only["output_lengths"]["HEALTH_CHECK_probe"] = 0
        diagnostic, _, _ = analysis.pp_evidence(self.pp_log([health_only]), 0, 0)
        self.assertEqual(diagnostic, [(0, 39)])

    def test_pp_allowlist_applies_before_progress_and_respects_time_bounds(self):
        measured = {"secret-request-" + str(index) for index in range(40)}
        rows = []
        for timestamp in range(50):
            row = self.pp(timestamp, n=40)
            row["rids"] += ["HEALTH_CHECK_probe", "foreign-request"]
            # Foreign metadata cannot inflate progress or invalidate measured IDs.
            row["output_lengths"].update(
                HEALTH_CHECK_probe=timestamp * 100, **{"foreign-request": -1}
            )
            rows.append(row)
        points, windows, duration = analysis.pp_evidence(
            self.pp_log(rows), 10, 40, measured
        )
        self.assertEqual(max(value for _, value in points), 40)
        self.assertEqual((windows, duration), (30, 30))
        self.assertEqual((points[0][0], points[-1][0]), (10, 40))

    def dpa(self, timestamp, profile="dpa2", count=40, server_time=None):
        size = int(profile[3:])
        return {
            "timestamp": timestamp,
            "activity": {"observable": True, "running": 400},
            "loads": {
                "loads": [
                    {
                        "dp_rank": rank,
                        "num_running_reqs": count // size,
                        "timestamp": timestamp if server_time is None else server_time,
                    }
                    for rank in range(size)
                ]
            },
        }

    def test_dpa_recomputes_groups_and_deduplicates_cached_polls(self):
        path = self.root / "telemetry.jsonl"
        rows = [self.dpa(i, server_time=i // 2 * 2) for i in range(35)]
        self.linefile(path, rows)
        points, evidence = analysis.dpa_evidence(path, 0, 34, "dpa2")
        self.assertEqual(len(points), 18)
        self.assertEqual(max(value for _, value in points), 40)
        self.assertEqual(analysis.longest_window(points), 34)
        self.assertEqual(evidence["cached_or_partially_advanced_polls"], 17)
        self.assertEqual(
            evidence["distinct_group_snapshot_interval_seconds"]["median"], 2
        )

    def test_dpa_stale_duplicate_missing_and_slow_snapshot_evidence(self):
        path = self.root / "telemetry.jsonl"
        rows = [self.dpa(i, server_time=i // 15 * 15) for i in range(46)]
        self.linefile(path, rows)
        points, evidence = analysis.dpa_evidence(path, 0, 45, "dpa2")
        self.assertEqual(analysis.longest_window(points), 0)
        self.assertEqual(
            evidence["distinct_group_snapshot_interval_seconds"]["median"], 15
        )
        broken = self.dpa(1)
        broken["loads"]["loads"][1]["dp_rank"] = 0
        self.linefile(
            path,
            [broken, {"timestamp": 2, "activity": {"observable": True, "running": 40}}],
        )
        points, _ = analysis.dpa_evidence(path, 0, 45, "dpa2")
        self.assertEqual(points, [(1, None), (2, None)])

    def campaign(
        self, profile="dpa2", phase="baseline", repetition=1, throughput=500, cache=True
    ):
        run_id = f"campaign-{profile}-{phase}"
        state = {
            "run_id": run_id,
            "profile": profile,
            "phase": phase,
            "repetitions": 1,
            "status": "measured",
            "purpose": "measurement",
            "measurement_job_started": True,
            "source_commit": "a" * 40,
            "tooling_commit": "c" * 40,
            "configmap_sha256": "d" * 64,
            "image": "private-host.example/image@sha256:" + "b" * 64,
            "node": "private-host-192.0.2.123",
            "pod_uid": "private-pod-uuid",
            "error": "credential-secret-host",
        }
        self.write(self.root / run_id / "run-state.json", state)
        (self.root / run_id / "server.log").write_text(
            "1970-01-01T00:00:01Z server observed\n"
        )
        self.write(
            self.root / "results" / run_id / "provenance.json",
            {
                "purpose": "measurement",
                "timestamp": -1,
                "source_commit": state["source_commit"],
                "serving_image_digest": state["image"],
                "tooling_commit": state["tooling_commit"],
                "configmap_sha256": state["configmap_sha256"],
            },
        )
        for kind in ("long-cold", "long-warm"):
            result = self.root / "results" / run_id / f"r{repetition:02d}-{kind}"
            records = []
            for index in range(300):
                records.append(
                    {
                        "measured": True,
                        "request_id": f"private-request-{index}",
                        "status": 200,
                        "sse_done": True,
                        "finish_reasons": {"0": "length"},
                        "completion_tokens": 1000,
                        "usage": {"completion_tokens": 1000}
                        | (
                            {"prompt_tokens_details": {"cached_tokens": 0}}
                            if cache
                            else {}
                        ),
                        "evidence_errors": [],
                        "started_at": 0,
                        "finished_at": 40,
                        "request_sha256": f"{index:064x}",
                        "output_preview": "PRIVATE PROMPT PAYLOAD",
                        "sampling": {
                            "max_completion_tokens": 1000,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                            "ignore_eos": True,
                            "temperature": 0.3,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    }
                )
            self.linefile(result / "requests.jsonl", records)
            verdict = {
                "workload": kind,
                "functional_valid": True,
                "exit_code": 0,
                "expected_requests": 300,
                "observed_requests": 300,
                "started_at": 0,
                "finished_at": 40,
                "cache_observability": {"reported": 300},
                "benchmark_metrics": {"api_key": "SECRET"},
            }
            self.write(result / "verdict.json", verdict)
            self.write(
                result / "vllm.json",
                {
                    "completed": 300,
                    "failed": 0,
                    "output_throughput": throughput,
                    "median_ttft_ms": 123,
                    "p99_tpot_ms": 20,
                    "pod_uid": "private-pod-uuid",
                    "tokenizer": "http://192.0.2.123/model",
                    "api_key": "credential-secret-host",
                    "GPU-PRIVACY-TEST-CANARY": 123,
                    "generated_texts": ["PRIVATE PROMPT PAYLOAD"],
                    "mean_itl_ms": float("nan"),
                },
            )
            self.linefile(
                result / "telemetry.jsonl", [self.dpa(i, profile) for i in range(41)]
            )
            self.write(
                result / "before-server-info.json",
                {"load_snapshot_publish_interval": 15},
            )
            for point, start, end in (("before", -2, -1), ("after", 41, 42)):
                self.write(
                    result / f"{point}-compile-cache.json",
                    {
                        "observable": True,
                        "status": "observed",
                        "files": [],
                        "errors": [],
                        "started_at": start,
                        "finished_at": end,
                    },
                )
        return run_id

    def test_real_artifact_shapes_privacy_cache_and_no_measured_selection(self):
        empty = analysis.analyze([self.root])
        self.assertEqual(empty["dpa_selection"]["status"], "NO_MEASUREMENTS")
        self.campaign(cache=False)
        output = analysis.analyze([self.root])
        self.assertEqual(len(output["runs"]), 2)
        self.assertTrue(output["runs"][0]["functional_valid"])
        self.assertTrue(output["runs"][0]["capacity_qualified"])
        self.assertEqual(
            output["runs"][0]["cache_observability"], {"not_observable": 300}
        )
        self.assertEqual(
            output["runs"][0]["dp_request_prefix_affinity"], "not_observable"
        )
        self.assertIsNone(output["dpa_selection"]["screening_candidate"])
        serialized = json.dumps(output, allow_nan=False)
        for private in (
            "private-host",
            "private-pod",
            "private-request",
            "PRIVATE PROMPT",
            "192.0.2.123",
            "credential-secret",
            "GPU-PRIVACY-TEST",
            "api_key",
            "tokenizer",
            "SECRET",
        ):
            self.assertNotIn(private, serialized)

    def test_baseline_repeat_merge_invalid_results_excluded_and_selection(self):
        for profile, speed in [("dpa2", 500), ("dpa4", 600), ("dpa8", 550)]:
            self.campaign(profile=profile, throughput=speed)
        self.campaign(profile="dpa4", phase="repeat", throughput=610)
        output = analysis.analyze([self.root, self.root])
        self.assertEqual(len(output["runs"]), 8)
        group = next(
            g
            for g in output["aggregates"]
            if g["profile"] == "dpa4" and g["workload"] == "long-cold"
        )
        self.assertEqual(group["repetitions"], 2)
        self.assertEqual(group["metrics"]["output_throughput"]["median"], 605)
        self.assertEqual(output["dpa_selection"]["screening_candidate"], "dpa4")
        self.assertFalse(output["dpa_selection"]["final_recommendation_ready"])
        target = self.root / "results/campaign-dpa4-repeat/r01-long-cold/vllm.json"
        raw = json.loads(target.read_text())
        raw["failed"] = 1
        self.write(target, raw)
        output = analysis.analyze([self.root])
        group = next(
            g
            for g in output["aggregates"]
            if g["profile"] == "dpa4" and g["workload"] == "long-cold"
        )
        self.assertEqual(group["metrics"]["output_throughput"]["median"], 600)
        self.assertEqual(group["valid_repetitions"], 1)
        self.assertFalse(group["all_functional_valid"])
        self.assertEqual(output["dpa_selection"]["screening_candidate"], "dpa8")

    def test_bad_counts_timestamps_and_sampling_invalidate_successful_verdict(self):
        run = self.campaign()
        path = self.root / "results" / run / "r01-long-cold/requests.jsonl"
        records = list(analysis.json_lines(path))
        records[0]["finish_reasons"] = {"0": "stop"}
        records[1]["sampling"]["stream_options"] = None
        records[2]["finish_reasons"] = {"0": {"secret": "PRIVATE"}}
        self.linefile(path, records)
        output = analysis.analyze([self.root])
        row = next(row for row in output["runs"] if row["workload"] == "long-cold")
        self.assertFalse(row["functional_valid"])
        self.assertEqual(row["invalid_requests"], 3)
        self.assertNotIn("PRIVATE", json.dumps(output))

    def test_pp_verdict_requires_complete_unique_measured_response_ids(self):
        run = self.campaign()
        state_path = self.root / run / "run-state.json"
        state = json.loads(state_path.read_text())
        state["profile"] = "pp2"
        self.write(state_path, state)
        for kind in ("long-cold", "long-warm"):
            path = self.root / "results" / run / f"r01-{kind}/requests.jsonl"
            records = list(analysis.json_lines(path))
            for index, record in enumerate(records):
                record["response_id"] = f"secret-request-{index}"
            # A simultaneous non-measured request must not enter the allowlist.
            records.append(
                records[0] | {"measured": False, "response_id": "warmup-rid"}
            )
            self.linefile(path, records)
        log = self.pp_log([self.pp(timestamp) for timestamp in range(41)])
        (self.root / run / "server.log").write_text(log)
        result = analysis.analyze([self.root])
        self.assertTrue(all(row["capacity_qualified"] for row in result["runs"]))
        path = self.root / "results" / run / "r01-long-cold/requests.jsonl"
        records = list(analysis.json_lines(path))
        records[299]["response_id"] = records[298]["response_id"]
        self.linefile(path, records)
        result = analysis.analyze([self.root])
        cold = next(row for row in result["runs"] if row["workload"] == "long-cold")
        self.assertEqual(cold["server_running_peak"], 40)
        self.assertEqual(
            cold["pp_measurement_identity"]["unique_measured_response_ids"], 299
        )
        self.assertEqual(cold["pp_measurement_identity"]["status"], "not_observable")
        self.assertFalse(cold["capacity_qualified"])
        self.assertFalse(cold["server_c40_observed"])
        self.assertFalse(cold["c40_sustained_30s"])
        self.assertFalse(cold["pp_progress_sustained_30s"])
        self.assertNotIn("secret-request", json.dumps(result))

    def pp_campaign(self, observations):
        run = self.campaign()
        state_path = self.root / run / "run-state.json"
        state = json.loads(state_path.read_text())
        state["profile"] = "pp2"
        self.write(state_path, state)
        for kind in ("long-cold", "long-warm"):
            path = self.root / "results" / run / f"r01-{kind}/requests.jsonl"
            records = list(analysis.json_lines(path))
            for index, record in enumerate(records):
                record["response_id"] = f"secret-request-{index}"
            self.linefile(path, records)
        (self.root / run / "server.log").write_text(self.pp_log(observations))

    def test_pp_turnover_preserves_c40_without_requiring_40_retained_decoders(self):
        observations = []
        for timestamp in range(41):
            ids = [
                f"secret-request-{index}" for index in range(timestamp, timestamp + 40)
            ]
            observations.append(
                self.pp(
                    timestamp, rids=ids, output_lengths={rid: timestamp for rid in ids}
                )
            )
        self.pp_campaign(observations)
        output = analysis.analyze([self.root])
        self.assertEqual(output["schema_version"], 4)
        for row in output["runs"]:
            self.assertTrue(row["functional_valid"])
            self.assertTrue(row["server_c40_observed"])
            self.assertTrue(row["capacity_qualified"])
            self.assertEqual(row["server_running_peak"], 40)
            self.assertEqual(row["server_c40_sample_fraction"], 1)
            self.assertTrue(row["c40_sustained_30s"])
            self.assertFalse(row["pp_progress_sustained_30s"])
            detail = row["pp_turnover_and_progress"]
            self.assertEqual(detail["c40_windows_with_active_membership_change"], 40)
            self.assertEqual(detail["advancing_common_requests"]["min"], 39)
            self.assertEqual(detail["windows_with_observed_decode_progress"], 40)
            self.assertEqual(detail["decode_progress_status"], "observed")
        for group in output["aggregates"]:
            self.assertTrue(group["all_server_c40_observed"])
            self.assertTrue(group["all_c40_sustained_30s"])
            self.assertFalse(group["all_pp_progress_sustained_30s"])
        self.assertNotIn("secret-request", json.dumps(output))

    def test_pp_complete_turnover_does_not_infer_decode_progress(self):
        observations = []
        for timestamp, start in ((1, 0), (2, 40)):
            ids = [f"secret-request-{index}" for index in range(start, start + 40)]
            observations.append(
                self.pp(timestamp, rids=ids, output_lengths={rid: 10 for rid in ids})
            )
        self.pp_campaign(observations)
        output = analysis.analyze([self.root])
        for row in output["runs"]:
            self.assertTrue(row["server_c40_observed"])
            self.assertFalse(row["c40_sustained_30s"])
            self.assertFalse(row["pp_progress_sustained_30s"])
            detail = row["pp_turnover_and_progress"]
            self.assertEqual(detail["c40_windows_with_active_membership_change"], 1)
            self.assertEqual(detail["common_active_requests"]["max"], 0)
            self.assertEqual(detail["entering_active_requests"]["max"], 40)
            self.assertEqual(detail["leaving_active_requests"]["max"], 40)
            self.assertEqual(detail["observed_generated_token_increments"], 0)
            self.assertEqual(detail["decode_progress_status"], "not_observable")

    def test_dpa_selection_uses_observed_c40_without_added_duration_threshold(self):
        for profile, speed in (("dpa2", 500), ("dpa4", 600), ("dpa8", 550)):
            run = self.campaign(profile=profile, throughput=speed)
            for kind in ("long-cold", "long-warm"):
                path = self.root / "results" / run / f"r01-{kind}/telemetry.jsonl"
                self.linefile(path, [self.dpa(5, profile)])
        output = analysis.analyze([self.root])
        self.assertEqual(output["dpa_selection"]["screening_candidate"], "dpa4")
        self.assertTrue(all(row["server_c40_observed"] for row in output["runs"]))
        self.assertFalse(any(row["c40_sustained_30s"] for row in output["runs"]))
        self.assertFalse(
            any(group["all_c40_sustained_30s"] for group in output["aggregates"])
        )
        self.assertFalse(output["dpa_selection"]["final_recommendation_ready"])
        # Client peak and queued requests cannot supply missing running requests.
        for kind in ("long-cold", "long-warm"):
            path = (
                self.root
                / "results/campaign-dpa4-baseline"
                / f"r01-{kind}/telemetry.jsonl"
            )
            sample = self.dpa(5, "dpa4", count=36)
            for group in sample["loads"]["loads"]:
                group["num_waiting_reqs"] = 100
            self.linefile(path, [sample])
        output = analysis.analyze([self.root])
        self.assertEqual(output["dpa_selection"]["screening_candidate"], "dpa8")
        self.assertTrue(
            all(
                not row["server_c40_observed"]
                for row in output["runs"]
                if row["profile"] == "dpa4"
            )
        )

    def test_missing_server_evidence_never_substitutes_success_or_client_c40(self):
        run = self.campaign()
        for kind in ("long-cold", "long-warm"):
            path = self.root / "results" / run / f"r01-{kind}/telemetry.jsonl"
            self.linefile(
                path,
                [{"timestamp": 5, "activity": {"observable": True, "running": 400}}],
            )
        output = analysis.analyze([self.root])
        self.assertTrue(all(row["functional_valid"] for row in output["runs"]))
        self.assertTrue(all(not row["server_c40_observed"] for row in output["runs"]))
        self.assertTrue(all(not row["capacity_qualified"] for row in output["runs"]))
        self.assertIsNone(output["dpa_selection"]["screening_candidate"])

    def native_affinity_campaign(self):
        run = self.campaign()
        fingerprints = [f"{1000 + index:064x}" for index in range(20)]
        catalog = {
            "schema_version": 1,
            "dataset": "prefix_repetition",
            "configuration": {
                "seed": 0,
                "prefix_len": 60000,
                "suffix_len": 15000,
                "output_len": 1000,
                "num_prefixes": 20,
                "num_requests": 300,
            },
            "prefixes": [{"prefix_fingerprint": value} for value in fingerprints],
            "samples": [
                {
                    "sample_index": index,
                    "prefix_fingerprint": fingerprints[index // 15],
                    "request_body_sha256": f"{index:064x}",
                }
                for index in range(300)
            ],
        }
        catalog_path = self.root / "catalog.json"
        self.write(catalog_path, catalog)
        events = []
        for kind in ("long-cold", "long-warm"):
            path = self.root / "results" / run / f"r01-{kind}/requests.jsonl"
            records = list(analysis.json_lines(path))
            for index, record in enumerate(records):
                record["request_id"] = f"glm53-dpa2-r01-{kind}-abcdef01-{index}"
                record["response_id"] = f"private-native-{kind}-{index}"
                rank = (index // 15) % 2
                if kind == "long-warm":
                    rank = 1 - rank
                events.append(
                    {
                        "event": "request.finished",
                        "rid": record["response_id"],
                        "obj": {
                            "text": "PRIVATE RAW PROMPT",
                            "rid": record["response_id"],
                        },
                        "headers": {"authorization": "PRIVATE CREDENTIAL"},
                        "out": {
                            "text": "PRIVATE OUTPUT",
                            "meta_info": {
                                "id": record["response_id"],
                                "dp_rank": rank,
                                "completion_tokens": 1000,
                                "cached_tokens": 60000 if index % 2 else 0,
                            },
                        },
                    }
                )
            self.linefile(path, records)
        self.write_native(run, events)
        return run, catalog_path, events

    def write_native(self, run, events):
        path = self.root / run / "server.log"
        path.write_text(
            "".join(
                "2026-09-13T00:00:00Z [2026-09-13 00:00:00] " + json.dumps(row) + "\n"
                for row in events
            )
        )

    def test_native_response_join_catalog_hash_and_cold_warm_placement(self):
        run, catalog, events = self.native_affinity_campaign()
        self.write_native(run, events + [events[0]])
        result = analysis.analyze([self.root], catalog)
        cold = next(row for row in result["runs"] if row["workload"] == "long-cold")
        affinity = cold["request_affinity"]
        self.assertEqual(affinity["native_request_dp_status"], "observed")
        self.assertEqual(affinity["native_cached_tokens_status"], "observed")
        self.assertEqual(affinity["prefix_affinity_status"], "observed")
        self.assertEqual(affinity["deduplicated_native_events"], 1)
        self.assertEqual(affinity["native_rank_covered_requests"], 300)
        self.assertEqual(
            [item["completed_requests_observed"] for item in affinity["per_dp"]],
            [150, 150],
        )
        self.assertEqual(
            affinity["prefix_replication"]["processing_group_count_distribution"],
            {"1": 20},
        )
        self.assertEqual(
            result["prefix_placement_comparisons"][0]["new_prefix_dp_pairs_in_warm"], 20
        )
        self.assertEqual(
            result["prefix_placement_comparisons"][0][
                "prefixes_with_changed_processing_groups"
            ],
            20,
        )
        aggregate = result["aggregates"][0]["request_affinity"]
        self.assertEqual(aggregate["prefix_affinity_observed_repetitions"], 1)
        self.assertTrue(aggregate["all_prefix_affinity_observed"])
        serialized = json.dumps(result)
        for forbidden in (
            "PRIVATE",
            "private-native",
            "glm53-dpa2-r01",
            "request_body_sha256",
            "prefix_fingerprint",
            "authorization",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_partial_cached_fields_do_not_erase_complete_prefix_assignment(self):
        run, catalog, events = self.native_affinity_campaign()
        del events[0]["out"]["meta_info"]["cached_tokens"]
        self.write_native(run, events)
        result = analysis.analyze([self.root], catalog)
        row = next(row for row in result["runs"] if row["workload"] == "long-cold")
        affinity = row["request_affinity"]
        self.assertEqual(affinity["prefix_affinity_status"], "observed")
        self.assertEqual(affinity["native_cached_tokens_status"], "not_observable")
        self.assertEqual(affinity["native_cached_tokens_covered_requests"], 299)
        self.assertEqual(
            affinity["coverage_issues"]["missing_or_invalid_native_cached_tokens"], 1
        )

    def test_missing_native_cannot_join_via_x_request_id_and_conflict_is_rejected(self):
        run, catalog, events = self.native_affinity_campaign()
        # Same request-id is deliberately a wrong native rid: only SSE response_id may join.
        events[0]["rid"] = "glm53-dpa2-r01-long-cold-abcdef01-0"
        conflicting = json.loads(json.dumps(events[1]))
        conflicting["out"]["meta_info"]["dp_rank"] = (
            1 - events[1]["out"]["meta_info"]["dp_rank"]
        )
        self.write_native(run, events + [conflicting])
        result = analysis.analyze([self.root], catalog)
        cold = next(row for row in result["runs"] if row["workload"] == "long-cold")[
            "request_affinity"
        ]
        self.assertEqual(cold["native_request_dp_status"], "not_observable")
        self.assertEqual(cold["prefix_affinity_status"], "not_observable")
        self.assertEqual(cold["native_rank_covered_requests"], 298)
        self.assertEqual(cold["coverage_issues"]["missing_native_finished_event"], 1)
        self.assertEqual(
            cold["coverage_issues"]["conflicting_native_finished_events"], 1
        )
        self.assertEqual(
            result["prefix_placement_comparisons"][0]["status"], "not_observable"
        )

    def test_catalog_body_mismatch_and_invalid_catalog_fail_closed(self):
        run, catalog, events = self.native_affinity_campaign()
        path = self.root / "results" / run / "r01-long-cold/requests.jsonl"
        records = list(analysis.json_lines(path))
        records[0]["request_sha256"] = "f" * 64
        self.linefile(path, records)
        result = analysis.analyze([self.root], catalog)
        cold = next(row for row in result["runs"] if row["workload"] == "long-cold")[
            "request_affinity"
        ]
        self.assertEqual(cold["native_request_dp_status"], "observed")
        self.assertEqual(cold["prefix_affinity_status"], "not_observable")
        self.assertEqual(cold["prefix_rank_covered_requests"], 299)
        self.assertEqual(
            cold["coverage_issues"]["catalog_request_body_sha256_mismatch"], 1
        )
        raw = json.loads(catalog.read_text())
        raw["samples"][0]["sample_index"] = 1
        self.write(catalog, raw)
        result = analysis.analyze([self.root], catalog)
        self.assertEqual(result["catalog_status"], "invalid")
        self.assertTrue(
            all(
                row["request_affinity"]["prefix_affinity_status"] == "not_observable"
                for row in result["runs"]
            )
        )

    def test_pp_native_none_rank_uses_explicit_single_dp_topology_only(self):
        run, catalog, events = self.native_affinity_campaign()
        for event in events:
            event["out"]["meta_info"]["dp_rank"] = None
        self.write_native(run, events)
        dpa = analysis.analyze([self.root], catalog)
        self.assertTrue(
            all(
                row["request_affinity"]["native_request_dp_status"] == "not_observable"
                for row in dpa["runs"]
            )
        )
        state_path = self.root / run / "run-state.json"
        state = json.loads(state_path.read_text())
        state["profile"] = "pp2"
        self.write(state_path, state)
        for kind in ("long-cold", "long-warm"):
            path = self.root / "results" / run / f"r01-{kind}/requests.jsonl"
            records = list(analysis.json_lines(path))
            for record in records:
                record["request_id"] = record["request_id"].replace(
                    "glm53-dpa2-", "glm53-pp2-"
                )
            self.linefile(path, records)
        pp = analysis.analyze([self.root], catalog)
        for row in pp["runs"]:
            affinity = row["request_affinity"]
            self.assertEqual(affinity["native_request_dp_status"], "observed")
            self.assertEqual(affinity["native_cached_tokens_status"], "observed")
            self.assertEqual(affinity["prefix_affinity_status"], "observed")
            self.assertEqual(affinity["dp_rank_sources"], {"single-DP topology": 300})
            self.assertEqual(affinity["per_dp"][0]["completed_requests_observed"], 300)

    def preparation_copy(self, run, folder="preparation"):
        root = self.root / "results" / run
        target = root / folder
        target.mkdir()
        provenance = json.loads((root / "provenance.json").read_text())
        provenance.update(
            purpose="preparation", secret_note="PRIVATE PREPARATION TOKEN"
        )
        self.write(target / "provenance.json", provenance)
        for workload in ("long-cold", "long-warm"):
            directory = target / f"r01-{workload}"
            shutil.copytree(root / directory.name, directory)
            raw = json.loads((directory / "vllm.json").read_text())
            raw["output_throughput"] = 999999
            self.write(directory / "vllm.json", raw)
        return target

    def test_explicit_preparation_excluded_from_every_comparison(self):
        for profile, throughput in (("dpa2", 500), ("dpa4", 600), ("dpa8", 550)):
            run = self.campaign(profile=profile, throughput=throughput)
            self.preparation_copy(run, folder="arbitrary-rehearsal-folder")
        result = analysis.analyze([self.root, self.root])
        self.assertEqual(len(result["runs"]), 6)
        self.assertEqual(result["preparation"]["excluded_workloads"], 6)
        self.assertEqual(result["preparation"]["functional_valid_workloads"], 6)
        self.assertEqual(result["dpa_selection"]["screening_candidate"], "dpa4")
        self.assertEqual(len(result["prefix_placement_comparisons"]), 3)
        self.assertTrue(
            all(group["repetitions"] == 1 for group in result["aggregates"])
        )
        rendered = json.dumps(result)
        for secret in (
            "999999",
            "arbitrary-rehearsal-folder",
            "PRIVATE PREPARATION TOKEN",
        ):
            self.assertNotIn(secret, rendered)

    def test_preparation_directory_name_has_no_classification_power(self):
        run = self.campaign()
        root = self.root / "results" / run
        target = root / "preparation"
        target.mkdir()
        for workload in ("long-cold", "long-warm"):
            shutil.move(root / f"r01-{workload}", target / f"r01-{workload}")
        # The nearest declaration remains explicit root purpose=measurement.
        result = analysis.analyze([self.root])
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(result["preparation"]["excluded_workloads"], 0)
        self.assertTrue(all(row["purpose"] == "measurement" for row in result["runs"]))

    def test_campaign_predeclaration_excludes_all_workloads_and_attempts(self):
        for profile in ("dpa2", "dpa4", "dpa8"):
            self.campaign(profile=profile)
        self.write(
            self.root / "PREPARATION.json",
            {
                "purpose": "preparation",
                "classified_at": -3,
                "exclude_from_comparative_measurements": True,
                "source_commit": "a" * 40,
                "reason": "PRIVATE DECISION REASON",
            },
        )
        result = analysis.analyze([self.root])
        self.assertEqual(result["runs"], [])
        self.assertEqual(result["aggregates"], [])
        self.assertEqual(result["attempts"], [])
        self.assertEqual(result["prefix_placement_comparisons"], [])
        self.assertEqual(result["dpa_selection"]["status"], "NO_MEASUREMENTS")
        self.assertEqual(result["preparation"]["excluded_workloads"], 6)
        self.assertEqual(
            result["preparation"]["campaigns"][0]["declaration_timing"],
            "before_workloads",
        )
        self.assertNotIn("PRIVATE DECISION", json.dumps(result))

    def test_late_preparation_declaration_cannot_enable_cherry_picked_ranking(self):
        for profile in ("dpa2", "dpa4", "dpa8"):
            run = self.campaign(profile=profile)
        target = self.preparation_copy(run)
        provenance = json.loads((target / "provenance.json").read_text())
        provenance["timestamp"] = 50
        self.write(target / "provenance.json", provenance)
        result = analysis.analyze([self.root])
        self.assertEqual(
            result["dpa_selection"]["status"], "PREPARATION_CLASSIFICATION_NOT_VERIFIED"
        )
        self.assertEqual(result["dpa_selection"]["ranking"], [])
        self.assertEqual(result["preparation"]["excluded_workloads"], 2)

    def test_tooling_versions_and_configmap_hashes_are_comparability_dimensions(self):
        for field, different in (
            ("tooling_commit", "e" * 40),
            ("configmap_sha256", "f" * 64),
        ):
            with self.subTest(field=field):
                for profile in ("dpa2", "dpa4", "dpa8"):
                    self.campaign(profile=profile)
                run = "campaign-dpa4-baseline"
                for path in (
                    self.root / run / "run-state.json",
                    self.root / "results" / run / "provenance.json",
                ):
                    value = json.loads(path.read_text())
                    value[field] = different
                    self.write(path, value)
                result = analysis.analyze([self.root])
                self.assertTrue(
                    all(row["experiment_identity_verified"] for row in result["runs"])
                )
                self.assertFalse(
                    result["dpa_selection"][
                        "same_node_serving_tooling_configmap_dataset"
                    ]
                )
                self.assertIsNone(result["dpa_selection"]["screening_candidate"])
                self.assertEqual(result["dpa_selection"]["ranking"], [])

    def test_asymmetric_provenance_mismatch_and_missing_identity_are_unqualified(self):
        for field in ("source_commit", "tooling_commit", "configmap_sha256"):
            for missing in (True, False):
                with self.subTest(field=field, missing=missing):
                    run = self.campaign()
                    path = self.root / "results" / run / "provenance.json"
                    provenance = json.loads(path.read_text())
                    if missing:
                        del provenance[field]
                    else:
                        provenance[field] = "e" * (
                            64 if field == "configmap_sha256" else 40
                        )
                    self.write(path, provenance)
                    result = analysis.analyze([self.root])
                    self.assertTrue(
                        all(row["functional_valid"] for row in result["runs"])
                    )
                    self.assertFalse(
                        any(
                            row["experiment_identity_verified"]
                            for row in result["runs"]
                        )
                    )
                    self.assertTrue(
                        all(
                            group["valid_repetitions"] == 0
                            for group in result["aggregates"]
                        )
                    )

    def test_legacy_missing_purpose_is_visible_but_never_qualified(self):
        run = self.campaign()
        (self.root / "results" / run / "provenance.json").unlink()
        result = analysis.analyze([self.root])
        self.assertEqual(len(result["runs"]), 2)
        self.assertTrue(all(row["purpose"] == "unspecified" for row in result["runs"]))
        self.assertTrue(
            all(group["valid_repetitions"] == 0 for group in result["aggregates"])
        )

    def test_failed_explicit_measurement_intent_still_counts_as_profile_attempt(self):
        for profile in ("dpa2", "dpa4", "dpa8"):
            self.campaign(profile=profile)
        run = "campaign-dpa8-baseline"
        state_path = self.root / run / "run-state.json"
        state = json.loads(state_path.read_text())
        state.update(status="failed", measurement_job_started=False)
        self.write(state_path, state)
        shutil.rmtree(self.root / "results" / run)
        result = analysis.analyze([self.root])
        self.assertTrue(result["dpa_selection"]["all_dpa_profiles_attempted"])
        failed = result["failed_profiles"][0]
        self.assertEqual(failed["purpose"], "measurement")
        self.assertFalse(failed["measurement_job_started"])

    def test_compile_changes_preserve_functional_result_but_exclude_comparison(self):
        run = self.campaign()
        directory = self.root / "results" / run / "r01-long-cold"
        after = json.loads((directory / "after-compile-cache.json").read_text())
        after["files"] = [
            {
                "path": "deepgemm/PRIVATE-CACHE-NAME/cubin.bin",
                "size": 123,
                "mtime_ns": 20_000_000_000,
            }
        ]
        self.write(directory / "after-compile-cache.json", after)
        result = analysis.analyze([self.root])
        cold = next(row for row in result["runs"] if row["workload"] == "long-cold")
        self.assertTrue(cold["functional_valid"])
        self.assertEqual(cold["purpose"], "measurement")
        evidence = cold["compile_preparation"]
        self.assertTrue(evidence["preparation_evidence"])
        self.assertFalse(evidence["comparison_qualified"])
        self.assertEqual(
            evidence["artifact_changes"]["mtime_classification"],
            {"during_measurement": 1},
        )
        self.assertNotIn("PRIVATE-CACHE-NAME", json.dumps(result))
        group = next(
            row for row in result["aggregates"] if row["workload"] == "long-cold"
        )
        self.assertEqual(group["valid_repetitions"], 0)
        self.assertTrue(group["all_functional_valid"])
        self.assertFalse(group["all_comparison_qualified"])

    def test_untimed_cache_changes_are_not_claimed_as_timed_compilation(self):
        run = self.campaign()
        directory = self.root / "results" / run / "r01-long-cold"
        evidence = analysis.compile_preparation_evidence(
            directory, "1970-01-01T00:00:11Z ok", 10, 40
        )
        self.assertTrue(evidence["comparison_qualified"])
        path = directory / "after-compile-cache.json"
        after = json.loads(path.read_text())
        after["files"] = [
            {"path": "triton/one.cubin", "size": 1, "mtime_ns": 5_000_000_000}
        ]
        self.write(path, after)
        evidence = analysis.compile_preparation_evidence(
            directory, "1970-01-01T00:00:11Z ok", 10, 40
        )
        self.assertEqual(
            evidence["artifact_changes"]["mtime_classification"],
            {"before_measurement": 1},
        )
        self.assertTrue(evidence["preparation_evidence"])
        self.assertEqual(evidence["in_window_markers"], {})

    def test_compile_markers_are_windowed_and_missing_inventory_not_zero(self):
        run = self.campaign()
        directory = self.root / "results" / run / "r01-long-cold"
        before = "1969-12-31T23:59:59Z DeepGEMM warmup\rDeepGEMM warmup"
        evidence = analysis.compile_preparation_evidence(directory, before, 0, 40)
        self.assertTrue(evidence["comparison_qualified"])
        inside = "1970-01-01T00:00:20Z Try DeepGEMM JIT Compiling for PRIVATE"
        evidence = analysis.compile_preparation_evidence(
            directory, before + "\n" + inside, 0, 40
        )
        self.assertEqual(evidence["in_window_markers"], {"deepgemm_compile_attempt": 1})
        self.assertTrue(evidence["preparation_evidence"])
        self.assertNotIn("PRIVATE", json.dumps(evidence))
        (directory / "before-compile-cache.json").unlink()
        evidence = analysis.compile_preparation_evidence(directory, before, 0, 40)
        self.assertEqual(evidence["inventory_status"], "not_observable")
        self.assertIsNone(evidence["artifact_changes"])
        self.assertFalse(evidence["comparison_qualified"])

    def test_partial_duplicate_and_wrong_window_compile_snapshots_are_unknown(self):
        for case in ("partial", "duplicate", "late_before"):
            with self.subTest(case=case):
                run = self.campaign()
                directory = self.root / "results" / run / "r01-long-cold"
                path = directory / "before-compile-cache.json"
                before = json.loads(path.read_text())
                if case == "partial":
                    before.update(status="partial", errors=[{"type": "PRIVATE ERROR"}])
                elif case == "duplicate":
                    before["files"] = [
                        {"path": "triton/one", "size": 1, "mtime_ns": 0}
                    ] * 2
                else:
                    before["finished_at"] = 1
                self.write(path, before)
                evidence = analysis.compile_preparation_evidence(
                    directory, "1970-01-01T00:00:01Z ok", 0, 40
                )
                self.assertEqual(evidence["inventory_status"], "not_observable")
                self.assertFalse(evidence["comparison_qualified"])
                self.assertNotIn("PRIVATE ERROR", json.dumps(evidence))


if __name__ == "__main__":
    unittest.main()
