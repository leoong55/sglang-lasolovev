import importlib.util
import json
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
            "source_commit": "a" * 40,
            "image": "private-host.example/image@sha256:" + "b" * 64,
            "node": "private-host-192.0.2.123",
            "pod_uid": "private-pod-uuid",
            "error": "credential-secret-host",
        }
        self.write(self.root / run_id / "run-state.json", state)
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


if __name__ == "__main__":
    unittest.main()
