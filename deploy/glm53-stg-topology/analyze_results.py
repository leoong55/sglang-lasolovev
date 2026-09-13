"""Aggregate private benchmark artifacts without publishing infrastructure data.

The frozen runtime is not imported or modified. Completion and concurrency claims
are reconstructed independently from its artifacts. Public output uses an explicit
schema; paths, URLs, IDs, prompts, errors, and arbitrary metric keys never pass
through. Raw artifacts remain under the supplied local campaign directories.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROFILES = {"pp2", "dpa2", "dpa4", "dpa8"}
PHASES = {"baseline", "repeat", "hicache"}
WORKLOADS = {"short", "long-cold", "long-warm"}
MAX_GAP = 3.5
SUSTAINED_SECONDS = 30.0
METRICS = {
    "duration",
    "completed",
    "failed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
}
LATENCY = re.compile(
    r"(?:mean|median|std|p(?:50|90|95|99))_(?:ttft|tpot|itl|e2el)_ms\Z"
)


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def json_lines(path):
    if not path.is_file():
        return
    with path.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
                yield value if isinstance(value, dict) else None
            except ValueError:
                yield None


def safe_metrics(raw):
    return {
        key: value
        for key, value in raw.items()
        if (key in METRICS or LATENCY.fullmatch(key)) and number(value) and value >= 0
    }


def source_identity(state):
    commit = state.get("source_commit", "")
    digest = str(state.get("image", "")).rsplit("@", 1)[-1]
    return {
        "source_commit": (
            commit
            if isinstance(commit, str) and re.fullmatch("[0-9a-f]{40}", commit)
            else None
        ),
        "image_digest": (
            digest if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) else None
        ),
    }


def longest_window(samples):
    """Invalid points and observation gaps break a window; duplicates add no time."""
    start = prev = None
    longest = 0.0
    unique = {}
    for timestamp, value in samples:
        if number(timestamp):
            unique[timestamp] = (
                value if timestamp not in unique or unique[timestamp] == value else None
            )
    for timestamp, value in sorted(unique.items()):
        if not number(timestamp):
            continue
        if prev is not None and timestamp <= prev:
            continue
        if (
            not number(value)
            or value < 40
            or (prev is not None and timestamp - prev > MAX_GAP)
        ):
            start = None
        if number(value) and value >= 40:
            start = timestamp if start is None else start
            longest = max(longest, timestamp - start)
        prev = timestamp
    return longest


def pp_evidence(log, begin, end):
    rows = {}
    for line in log.splitlines():
        if "GLM53_PP_ACTIVITY " not in line:
            continue
        try:
            row = json.loads(line.split("GLM53_PP_ACTIVITY ", 1)[1])
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        timestamp = row.get("timestamp")
        if (
            not number(timestamp)
            or not begin <= timestamp <= end
            or row.get("stage") != 0
            or row.get("tp") != 0
        ):
            continue
        rids = row.get("rids")
        valid = (
            row.get("valid") is True
            and isinstance(rids, list)
            and all(isinstance(rid, str) for rid in rids)
        )
        ids = set(rids) if valid else set()
        lengths = row.get("output_lengths", {})
        lengths_valid = (
            isinstance(lengths, dict)
            and set(lengths) == ids
            and all(count(value) for value in lengths.values())
        )
        item = {
            "timestamp": timestamp,
            "count": len(ids) if valid else None,
            "ids": ids,
            "lengths": lengths if valid and lengths_valid else None,
        }
        # Conflicting duplicate timestamps indicate ambiguous observations, not
        # an opportunity to choose the more favorable sample.
        if timestamp in rows and rows[timestamp] != item:
            item = {
                "timestamp": timestamp,
                "count": None,
                "ids": set(),
                "lengths": None,
            }
        rows[timestamp] = item
    records = [rows[timestamp] for timestamp in sorted(rows)]
    points = [(row["timestamp"], row["count"]) for row in records]
    progress_windows = 0
    longest_progress = 0.0
    current_progress = 0.0
    for left, right in zip(records, records[1:]):
        delta = right["timestamp"] - left["timestamp"]
        advancing = 0
        if (
            0 < delta <= MAX_GAP
            and left["lengths"] is not None
            and right["lengths"] is not None
        ):
            advancing = sum(
                right["lengths"][rid] > left["lengths"][rid]
                for rid in left["ids"] & right["ids"]
            )
        if advancing >= 40:
            progress_windows += 1
            current_progress += delta
            longest_progress = max(longest_progress, current_progress)
        else:
            current_progress = 0.0
    return points, progress_windows, longest_progress


def dpa_evidence(path, begin, end, profile):
    expected = int(profile[3:])
    points = []
    previous = None
    cached_polls = 0
    ages, intervals = [], []
    previous_point = None
    for sample in json_lines(path):
        if not sample:
            # A damaged line has no trustworthy timestamp. A full evidence gap
            # is still detected by neighboring samples; conservatively invalidate
            # the most recent point so an error cannot bridge a good window.
            if points:
                points[-1] = (points[-1][0], None)
            continue
        timestamp = sample.get("timestamp")
        if not number(timestamp) or not begin <= timestamp <= end:
            continue
        payload = sample.get("loads", {})
        groups = payload.get("loads") if isinstance(payload, dict) else payload
        valid = isinstance(groups, list) and len(groups) == expected
        if valid:
            valid = all(isinstance(group, dict) for group in groups)
        if valid:
            ranks = [group.get("dp_rank") for group in groups]
            valid = (
                all(count(rank) for rank in ranks)
                and set(ranks) == set(range(expected))
                and all(count(group.get("num_running_reqs")) for group in groups)
            )
        if valid:
            # The pinned /v1/loads timestamp is time.time(). Reject stale cached
            # snapshots and future samples; don't trust runner's derived 'activity'.
            valid = all(
                number(group.get("timestamp"))
                and -MAX_GAP <= timestamp - group["timestamp"] <= MAX_GAP
                for group in groups
            )
        if not valid:
            points.append((timestamp, None))
            continue
        groups = sorted(groups, key=lambda group: group["dp_rank"])
        vector = tuple(group["timestamp"] for group in groups)
        ages.extend(timestamp - value for value in vector)
        if previous is not None and not all(
            new > old for new, old in zip(vector, previous)
        ):
            cached_polls += 1
            continue
        previous = vector
        # Use server time, not poll time. Cached polls neither extend a sustained
        # window nor add independent evidence. All groups must have advanced.
        point_time = max(vector)
        if min(vector) < begin or point_time > end:
            continue
        points.append((point_time, sum(group["num_running_reqs"] for group in groups)))
        if previous_point is not None:
            intervals.append(point_time - previous_point)
        previous_point = point_time
    return points, {
        "cached_or_partially_advanced_polls": cached_polls,
        "server_snapshot_age_seconds": distribution(ages),
        "distinct_group_snapshot_interval_seconds": distribution(intervals),
        "snapshot_publish_interval_unit": "decode_iterations",
        "snapshot_missing_timestamps": "unobservable; no interval inferred from polling",
    }


def distribution(values):
    return (
        {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "samples": len(values),
        }
        if values
        else None
    )


def completion_evidence(path, verdict, workload):
    expected = 400 if workload == "short" else 300
    measured = []
    malformed = 0
    for record in json_lines(path):
        if record is None:
            malformed += 1
        elif record.get("measured") is True:
            measured.append(record)
    identifiers = [record.get("request_id") for record in measured]
    complete_ids = all(
        isinstance(value, str) and value for value in identifiers
    ) and len(set(identifiers)) == len(identifiers)
    invalid = 0
    finishes, tokens, cache = Counter(), Counter(), Counter()
    for record in measured:
        reason_map = record.get("finish_reasons", {})
        reason = reason_map.get("0") if isinstance(reason_map, dict) else None
        token_count = record.get("completion_tokens")
        usage = record.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("prompt_tokens_details") or {}
        cached = (
            details.get("cached_tokens", usage.get("cached_tokens"))
            if isinstance(details, dict)
            else None
        )
        cache["reported" if count(cached) else "not_observable"] += 1
        finishes[
            (
                reason
                if isinstance(reason, str) and reason in {"stop", "length"}
                else "other_or_missing"
            )
        ] += 1
        tokens[
            (
                str(token_count)
                if count(token_count) and token_count <= 1000
                else "invalid_or_missing"
            )
        ] += 1
        sampling = record.get("sampling", {})
        if not isinstance(sampling, dict):
            sampling = {}
        stream_options = sampling.get("stream_options")
        valid_sampling = (
            isinstance(sampling, dict)
            and sampling.get("max_completion_tokens", sampling.get("max_tokens"))
            == 1000
            and sampling.get("stream") is True
            and isinstance(stream_options, dict)
            and stream_options.get("include_usage") is True
        )
        if workload == "short":
            valid_sampling = (
                valid_sampling
                and not sampling.get("ignore_eos")
                and "temperature" not in sampling
                and "chat_template_kwargs" not in sampling
            )
        else:
            valid_sampling = (
                valid_sampling
                and sampling.get("ignore_eos") is True
                and sampling.get("temperature") == 0.3
                and sampling.get("chat_template_kwargs") == {"enable_thinking": True}
            )
        valid = (
            record.get("status") == 200
            and record.get("sse_done") is True
            and not record.get("evidence_errors")
            and not record.get("proxy_error")
            and isinstance(reason_map, dict)
            and set(reason_map) == {"0"}
            and isinstance(reason, str)
            and reason in {"stop", "length"}
            and count(token_count)
            and 0 < token_count <= 1000
            and usage.get("completion_tokens") == token_count
            and valid_sampling
            and number(record.get("started_at"))
            and number(record.get("finished_at"))
            and record["finished_at"] >= record["started_at"]
        )
        if workload != "short":
            valid = valid and reason == "length" and token_count == 1000
        invalid += not valid
    starts = [
        record["started_at"] for record in measured if number(record.get("started_at"))
    ]
    ends = [
        record["finished_at"]
        for record in measured
        if number(record.get("finished_at"))
    ]
    begin, end = min(starts, default=None), max(ends, default=None)
    timing_valid = (
        begin is not None
        and end is not None
        and number(verdict.get("started_at"))
        and number(verdict.get("finished_at"))
        and abs(begin - verdict["started_at"]) < 0.001
        and abs(end - verdict["finished_at"]) < 0.001
    )
    valid = (
        len(measured) == expected
        and complete_ids
        and invalid == 0
        and malformed == 0
        and timing_valid
        and verdict.get("functional_valid") is True
        and verdict.get("exit_code") == 0
        and verdict.get("expected_requests") == expected
        and verdict.get("observed_requests") == expected
    )
    # Body hashes are compared only locally, never emitted in the aggregate.
    hashes = [record.get("request_sha256") for record in measured]
    fingerprint = (
        tuple(sorted(hashes))
        if len(hashes) == expected
        and all(
            isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
            for value in hashes
        )
        else None
    )
    return {
        "functional_valid": valid,
        "expected_requests": expected,
        "observed_requests": len(measured),
        "invalid_requests": invalid,
        "malformed_records": malformed,
        "timestamps_valid": timing_valid,
        "finish_reasons": dict(finishes),
        "completion_tokens": dict(tokens),
        "cache_observability": dict(cache),
        "_begin": begin,
        "_end": end,
        "_dataset": fingerprint,
    }


def load_catalog(path):
    if path is None:
        return {"status": "absent"}
    raw = read_json(Path(path))
    expected = {
        "seed": 0,
        "prefix_len": 60000,
        "suffix_len": 15000,
        "output_len": 1000,
        "num_prefixes": 20,
        "num_requests": 300,
    }
    invalid = {"status": "invalid"}
    if (
        raw.get("schema_version") != 1
        or raw.get("dataset") != "prefix_repetition"
        or raw.get("configuration") != expected
    ):
        return invalid
    prefixes, samples = raw.get("prefixes"), raw.get("samples")
    if (
        not isinstance(prefixes, list)
        or len(prefixes) != 20
        or not isinstance(samples, list)
        or len(samples) != 300
    ):
        return invalid
    labels = {}
    for index, item in enumerate(prefixes):
        fingerprint = item.get("prefix_fingerprint") if isinstance(item, dict) else None
        if (
            not isinstance(fingerprint, str)
            or not re.fullmatch("[0-9a-f]{64}", fingerprint)
            or fingerprint in labels
        ):
            return invalid
        labels[fingerprint] = index
    by_index = {}
    counts = Counter()
    for item in samples:
        if not isinstance(item, dict):
            return invalid
        index, body_hash, fingerprint = (
            item.get("sample_index"),
            item.get("request_body_sha256"),
            item.get("prefix_fingerprint"),
        )
        if (
            not count(index)
            or index >= 300
            or index in by_index
            or not isinstance(body_hash, str)
            or not re.fullmatch("[0-9a-f]{64}", body_hash)
            or not isinstance(fingerprint, str)
            or fingerprint not in labels
        ):
            return invalid
        label = labels[fingerprint]
        by_index[index] = {"body_hash": body_hash, "group": label}
        counts[label] += 1
    if set(by_index) != set(range(300)) or any(
        value != 15 for value in counts.values()
    ):
        return invalid
    return {"status": "verified", "samples": by_index}


def native_finished_events(log):
    """Read only native request.finished events; do not join x-request-id."""
    events, conflicts, duplicates = {}, set(), Counter()
    for line in log.splitlines():
        if "request.finished" not in line or "{" not in line:
            continue
        try:
            row, _ = json.JSONDecoder().raw_decode(line[line.index("{") :])
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("event") != "request.finished":
            continue
        rid = row.get("rid")
        if not isinstance(rid, str) or not rid:
            continue
        out = row.get("out")
        info = out.get("meta_info") if isinstance(out, dict) else None
        info = info if isinstance(info, dict) else {}
        # Restrict the private join index too. Text, headers and nested arbitrary
        # metadata are never retained by the analyzer.
        projection = {
            key: info.get(key)
            for key in ("id", "dp_rank", "cached_tokens", "completion_tokens")
        }
        if rid in events:
            if events[rid] != projection:
                conflicts.add(rid)
            else:
                duplicates[rid] += 1
        else:
            events[rid] = projection
    return events, conflicts, duplicates


def cached_distribution(values):
    result = distribution(values)
    if result is not None:
        result.update(total=sum(values), zero=sum(value == 0 for value in values))
    return result


def request_affinity(path, log, profile, workload, catalog):
    expected = 400 if workload == "short" else 300
    size = 1 if profile == "pp2" else int(profile[3:])
    records = [
        record
        for record in json_lines(path)
        if record and record.get("measured") is True
    ]
    events, conflicts, duplicates = native_finished_events(log)
    issues, per_dp, prefix_dp, cache_values = (
        Counter(),
        Counter(),
        defaultdict(Counter),
        defaultdict(list),
    )
    response_ids, request_ids, sample_indices = set(), set(), set()
    joined = ranked = cached = matched = catalog_verified_requests = (
        duplicate_events
    ) = 0
    for record in records:
        request_id, response_id = record.get("request_id"), record.get("response_id")
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in request_ids
        ):
            issues["missing_or_duplicate_measured_request_id"] += 1
            continue
        request_ids.add(request_id)
        if (
            not isinstance(response_id, str)
            or not response_id
            or response_id in response_ids
        ):
            issues["missing_or_duplicate_response_id"] += 1
            continue
        response_ids.add(response_id)
        if response_id not in events:
            issues["missing_native_finished_event"] += 1
            continue
        if response_id in conflicts:
            issues["conflicting_native_finished_events"] += 1
            continue
        native = events[response_id]
        duplicate_events += duplicates[response_id]
        if native["id"] is not None and native["id"] != response_id:
            issues["native_meta_id_mismatch"] += 1
            continue
        if native["completion_tokens"] is not None and native[
            "completion_tokens"
        ] != record.get("completion_tokens"):
            issues["native_completion_tokens_mismatch"] += 1
            continue
        joined += 1
        rank = native["dp_rank"]
        if not count(rank) or rank >= size:
            issues["missing_or_invalid_native_dp_rank"] += 1
            continue
        ranked += 1
        per_dp[rank] += 1
        value = native["cached_tokens"]
        if count(value):
            cached += 1
            cache_values[rank].append(value)
        else:
            issues["missing_or_invalid_native_cached_tokens"] += 1
        if workload == "short" or catalog["status"] != "verified":
            continue
        match = re.fullmatch(
            r"glm53-" + profile + r"-r\d+-" + workload + r"-[0-9a-f]{8}-(\d+)",
            request_id,
        )
        if not match:
            issues["request_id_sample_index_unrecognized"] += 1
            continue
        index = int(match[1])
        sample = catalog["samples"].get(index)
        if sample is None or index in sample_indices:
            issues["missing_or_duplicate_catalog_sample_index"] += 1
            continue
        sample_indices.add(index)
        if record.get("request_sha256") != sample["body_hash"]:
            issues["catalog_request_body_sha256_mismatch"] += 1
            continue
        catalog_verified_requests += 1
        prefix_dp[sample["group"]][rank] += 1
        matched += 1
    complete = len(records) == expected and ranked == expected
    prefix_complete = workload != "short" and complete and matched == expected
    groups = [
        {
            "prefix_group": group,
            "requests_observed": sum(ranks.values()),
            "per_dp_completed_requests": {
                str(rank): ranks[rank] for rank in sorted(ranks)
            },
            "processing_dp_groups": len(ranks),
        }
        for group, ranks in sorted(prefix_dp.items())
    ]
    scope = "DP groups that processed a prefix during this run; not simultaneous cache residency."
    result = {
        "native_request_dp_status": "observed" if complete else "not_observable",
        "native_cached_tokens_status": (
            "observed" if complete and cached == expected else "not_observable"
        ),
        "prefix_affinity_status": (
            "not_applicable"
            if workload == "short"
            else "observed" if prefix_complete else "not_observable"
        ),
        "catalog_status": catalog["status"],
        "expected_requests": expected,
        "measured_requests": len(records),
        "native_joined_requests": joined,
        "native_rank_covered_requests": ranked,
        "native_cached_tokens_covered_requests": cached,
        "catalog_body_verified_requests": catalog_verified_requests,
        "prefix_rank_covered_requests": matched,
        "deduplicated_native_events": duplicate_events,
        "coverage_issues": dict(issues),
        "counts_scope": (
            "All measured requests"
            if complete
            else "Observed subset; incomplete coverage"
        ),
        "per_dp": [
            {
                "dp_rank": rank,
                "completed_requests_observed": per_dp[rank],
                "cached_tokens": cached_distribution(cache_values[rank]),
            }
            for rank in sorted(per_dp)
        ],
        "prefix_groups": groups,
        "prefix_replication": {
            "scope": scope,
            "status": "observed" if prefix_complete else "not_observable",
            "prefix_groups_observed": len(groups),
            "groups_processed_by_multiple_dp": sum(
                group["processing_dp_groups"] > 1 for group in groups
            ),
            "processing_group_count_distribution": dict(
                Counter(str(group["processing_dp_groups"]) for group in groups)
            ),
        },
    }
    return result, dict(cache_values)


def summarize_workload(state, verdict_file, log, catalog):
    verdict = read_json(verdict_file)
    match = re.fullmatch(
        r"r(\d+)-(short|long-cold|long-warm)", verdict_file.parent.name
    )
    if not match or verdict.get("workload") != match[2]:
        return None
    workload = match[2]
    row = completion_evidence(verdict_file.parent / "requests.jsonl", verdict, workload)
    raw = read_json(verdict_file.parent / "vllm.json")
    expected = row["expected_requests"]
    row["functional_valid"] &= (
        raw.get("completed") == expected and raw.get("failed", 0) == 0
    )
    row["orchestration_valid"] = state.get("status") == "measured" and not any(
        state.get(key)
        for key in ("cleanup_error", "export_error", "final_collection_error")
    )
    begin, end = row["_begin"], row["_end"]
    points, progress_windows, progress_seconds = [], None, None
    dpa_observation = None
    if begin is not None and end is not None and row["timestamps_valid"]:
        if state["profile"] == "pp2":
            points, progress_windows, progress_seconds = pp_evidence(log, begin, end)
        else:
            points, dpa_observation = dpa_evidence(
                verdict_file.parent / "telemetry.jsonl", begin, end, state["profile"]
            )
            info = read_json(verdict_file.parent / "before-server-info.json")
            interval = info.get("load_snapshot_publish_interval")
            dpa_observation["configured_snapshot_publish_interval"] = (
                interval if count(interval) else None
            )
    valid_points = [value for _, value in points if number(value)]
    sustained = longest_window(points)
    affinity, affinity_cache_values = request_affinity(
        verdict_file.parent / "requests.jsonl", log, state["profile"], workload, catalog
    )
    row.update(
        {
            "profile": state["profile"],
            "phase": state["phase"],
            "cache_mode": "hicache" if state["phase"] == "hicache" else "baseline",
            "workload": workload,
            "repetition": int(match[1]),
            **source_identity(state),
            "server_running_peak": max(valid_points, default=None),
            "server_samples": len(valid_points),
            "invalid_server_samples": len(points) - len(valid_points),
            "c40_longest_seconds": sustained,
            "c40_sustained_30s": sustained >= SUSTAINED_SECONDS,
            "pp_windows_40_advancing": progress_windows,
            "pp_progress_longest_seconds": progress_seconds,
            "pp_progress_sustained_30s": (
                progress_seconds >= SUSTAINED_SECONDS
                if progress_seconds is not None
                else None
            ),
            "dpa_snapshot_observation": dpa_observation,
            "capacity_qualified": sustained >= SUSTAINED_SECONDS
            and (state["profile"] != "pp2" or progress_seconds >= SUSTAINED_SECONDS),
            "metrics": safe_metrics(raw),
            "_node": state.get("node") if isinstance(state.get("node"), str) else None,
            "_campaign_run_id": state["run_id"],
            "dp_request_prefix_affinity": affinity["prefix_affinity_status"],
            "request_affinity": affinity,
            "_affinity_cache_values": affinity_cache_values,
            "dp_request_prefix_affinity_reason": (
                "Joined native finished rid to recorder response_id, and verified catalog request-body hash."
                if affinity["prefix_affinity_status"] == "observed"
                else "Complete native-rid, DP-rank and verified catalog coverage is required; load/cache aggregates cannot establish prefix placement."
            ),
        }
    )
    return row


def affinity_aggregate(group):
    valid = [
        row for row in group if row["functional_valid"] and row["orchestration_valid"]
    ]
    native = [
        row
        for row in valid
        if row["request_affinity"]["native_request_dp_status"] == "observed"
    ]
    prefixes = [
        row
        for row in valid
        if row["request_affinity"]["prefix_affinity_status"] == "observed"
    ]
    cached = [
        row
        for row in valid
        if row["request_affinity"]["native_cached_tokens_status"] == "observed"
    ]
    size = 1 if group[0]["profile"] == "pp2" else int(group[0]["profile"][3:])
    per_dp = []
    for rank in range(size):
        completed = [
            next(
                (
                    item["completed_requests_observed"]
                    for item in row["request_affinity"]["per_dp"]
                    if item["dp_rank"] == rank
                ),
                0,
            )
            for row in native
        ]
        values = [
            value
            for row in valid
            for value in row["_affinity_cache_values"].get(rank, [])
        ]
        per_dp.append(
            {
                "dp_rank": rank,
                "completed_requests_per_run": distribution(completed),
                "cached_tokens_observed_subset": cached_distribution(values),
            }
        )
    prefix_stats = {}
    for label in range(20):
        counts = [
            next(
                item["processing_dp_groups"]
                for item in row["request_affinity"]["prefix_groups"]
                if item["prefix_group"] == label
            )
            for row in prefixes
        ]
        if counts:
            prefix_stats[str(label)] = distribution(counts)
    return {
        "native_request_dp_observed_repetitions": len(native),
        "native_cached_tokens_observed_repetitions": len(cached),
        "prefix_affinity_observed_repetitions": len(prefixes),
        "all_native_request_dp_observed": len(native) == len(group),
        "all_native_cached_tokens_observed": len(cached) == len(group),
        "all_prefix_affinity_observed": len(prefixes) == len(group),
        "per_dp": per_dp,
        "per_prefix_processing_group_count_per_run": prefix_stats,
        "scope": "Per-run placement; processing groups are not unioned across repetitions to infer replication.",
    }


def prefix_placement_comparisons(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row["workload"].startswith("long"):
            key = (
                row["_campaign_run_id"],
                row["profile"],
                row["phase"],
                row["repetition"],
                row["source_commit"],
                row["image_digest"],
            )
            pairs[key][row["workload"]] = row
    comparisons = []
    for key, pair in sorted(pairs.items(), key=lambda item: str(item[0])):
        cold, warm = pair.get("long-cold"), pair.get("long-warm")
        entry = {
            "profile": key[1],
            "phase": key[2],
            "repetition": key[3],
            "source_commit": key[4],
            "image_digest": key[5],
            "status": "not_observable",
        }
        if all(
            row
            and row["functional_valid"]
            and row["orchestration_valid"]
            and row["request_affinity"]["prefix_affinity_status"] == "observed"
            for row in (cold, warm)
        ):

            def groups(row):
                return {
                    item["prefix_group"]: {
                        int(rank) for rank in item["per_dp_completed_requests"]
                    }
                    for item in row["request_affinity"]["prefix_groups"]
                }

            left, right = groups(cold), groups(warm)
            entries = [
                {
                    "prefix_group": label,
                    "cold_processing_dp_groups": sorted(left[label]),
                    "warm_processing_dp_groups": sorted(right[label]),
                    "new_processing_groups_in_warm": len(right[label] - left[label]),
                    "same_processing_group_set": left[label] == right[label],
                }
                for label in range(20)
            ]
            entry.update(
                status="observed",
                prefix_groups=entries,
                prefixes_with_changed_processing_groups=sum(
                    not item["same_processing_group_set"] for item in entries
                ),
                new_prefix_dp_pairs_in_warm=sum(
                    item["new_processing_groups_in_warm"] for item in entries
                ),
                scope="Verified same sampled prefixes; placement means observed processing assignment, not persistent cache residency.",
            )
        comparisons.append(entry)
    return comparisons


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["profile"],
            row["cache_mode"],
            row["workload"],
            row["source_commit"],
            row["image_digest"],
        )
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        valid = [
            row
            for row in group
            if row["functional_valid"] and row["orchestration_valid"]
        ]
        stats = {}
        for name in set().union(*(row["metrics"] for row in valid)):
            values = [row["metrics"][name] for row in valid if name in row["metrics"]]
            stats[name] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "samples": len(values),
            }
        fingerprints = [row["_dataset"] for row in valid]
        result.append(
            {
                "profile": key[0],
                "cache_mode": key[1],
                "workload": key[2],
                "source_commit": key[3],
                "image_digest": key[4],
                "repetitions": len(group),
                "valid_repetitions": len(valid),
                "all_functional_valid": len(valid) == len(group),
                "all_c40_sustained_30s": all(
                    row["capacity_qualified"] for row in group
                ),
                "dataset_consistent": bool(fingerprints)
                and None not in fingerprints
                and all(value == fingerprints[0] for value in fingerprints),
                "metrics": stats,
                "request_affinity": affinity_aggregate(group),
            }
        )
    return result


def dpa_selection(rows, aggregates, attempts):
    baseline = [
        row
        for row in rows
        if row["cache_mode"] == "baseline" and row["workload"].startswith("long")
    ]
    candidates = []
    for profile in ("dpa2", "dpa4", "dpa8"):
        groups = [
            group
            for group in aggregates
            if group["profile"] == profile
            and group["cache_mode"] == "baseline"
            and group["workload"].startswith("long")
        ]
        by_workload = {group["workload"]: group for group in groups}
        if len(groups) != 2 or set(by_workload) != {"long-cold", "long-warm"}:
            continue
        cold, warm = by_workload["long-cold"], by_workload["long-warm"]
        if not all(
            group["all_functional_valid"]
            and group["all_c40_sustained_30s"]
            and group["dataset_consistent"]
            and group["source_commit"]
            and group["image_digest"]
            and group["metrics"].get("output_throughput", {}).get("median", 0) > 0
            for group in (cold, warm)
        ):
            continue
        candidates.append(
            {
                "profile": profile,
                "cold_output_tok_s": cold["metrics"]["output_throughput"]["median"],
                "warm_output_tok_s": warm["metrics"]["output_throughput"]["median"],
                "minimum_repetitions": min(
                    cold["valid_repetitions"], warm["valid_repetitions"]
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            -min(row["cold_output_tok_s"], row["warm_output_tok_s"]),
            -row["cold_output_tok_s"],
            row["profile"],
        )
    )
    comparable_rows = [
        row for row in baseline if row["profile"] in {c["profile"] for c in candidates}
    ]
    nodes = {row["_node"] for row in comparable_rows}
    identities = {
        (row["source_commit"], row["image_digest"]) for row in comparable_rows
    }
    datasets = {row["_dataset"] for row in comparable_rows}
    comparable = (
        bool(comparable_rows)
        and len(nodes) == len(identities) == len(datasets) == 1
        and None not in nodes
        and None not in datasets
    )
    attempted = {
        attempt["profile"] for attempt in attempts if attempt["phase"] == "baseline"
    }
    all_screened = {"dpa2", "dpa4", "dpa8"} <= attempted
    candidate = (
        candidates[0]["profile"] if comparable and all_screened and candidates else None
    )
    return {
        "screening_candidate": candidate,
        "status": (
            "NO_MEASUREMENTS"
            if not baseline
            else (
                "PROVISIONAL_SCREENING"
                if candidate
                else "INSUFFICIENT_QUALIFIED_COMPARABLE_EVIDENCE"
            )
        ),
        "all_dpa_profiles_attempted": all_screened,
        "same_node_source_image_dataset": comparable,
        "ranking": candidates,
        "ranking_rule": "Maximize the smaller cold/warm median output throughput; cold throughput breaks ties.",
        "final_recommendation_ready": bool(
            candidate and candidates[0]["minimum_repetitions"] >= 3
        ),
        "limitations": [
            "Screening selection is provisional until finalist repetitions complete.",
            "Latency distributions and throughput spread must accompany any performance recommendation.",
            "No numerical TTFT/ITL SLO was specified; throughput ranking does not prove latency acceptability.",
        ],
    }


def analyze(roots, catalog_path=None):
    rows, attempts, issues = [], [], Counter()
    catalog = load_catalog(catalog_path)
    seen_states = set()
    for root in map(Path, roots):
        for state_file in sorted(root.glob("*/run-state.json")):
            if state_file.resolve() in seen_states:
                continue
            seen_states.add(state_file.resolve())
            state = read_json(state_file)
            if state.get("profile") not in PROFILES or state.get("phase") not in PHASES:
                issues["invalid_run_state"] += 1
                continue
            run_id = state.get("run_id")
            if not isinstance(run_id, str) or run_id != state_file.parent.name:
                issues["invalid_run_id"] += 1
                continue
            attempt = {
                "profile": state["profile"],
                "phase": state["phase"],
                "status": (
                    state.get("status")
                    if state.get("status") in {"measured", "failed", "benchmark_failed"}
                    else "unknown"
                ),
                "export_failed": bool(state.get("export_error")),
                "cleanup_failed": bool(state.get("cleanup_error")),
            }
            attempts.append(attempt)
            result = root / "results" / run_id
            log_path = state_file.parent / "server.log"
            log = log_path.read_text(errors="replace") if log_path.is_file() else ""
            verdicts = sorted(result.glob("r*/verdict.json"))
            if not verdicts:
                issues["attempt_without_workload_verdict"] += 1
            for verdict_file in verdicts:
                row = summarize_workload(state, verdict_file, log, catalog)
                if row is None:
                    issues["invalid_workload_verdict"] += 1
                else:
                    rows.append(row)
    aggregates = aggregate(rows)
    selection = dpa_selection(rows, aggregates, attempts)
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    return {
        "schema_version": 3,
        "runs": public_rows,
        "aggregates": aggregates,
        "catalog_status": catalog["status"],
        "prefix_placement_comparisons": prefix_placement_comparisons(rows),
        "attempts": attempts,
        "failed_profiles": [row for row in attempts if row["status"] != "measured"],
        "evidence_issues": dict(issues),
        "dpa_selection": selection,
        "c40_rule": {
            "minimum_seconds": SUSTAINED_SECONDS,
            "maximum_observation_gap_seconds": MAX_GAP,
            "pp": "Unique active request IDs on PP0/TP0; additionally 40 common IDs must advance output tokens in consecutive windows.",
            "dpa": "One fresh /v1/loads row for every distinct DP rank; each rank timestamp must advance. The configured interval counts decode iterations, not seconds.",
            "invalid_sample": "Breaks sustained windows; a client concurrency peak never proves server C40.",
        },
        "privacy": "Only enumerated labels, counts, allowlisted numeric metrics, source commit and image digest are emitted. Raw infrastructure, request and error metadata remain local.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--catalog", help="Optional locally verified metadata-only dataset catalog"
    )
    args = parser.parse_args()
    output = analyze(args.root, args.catalog)
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "measured_workloads": len(output["runs"]),
                "aggregate_groups": len(output["aggregates"]),
                "selection_status": output["dpa_selection"]["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
