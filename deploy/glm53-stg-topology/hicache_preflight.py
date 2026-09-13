"""Read-only HiCache RAM admission: evidence summary locally, fresh host check in a CPU Job."""

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

GIB = 1024**3
UPSTREAM_COMMIT = "0bcd822377da7b5718e674eaf9c870d349424dd1"
SOURCE_SHA256 = {
    "pool_host/base.py": "14498937a59d3747028a6250840ae9d8f5f5f960da2731083af2fd801d7f795b",
    "pool_host/dsa.py": "aa408d25423fb3ecc30c6a0392a6550f520904634d66d00ca7cbdd9096a7d9c8",
    "pool_host/mla.py": "b0aa464a0e0b9e3a10f92dbaf3dba1d198839997c69a2802acaa280de05cbb7e",
    "hybrid_cache/hybrid_pool_assembler.py": "739cf020ceaffe6698c3b4e92044606f87263f6ab92bcefd811b294cf800ef7b",
    "kv_cache_configurator.py": "40a4d40ba17dbd5065048a96077484b3158a26c00638a087c4678569597f5f04",
    "memory_pool.py": "7955db1470ea36b99eb43483a5bd999dbda62119ff54aee7f9e8e01f31b2a9b5",
}
PROFILES = ("pp2", "dpa2", "dpa4", "dpa8")
SERVING_LIMIT = 640 * GIB
OPERATIONAL_MARGIN = 32 * GIB
NATIVE_RESERVE = 10 * GIB
HOST_VALID_SECONDS = 120
MAX_SAMPLE_GAP_SECONDS = 15
MAX_JSON_BYTES = 4 * 1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def positive_int(value):
    return type(value) is int and value > 0


def read_json(path):
    require(path.stat().st_size <= MAX_JSON_BYTES, "json_size_limit")
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def timestamp(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(date.tzinfo is not None, "timestamp_timezone_missing")
    return date.timestamp()


def memory_formula(profile, config):
    """Exact fixed-size/page arithmetic; native floor is an allocation-order bound."""
    require(profile in PROFILES, "unsupported_profile")
    expected = {
        "num_hidden_layers": 78,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
    }
    require(
        all(config.get(k) == v for k, v in expected.items()), "model_dimensions_changed"
    )
    require(
        config.get("architectures") == ["GlmMoeDsaForCausalLM"],
        "model_architecture_changed",
    )
    quant = config.get("quantization_config", {})
    require(
        quant.get("quant_method") == "w4afp8" and quant.get("group_size") == 128,
        "model_quantization_changed",
    )
    layers, ranks, page = (39 if profile == "pp2" else 78), 8, 64
    # Scaled FP8 MLA: 512 bytes + four FP32 scales + 64 BF16 RoPE elements.
    kv_bytes = (
        config["kv_lora_rank"]
        + config["kv_lora_rank"] // 128 * 4
        + config["qk_rope_head_dim"] * 2
    )
    index_bytes = config["index_head_dim"] + config["index_head_dim"] // 128 * 4
    tokens = (32_000_000_000 // (kv_bytes * layers) // page + 1) * page
    anchor = tokens * kv_bytes * layers
    index_check = tokens * index_bytes * layers
    index_actual = ((tokens + page + 1) // page) * page * index_bytes * layers
    # Each rank checks anchor, allocates it, then checks/allocates indexer.
    # Max prior allocations before a last anchor: other N-1 complete ranks.
    # Before a last indexer: all N anchors plus N-1 complete indexers.
    last_anchor_floor = (ranks - 1) * (anchor + index_actual) + ranks * anchor
    last_indexer_floor = (
        ranks * anchor + (ranks - 1) * index_actual + ranks * index_check
    )
    return {
        "ranks_per_host": ranks,
        "layers_per_rank": layers,
        "page_size": page,
        "anchor_requested_decimal_gb_per_rank": 32,
        "host_tokens_per_rank": tokens,
        "anchor_bytes_per_rank": anchor,
        "indexer_check_bytes_per_rank": index_check,
        "indexer_allocation_bytes_per_rank": index_actual,
        "total_cache_allocation_bytes": ranks * (anchor + index_actual),
        "native_last_anchor_floor_bytes": last_anchor_floor,
        "native_last_indexer_floor_bytes": last_indexer_floor,
        "native_allocation_order_floor_bytes": max(
            last_anchor_floor, last_indexer_floor
        ),
        "native_host_reserve_bytes": NATIVE_RESERVE,
        "scope": "FP8 flashmla_kv DSA, page64, layer_first/direct/write_through, cache mode, no CP/DCP/MTP/L3; payload buffers only",
    }


def summarize_samples(texts, stages):
    """Ignore GPU identities/rows; retain only serving cgroup counters and coverage."""
    samples = {}
    for text in texts:
        for line in text.split("\n"):
            # Exact LF record payload: never split embedded carriage returns.
            match = re.match(r"^\S+ GLM53_GPU (\{.*\})$", line)
            if not match:
                continue
            point = json.loads(match[1])
            t, current = point.get("timestamp"), point.get("cgroup_memory_bytes")
            require(
                number(t) and positive_int(current), "cgroup_current_not_observable"
            )
            require(
                abs(timestamp(line.split(" ", 1)[0]) - t) <= MAX_SAMPLE_GAP_SECONDS,
                "telemetry_clock_mismatch",
            )
            errors = point.get("cgroup_memory_counter_errors", {})
            require(
                not errors.get("cgroup_memory_bytes"), "cgroup_current_counter_error"
            )
            peak = point.get("cgroup_memory_peak_bytes")
            if peak is not None:
                require(
                    positive_int(peak)
                    and peak >= current
                    and not errors.get("cgroup_memory_peak_bytes"),
                    "kernel_peak_invalid",
                )
            limit = point.get("cgroup_memory_limit_bytes")
            if limit is not None:
                require(
                    limit == SERVING_LIMIT
                    and not errors.get("cgroup_memory_limit_bytes"),
                    "observed_serving_limit_mismatch",
                )
            row = (current, peak, limit)
            require(
                t not in samples or samples[t] == row, "conflicting_replayed_telemetry"
            )
            samples[t] = row
    times = sorted(samples)
    require(bool(times), "cgroup_samples_missing")
    coverage = []
    for stage in stages:
        begin, end = stage["started_at"], stage["finished_at"]
        before = [t for t in times if t <= begin]
        after = [t for t in times if t >= end]
        require(before and after, "baseline_stage_not_bracketed")
        selected = [before[-1], *[t for t in times if begin < t < end], after[0]]
        gap = max((b - a for a, b in zip(selected, selected[1:])), default=0)
        require(gap <= MAX_SAMPLE_GAP_SECONDS, "baseline_stage_telemetry_gap")
        coverage.append(
            {**stage, "sample_count": len(set(selected)), "max_sample_gap_seconds": gap}
        )
    peaks = [row[1] for row in samples.values() if row[1] is not None]
    return {
        "sample_count": len(times),
        "first_sample_at": times[0],
        "last_sample_at": times[-1],
        "last_current_bytes": samples[times[-1]][0],
        "observed_peak_bytes": max(row[0] for row in samples.values()),
        "kernel_peak_bytes": max(peaks) if peaks else None,
        "kernel_peak_status": "observed" if peaks else "not_observable",
        "kernel_peak_sample_count": len(peaks),
        "limit_sample_count": sum(row[2] is not None for row in samples.values()),
        "stages": coverage,
    }


def summarize_run(directory, results):
    # This verifier stays local. The CPU Job receives only the compact summary.
    import analyze_results

    state, state_hash = read_json(directory / "run-state.json")
    require(
        state.get("phase") == "baseline" and state.get("status") == "measured",
        "baseline_not_complete",
    )
    require(
        state.get("job_success") is True
        and state.get("preparation_job_success") is True
        and not state.get("preparation_skipped"),
        "baseline_stages_not_complete",
    )
    require(
        not state.get("cleanup_error") and not state.get("final_collection_error"),
        "baseline_cleanup_or_collection_error",
    )
    require(state.get("profile") in PROFILES, "baseline_profile_invalid")
    require(
        re.fullmatch(r"[0-9a-f]{40}", state.get("source_commit", "")),
        "baseline_source_missing",
    )
    require(
        re.fullmatch(
            r"ghcr\.io/leoong55/sglang-lasolovev@sha256:[0-9a-f]{64}",
            state.get("image", ""),
        ),
        "baseline_image_missing",
    )
    require(
        isinstance(state.get("node"), str) and state["node"], "baseline_node_missing"
    )
    pods = []
    for path in sorted(directory.glob("glm53-topology-serving-*.json")):
        pod, pod_hash = read_json(path)
        if pod.get("kind") == "Pod" and pod.get("metadata", {}).get("uid") == state.get(
            "pod_uid"
        ):
            pods.append((pod, pod_hash))
    require(len(pods) == 1, "independent_serving_pod_snapshot_missing")
    pod, pod_hash = pods[0]
    require(pod["spec"].get("nodeName") == state["node"], "baseline_node_mismatch")
    container = next(c for c in pod["spec"]["containers"] if c["name"] == "sglang")
    status = next(
        c for c in pod["status"]["containerStatuses"] if c["name"] == "sglang"
    )
    require(
        container["image"] == state["image"] and status["restartCount"] == 0,
        "baseline_container_mismatch",
    )
    require(
        container["resources"]["limits"]["memory"] == "640Gi",
        "baseline_cgroup_budget_mismatch",
    )
    require("--hicache" not in container.get("args", []), "baseline_hicache_enabled")
    require(
        next(e.get("value") for e in container["env"] if e["name"] == "SOURCE_COMMIT")
        == state["source_commit"],
        "baseline_pod_source_mismatch",
    )
    meta, meta_hash = read_json(directory / "server-follow.meta.json")
    require(
        meta.get("source_commit") == state["source_commit"]
        and meta.get("container_id") == status.get("containerID")
        and meta.get("image_id") == status.get("imageID"),
        "baseline_log_identity_mismatch",
    )
    require(
        status.get("imageID", "").endswith(state["image"].split("@", 1)[1]),
        "baseline_image_id_mismatch",
    )
    require(
        (directory / "server-follow.log").stat().st_size <= 2 * 1024**3,
        "baseline_log_size_limit",
    )
    texts, proof = analyze_results.verified_follow_log(directory, state)
    require(proof["status"] == "verified", "baseline_log_not_verified")
    require(
        all(s["ended_at"] is not None for s in proof["segments"]),
        "baseline_collector_still_running",
    )
    stages, model_hashes, artifact_hashes = [], set(), {}
    selected_segments = set()
    run_id = state["run_id"]
    for stage, suffix, subdir, purpose in (
        ("admission", "-smoke", "admission", "admission"),
        ("preparation", "-prep", "preparation", "preparation"),
        ("measurement", "", "", "measurement"),
    ):
        job, job_hash = read_json(directory / f"glm53-{run_id}{suffix}.json")
        require(
            job.get("kind") == "Job"
            and any(
                c.get("type") == "Complete" and c.get("status") == "True"
                for c in job.get("status", {}).get("conditions", [])
            ),
            "baseline_job_not_complete",
        )
        begin, end = timestamp(job["status"]["startTime"]), timestamp(
            job["status"]["completionTime"]
        )
        require(begin < end, "baseline_stage_times_invalid")
        covering = [
            index
            for index, segment in enumerate(proof["segments"])
            if segment["status"] == "verified"
            and segment["started_at"] <= begin <= end <= segment["ended_at"]
            and segment["first_cri_timestamp"]
            <= begin
            <= end
            <= segment["last_cri_timestamp"]
        ]
        require(covering, "baseline_stage_stream_not_continuous")
        selected_segments.update(covering)
        stages.append(
            {
                "stage": stage,
                "started_at": begin,
                "finished_at": end,
                "job_sha256": job_hash,
            }
        )
        result_dir = results / subdir
        provenance, provenance_hash = read_json(result_dir / "provenance.json")
        require(
            provenance.get("purpose") == purpose
            and provenance.get("profile") == state["profile"]
            and provenance.get("source_commit") == state["source_commit"]
            and provenance.get("serving_image_digest") == state["image"],
            "baseline_stage_provenance_mismatch",
        )
        # Admission intentionally has no expensive dataset catalog.
        if stage != "admission":
            catalog, catalog_hash = read_json(result_dir / "dataset-catalog.json")
            require(
                provenance.get("dataset_catalog_sha256") == catalog_hash,
                "baseline_catalog_hash_mismatch",
            )
            model_hashes.add(catalog["model_metadata"]["files"]["config.json"])
            artifact_hashes[stage + "_catalog_sha256"] = catalog_hash
        artifact_hashes[stage + "_provenance_sha256"] = provenance_hash
    require(
        len(model_hashes) == 1
        and re.fullmatch(r"[0-9a-f]{64}", next(iter(model_hashes))),
        "baseline_model_hash_mismatch",
    )
    return {
        "profile": state["profile"],
        "node_sha256": digest(state["node"].encode()),
        "pod_uid_sha256": digest(state["pod_uid"].encode()),
        "source_commit": state["source_commit"],
        "serving_image_digest": state["image"],
        "model_config_sha256": next(iter(model_hashes)),
        "run_state_sha256": state_hash,
        "pod_snapshot_sha256": pod_hash,
        "log_metadata_sha256": meta_hash,
        "log_sha256": meta["log_sha256"],
        "collector_sha256": meta["collector_sha256"],
        "verifier_sha256": digest(Path(analyze_results.__file__).read_bytes()),
        **artifact_hashes,
        "selected_follow_segments": sorted(selected_segments),
        **summarize_samples([texts[i] for i in sorted(selected_segments)], stages),
    }


def summarize(campaign_roots, profiles):
    require(
        profiles
        and len(set(profiles)) == len(profiles)
        and set(profiles) <= set(PROFILES),
        "baseline_profiles_invalid",
    )
    rows = []
    for root in campaign_roots:
        for state_path in sorted(root.glob("*/run-state.json")):
            state, _ = read_json(state_path)
            if state.get("phase") == "baseline" and state.get("profile") in profiles:
                rows.append(
                    summarize_run(state_path.parent, root / "results" / state["run_id"])
                )
    require(
        {row["profile"] for row in rows} == set(profiles),
        "required_baseline_profile_missing",
    )
    identities = {
        (
            r["node_sha256"],
            r["source_commit"],
            r["serving_image_digest"],
            r["model_config_sha256"],
        )
        for r in rows
    }
    require(len(identities) == 1, "baseline_identities_not_comparable")
    identity = {
        k: rows[0][k]
        for k in (
            "node_sha256",
            "source_commit",
            "serving_image_digest",
            "model_config_sha256",
        )
    }
    return {
        "schema_version": 1,
        "purpose": "hicache_ram_baseline_evidence",
        "status": "verified",
        "created_at": time.time(),
        "helper_sha256": digest(Path(__file__).read_bytes()),
        "required_profiles": sorted(profiles),
        **identity,
        "profiles": rows,
        "observed_peak_scope": "Maximum sampled serving cgroup memory.current across all captured baseline records, including preparation; not kernel high-water mark",
    }


def host_snapshot(node, meminfo, sampled_at):
    values = {}
    for key in ("MemTotal", "MemAvailable"):
        match = re.search(rf"^{key}:\s+(\d+)\s+kB$", meminfo, re.MULTILINE)
        require(match is not None, "host_meminfo_not_observable")
        values[key] = int(match[1]) * 1024
    require(0 < values["MemAvailable"] <= values["MemTotal"], "host_meminfo_invalid")
    require(bool(node), "selected_node_not_observable")
    return {
        "sampled_at": sampled_at,
        "node_sha256": digest(node.encode()),
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "scope": "whole-host /proc/meminfo; CPU Job cgroup is never used",
    }


def evaluate(baseline, plan, config_raw, host, now):
    result = {
        "schema_version": 1,
        "purpose": "hicache_ram_preflight",
        "status": "BLOCKED",
        "checked_at": now,
        "reasons": [],
        "helper_sha256": digest(Path(__file__).read_bytes()),
    }
    try:
        require(
            baseline.get("schema_version") == 1
            and baseline.get("purpose") == "hicache_ram_baseline_evidence"
            and baseline.get("status") == "verified",
            "baseline_evidence_missing",
        )
        require(
            baseline.get("helper_sha256") == result["helper_sha256"],
            "baseline_helper_revision_mismatch",
        )
        require(
            plan.get("upstream_commit") == UPSTREAM_COMMIT
            and plan.get("upstream_source_sha256") == SOURCE_SHA256,
            "allocator_source_not_verified",
        )
        require(
            plan.get("serving_limit_bytes") == SERVING_LIMIT,
            "planned_cgroup_limit_changed",
        )
        require(
            plan.get("operational_margin_bytes") == OPERATIONAL_MARGIN,
            "operational_margin_changed",
        )
        for key in (
            "node_sha256",
            "source_commit",
            "serving_image_digest",
            "model_config_sha256",
        ):
            require(
                isinstance(plan.get(key), str)
                and plan[key]
                and baseline.get(key) == plan[key],
                "baseline_" + key + "_mismatch",
            )
        require(host["node_sha256"] == plan["node_sha256"], "selected_node_mismatch")
        require(
            number(host.get("sampled_at"))
            and 0 <= now - host["sampled_at"] <= HOST_VALID_SECONDS,
            "host_snapshot_stale",
        )
        require(
            positive_int(host.get("mem_available_bytes"))
            and positive_int(host.get("mem_total_bytes"))
            and host["mem_available_bytes"] <= host["mem_total_bytes"],
            "host_memory_missing",
        )
        require(
            digest(config_raw) == plan["model_config_sha256"],
            "mounted_model_config_mismatch",
        )
        formula = memory_formula(plan["profile"], json.loads(config_raw))
        rows = baseline.get("profiles", [])
        require(
            rows
            and set(baseline.get("required_profiles", []))
            == {r["profile"] for r in rows}
            and plan["profile"] in {r["profile"] for r in rows},
            "baseline_profiles_incomplete",
        )
        for row in rows:
            require(
                all(
                    row.get(k) == plan[k]
                    for k in (
                        "node_sha256",
                        "source_commit",
                        "serving_image_digest",
                        "model_config_sha256",
                    )
                ),
                "baseline_row_identity_mismatch",
            )
            require(
                positive_int(row.get("observed_peak_bytes"))
                and positive_int(row.get("last_current_bytes"))
                and row["observed_peak_bytes"] >= row["last_current_bytes"]
                and positive_int(row.get("sample_count")),
                "baseline_observed_peak_missing",
            )
            require(
                number(row.get("last_sample_at")) and 0 <= now - row["last_sample_at"],
                "baseline_sample_time_invalid",
            )
            require(
                {s["stage"] for s in row.get("stages", [])}
                == {"admission", "preparation", "measurement"},
                "baseline_stage_coverage_missing",
            )
            for stage in row["stages"]:
                require(
                    positive_int(stage.get("sample_count"))
                    and number(stage.get("max_sample_gap_seconds"))
                    and 0 <= stage["max_sample_gap_seconds"] <= MAX_SAMPLE_GAP_SECONDS,
                    "baseline_stage_coverage_invalid",
                )
            if row.get("kernel_peak_bytes") is not None:
                require(
                    positive_int(row["kernel_peak_bytes"])
                    and row["kernel_peak_bytes"] >= row["observed_peak_bytes"],
                    "baseline_kernel_peak_invalid",
                )
        observed = max(r["observed_peak_bytes"] for r in rows)
        kernel = max(
            (
                r["kernel_peak_bytes"]
                for r in rows
                if r.get("kernel_peak_bytes") is not None
            ),
            default=None,
        )
        baseline_bytes = max(observed, kernel or 0)
        cgroup_required = (
            baseline_bytes
            + formula["total_cache_allocation_bytes"]
            + OPERATIONAL_MARGIN
        )
        host_required = (
            baseline_bytes
            + OPERATIONAL_MARGIN
            + formula["native_allocation_order_floor_bytes"]
            + NATIVE_RESERVE
        )
        result.update(
            formula=formula,
            observed_baseline_peak_bytes=observed,
            optional_kernel_peak_bytes=kernel,
            baseline_budget_bytes=baseline_bytes,
            oldest_baseline_last_sample_age_seconds=max(
                now - r["last_sample_at"] for r in rows
            ),
            configured_cgroup_budget={
                "limit_bytes": SERVING_LIMIT,
                "required_bytes": cgroup_required,
                "headroom_bytes": SERVING_LIMIT - cgroup_required,
            },
            conservative_native_admission={
                "available_bytes": host["mem_available_bytes"],
                "required_bytes": host_required,
                "headroom_bytes": host["mem_available_bytes"] - host_required,
            },
            operational_margin_bytes=OPERATIONAL_MARGIN,
            host=host,
            plan_sha256=digest(encoded(plan)),
            baseline_sha256=digest(encoded(baseline)),
            tooling_commit=plan["tooling_commit"],
            source_commit=plan["source_commit"],
            serving_image_digest=plan["serving_image_digest"],
            serving_manifest_sha256=plan["serving_manifest_sha256"],
            upstream_commit=UPSTREAM_COMMIT,
            upstream_source_sha256=SOURCE_SHA256,
            model_config_sha256=plan["model_config_sha256"],
            profile=plan["profile"],
            valid_until=host["sampled_at"] + HOST_VALID_SECONDS,
            limitations=[
                "Observed peak is sampled, not proof of the true maximum; 32Gi is an explicit operational margin",
                "Native floor assumes all allocations resident and the worst valid rank interleaving; it is not actual buffer size",
                "Host memory can change after this snapshot; recheck immediately before serving apply",
                "RAM PASS does not prove HiCache compatibility, cache hits, C40 or performance",
            ],
        )
        if cgroup_required > SERVING_LIMIT:
            result["reasons"].append("configured_serving_cgroup_budget_insufficient")
        if host_required > host["mem_available_bytes"]:
            result["reasons"].append("conservative_native_host_floor_insufficient")
        result["status"] = "BLOCKED" if result["reasons"] else "PASS"
    except (ValueError, KeyError, TypeError) as error:
        result["reasons"].append(
            str(error)
            if isinstance(error, ValueError)
            else "required_evidence_field_missing_or_invalid"
        )
    return result


def verify_report(report, manifest_raw, node, now):
    """Consume the report just before an independently managed serving apply."""
    require(
        report.get("schema_version") == 1
        and report.get("purpose") == "hicache_ram_preflight"
        and report.get("status") == "PASS"
        and report.get("reasons") == [],
        "ram_gate_not_passed",
    )
    require(
        report.get("helper_sha256") == digest(Path(__file__).read_bytes()),
        "ram_gate_helper_revision_mismatch",
    )
    require(
        number(report.get("checked_at"))
        and number(report.get("valid_until"))
        and report["checked_at"] <= now <= report["valid_until"],
        "ram_gate_report_expired",
    )
    require(
        report.get("serving_manifest_sha256") == digest(manifest_raw),
        "ram_gate_serving_manifest_changed",
    )
    require(
        report.get("host", {}).get("node_sha256") == digest(node.encode()),
        "ram_gate_selected_node_changed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--campaign-root", type=Path, action="append", required=True)
    summary.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=list(PROFILES)
    )
    summary.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--input-directory", type=Path, default=Path("/scripts"))
    check.add_argument("--model-config", type=Path, default=Path("/model/config.json"))
    consume = commands.add_parser("verify-report")
    consume.add_argument("--report", type=Path, required=True)
    consume.add_argument("--serving-manifest", type=Path, required=True)
    consume.add_argument("--node", required=True)
    args = parser.parse_args()
    try:
        if args.command == "summarize":
            result = summarize(args.campaign_root, args.profiles)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2)
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "profiles": args.profiles,
                        "summary_sha256": digest(args.output.read_bytes()),
                    }
                )
            )
            return 0
        if args.command == "verify-report":
            report, _ = read_json(args.report)
            require(
                args.serving_manifest.stat().st_size <= MAX_JSON_BYTES,
                "serving_manifest_size_limit",
            )
            verify_report(
                report, args.serving_manifest.read_bytes(), args.node, time.time()
            )
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope": "fresh RAM report, same manifest and node",
                    }
                )
            )
            return 0
        data = {
            name: (args.input_directory / name).read_text()
            for name in ("hicache_preflight.py", "baseline.json", "plan.json")
        }
        require(
            digest(encoded(data)) == os.environ.get("SCRIPT_CONFIG_SHA256"),
            "immutable_configmap_hash_mismatch",
        )
        require(
            data["hicache_preflight.py"].encode() == Path(__file__).read_bytes(),
            "executed_helper_mismatch",
        )
        plan, baseline = json.loads(data["plan.json"]), json.loads(
            data["baseline.json"]
        )
        require(
            args.model_config.stat().st_size <= MAX_JSON_BYTES,
            "model_config_size_limit",
        )
        host = host_snapshot(
            os.environ.get("NODE_NAME"), Path("/proc/meminfo").read_text(), time.time()
        )
        result = evaluate(
            baseline, plan, args.model_config.read_bytes(), host, time.time()
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        result = {
            "schema_version": 1,
            "purpose": "hicache_ram_preflight",
            "status": "BLOCKED",
            "reasons": [
                str(error) if isinstance(error, ValueError) else type(error).__name__
            ],
            "helper_sha256": digest(Path(__file__).read_bytes()),
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
