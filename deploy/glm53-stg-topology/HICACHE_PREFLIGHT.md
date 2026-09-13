# HiCache RAM preflight

This separate gate must pass immediately before a HiCache serving apply.
It does not change the frozen benchmark/orchestrator, allocate GPUs, or establish
HiCache compatibility or performance. The operator retains the existing single
serving deployment and GPU cleanup checks.

## Evidence and scope

First collect completed baseline artifacts for the profiles being compared. The
helper reads `run-state.json`, the independent serving Pod snapshot, completed
admission/preparation/measurement Job snapshots, catalog/model hashes, and binary
follow logs through the existing `verified_follow_log` verifier. It rejects an
incomplete baseline, identity/hash mismatch, absent counter, uncovered stage, or
telemetry gap over 15 seconds. This coverage limit is an operational policy; it
does not interpolate memory values between samples. Replayed identical samples
are deduplicated by timestamp, and conflicting samples are rejected.

`observed_peak_bytes` is the maximum **sampled serving `memory.current`** over every
captured baseline record, including preparation and captured initialization.
It is not a kernel high-water mark. A separately reported serving `memory.peak`,
when available, increases the budget. Missing kernel peak is `not_observable`;
missing observed peak blocks admission. The last current reading, counts, stage
coverage, and evidence hashes are retained. GPU identities, raw node names,
Pod UIDs, prompts, request IDs, and private paths are not copied into this summary.
Each stage must fit inside one verified, closed follow segment. Reconnects are
never joined to cover a stage. A later empty or malformed segment does not
invalidate a previously covered stage, and its samples are not used. The maximum
uses only verified segments that individually cover at least one required stage.

All included baselines must match the planned serving source, immutable image,
model config, and selected node. `--profiles` explicitly identifies the required
baseline profiles; every matching attempt in the supplied campaign roots is
checked, not selected by its memory or performance result. The default requires
PP2/DPA2/DPA4/DPA8. A failed or interrupted attempt remains evidence and blocks this
summary; use the completed comparison campaign after resolving that failure.

The CPU Job samples **whole-host `MemAvailable`** from `/proc/meminfo` and checks
its actual node via Downward API. It reads the mounted model config again and
compares its SHA256. It never reads its own cgroup counters. The serving 640 GiB
limit comes from the exact reviewed serving manifest. The Job requests one CPU
and 256 MiB, mounts only the existing model PVC read-only and an immutable
ConfigMap, and emits JSON to ordinary container logs. It has no service account
token, GPU request, Service, or results PVC mount.

## Exact allocator arithmetic

The reviewed source is upstream `0bcd822377da7b5718e674eaf9c870d349424dd1`.
The helper pins six allocator source hashes; the renderer verifies them against
Git and requires the serving revision's installer manifest to verify them too.
Relevant source locations are:

- `python/sglang/srt/mem_cache/pool_host/base.py:26`: native 10 GiB OS reserve;
  `:32`: ranks per host; `:51`: `(current available - reserve) // ranks`;
  `:149`: fixed decimal GB and page alignment; `:173`: anchor admission check.
- `python/sglang/srt/mem_cache/kv_cache_configurator.py:2410`: scaled FP8 DSA
  stores 512 KV bytes, four 4-byte scales, and 64 BF16 RoPE elements: 656 bytes
  per token per layer.
- `python/sglang/srt/mem_cache/memory_pool.py:4397`: index block 128 and uint8
  storage; `pool_host/dsa.py:76` uses 128 + 4 = 132 indexer bytes per token/layer.
- `pool_host/mla.py:125`: anchor bytes/token/layer and local layer count;
  `pool_host/dsa.py:87` adds a page to the actual indexer allocation, while
  `:95` checks the one-page-smaller anchor slot count; `:148` allocates it.
- `hybrid_cache/hybrid_pool_assembler.py:932`: each rank constructs its anchor
  before its indexer; `:1940` connects the DSA KV + INDEXER stack to HiCache.

For page 64, 32 decimal GB per rank, eight ranks, no MTP or DCP:

| Profile | Local layers | Host tokens/rank | Anchor bytes/rank | Indexer allocated bytes/rank | Total allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| PP2 | 39 | 1,250,816 | 32,000,876,544 | 6,439,530,240 | 307,523,254,272 |
| DPA2/4/8 | 78 | 625,408 | 32,000,876,544 | 6,439,859,712 | 307,525,890,048 |

Thus 256 decimal GB is the requested anchor budget only. Actual anchor + indexer
payload buffers occupy about **286.4 GiB**; allocator/process overhead is separate.
The gate assumes `flashmla_kv`, FP8 KV, `layer_first/direct/write_through`, cache
mode, no CP/DCP/MTP/L3, and PP 39/39 or the full 78 layers for DPA.

Let `N=8`, `A=anchor bytes/rank`, `Ic=indexer checked bytes/rank`, and
`Ia=indexer actual bytes/rank`. Each rank checks memory separately before each
allocation. For a conservative bound over all legal rank interleavings:

```text
last_anchor_floor  = (N-1)*(A+Ia) + N*A
last_indexer_floor = N*A + (N-1)*Ia + N*Ic
native_floor       = max(last_anchor_floor, last_indexer_floor)
cache_allocation   = N*(A+Ia)
B                  = max(observed baseline peak, available kernel peak)
R                  = 32 GiB operational margin

configured cgroup condition: B + cache_allocation + R <= 640 GiB
whole-host condition:        B + R + native_floor + 10 GiB <= fresh MemAvailable
```

The first native term dominates here (about 489.03 GiB). It accounts for other
ranks already consuming memory before the last rank's repeatedly divided check.
It is an **allocation-order admission bound**, not actual cache allocation or a
claim that every startup needs that much free RAM. It conservatively assumes
prior buffers reduce available memory by their full size and reserves `B` again
for serving startup. The explicit 32 GiB margin is operational policy, not an
upstream setting or guarantee that an observed peak captures every future peak.
A synthetic exhaustive-interleaving test checks the formula including the
indexer's different checked/allocated sizes.

## Commands

Use a committed checkout and private output paths. These helpers never apply
resources. The example variables denote operator-supplied paths and identities.

```bash
python deploy/glm53-stg-topology/hicache_preflight.py summarize \
  --campaign-root "$BASELINE_CAMPAIGN" \
  --profiles pp2 dpa2 dpa4 dpa8 \
  --output "$PRIVATE_BASELINE_SUMMARY"

python deploy/glm53-stg-topology/render_hicache_preflight.py \
  --run-id "hicache-preflight-$UNIQUE_ID" \
  --node "$VERIFIED_NODE" \
  --tooling-commit "$FULL_TOOLING_COMMIT" \
  --baseline "$PRIVATE_BASELINE_SUMMARY" \
  --serving-manifest "$PRIVATE_REVIEWED_HICACHE_MANIFEST" \
  > "$PRIVATE_RAM_GATE_MANIFEST"
```

Run that CPU Job through the existing authorized Kubernetes workflow and save its
single JSON log report privately. Baseline evidence is historical by design; its
age is recorded and its source/image/model/node must still match. The fresh host
snapshot/report is valid for 120 seconds. Immediately before applying the exact
reviewed serving manifest, verify the report using the same helper revision:

```bash
python deploy/glm53-stg-topology/hicache_preflight.py verify-report \
  --report "$PRIVATE_RAM_REPORT" \
  --serving-manifest "$PRIVATE_REVIEWED_HICACHE_MANIFEST" \
  --node "$VERIFIED_NODE"
```

Any nonzero exit or `BLOCKED` status stops admission. Expired reports require a
new uniquely named CPU Job. Source/image, model,
node, manifest, or helper changes invalidate the corresponding evidence. This
check cannot reserve host memory against concurrent workloads. Continue checking
actual serving allocation, cgroup counters, completed smoke, cache transfers and
hits, and workload evidence after startup. The existing asynchronous HiCache
flush/idle behavior may independently fail a smoke or suite; RAM PASS does not
turn such a failure into a valid measurement.
