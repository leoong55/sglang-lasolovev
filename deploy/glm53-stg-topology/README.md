# GLM-5.3 STG: PP2 and DP-attention on eight H200

This experiment compares full `PhalaCloud/GLM-5.3-W4AFP8` with PP2 and
DPA2/DPA4/DPA8 in namespace `inf-glm53`. Long C40 traffic is primary; short
traffic diagnoses decode. It does not change the independent CP/DCP work.
Only ClusterIP is exposed. There is no Envoy, ingress, or public serving API.

The runtime is pinned to upstream
`0bcd822377da7b5718e674eaf9c870d349424dd1`. The installer verifies critical
runtime files against this commit and refuses a different base; it does not
patch scheduling, attention, quantization, or cache policy. The PP and DPA
observers read CPU request metadata without changing scheduler decisions.

## Profiles

| Profile | TP | PP | DP attention | EP | Attention TP | Local running limit |
|---|---:|---:|---:|---:|---:|---:|
| pp2 | 4 | 2 | 1 | 4 | 4 | 24 per microbatch |
| dpa2 | 8 | 1 | 2 | 8 | 4 | 24 per DP worker |
| dpa4 | 8 | 1 | 4 | 8 | 2 | 12 per DP worker |
| dpa8 | 8 | 1 | 8 | 8 | 1 | 6 per DP worker |

All use DCP1, CP disabled, no speculation, no DeepEP, ordinary GPU radix cache,
W4AFP8 weights, FP8 E4M3 KV, page64, context98304, `flashmla_kv` for DSA prefill
and decode, mem-fraction0.80, global request limit48, and chunk16384 before the
native DP division. Resolved per-DP chunks are8192/4096/2048. Prefill graphs are
disabled; decode graphs cover every local batch size through the listed limit.
PP uses layers39/39, microbatch24, async depth0. PP disables overlap scheduling
in upstream; DPA retains its normal overlap schedule. Native DPA round-robin
has no prefix-affinity routing, which is part of what this experiment measures.

The colleague archive's residual/KPool patches target different code. Its
admission/parking patches and FP8/MTP layer-partition assumptions are not
included. This is not the already published PP4 image.

`--hicache` is a separate second phase, with
`write_through/direct/layer_first`, no L3, and `--hicache-size 32`. The exact
upstream DSA allocator applies this decimal-GB size to the anchor MLA KV host
pool, then allocates indexer sidecars separately. Thus256GB across eight ranks
is the anchor allocation, **not a cap on all host cache memory**. Record actual
anchor+indexer allocations and cgroup RSS before interpreting its performance.

## STG preflight

Keep raw infrastructure inventory and pod manifests in the local evidence
directory, outside Git. The preflight checks the full checkpoint architecture,
all indexed shards and safetensors headers, eight allocated H200 GPUs,
interconnect, host resources and public image pull. Header validation checks
presence and structure; it does not checksum every weight byte.

The GPU allocator requires explicit `hami-scheduler` with `gpucores=100` and
`gpumem-percentage=100`. A default-scheduler pod may fail allocation because
the HAMi device binding annotations were never created.

GPU pods must use HAMi scheduling. Do not set `nodeName` directly; the renderer
pins the verified node using scheduler affinity. Model volumes are read-only.
Dedicated100Gi Longhorn RWX PVCs hold compile cache and benchmark results.
GPU containers reserve64CPU/640GiB; the CPU-only vLLM Job uses4CPU/16GiB.

## Images and execution

The dedicated GitHub Actions workflow builds linux/amd64 from the pinned public
SGLang base, runs the tests and source verification, and publishes
`ghcr.io/leoong55/sglang-lasolovev:glm53-stg-topology-<commit12>`.
The workflow summary and `image.json` artifact record the digest. Kubernetes
accepts only a digest reference; a tag or successful build alone is not a GPU
validation result. No model weights or kubeconfig enter the build context.

The benchmark uses public vLLM0.23.0 amd64:
`vllm/vllm-openai@sha256:3a1e7f5904e1a1192a02aa0086ceaffc33985d7044c7bb25b3a43d61bdbe3ac0`.
Its tokenizer is the mounted checkpoint, with offline Hugging Face loading.

From the repository root, after obtaining a successful image digest:

```bash
python3 deploy/glm53-stg-topology/orchestrate.py \
  --kubeconfig /absolute/path/inf-glm53-stg.kubeconfig \
  --image ghcr.io/leoong55/sglang-lasolovev@sha256:IMAGE_DIGEST \
  --commit SOURCE_COMMIT \
  --tooling-commit TOOLING_COMMIT \
  --node VERIFIED_GPU_NODE \
  --campaign s0913a \
  --output /absolute/path/runs/s0913a
```

`IMAGE_DIGEST` and `SOURCE_COMMIT` are placeholders for the verified CI output,
not runnable example values. A campaign identifier must be unique because Job
names and result directories preserve failed and successful attempts. The
orchestrator journals mutations, refuses resources without its ownership
label, runs profiles sequentially, collects ordinary API/log evidence, stops
serving between profiles, and exports results in checksummed log chunks.
Completed Jobs remain as evidence; GPU Deployments are scaled to zero.

`TOOLING_COMMIT` identifies the checked-out benchmark/orchestration revision and
defaults to `SOURCE_COMMIT`. They may differ only when the launcher, installer,
source manifest, observer, supervisor, Dockerfile and renderer remain identical
to the serving image's source. Both revisions and the immutable ConfigMap hash
are recorded. This permits client fixes without changing the model runtime.
The orchestrator makes one declared operational adjustment after rendering:
startup probes retain generated `/health`, while ongoing readiness uses
`/model_info`. Native generated health has a20-second deadline and falsely
failed during a progressing first75k prefill. Admission, worker watchdogs,
pod-generation checks and full workload correctness remain required; API
readiness alone never establishes working inference or capacity.

Every profile first passes admission smoke, then a full unmeasured rehearsal
(short, cold long, repeated long), then the requested measurements on the same
GPU pod. Rehearsal results have `purpose=preparation` in a separate subdirectory;
the analyzer excludes them from comparisons. A failed rehearsal prevents the
measured Job from starting. The optional `--skip-preparation` is only for a
repeat whose image, profile and cache mode already have verified preparation.

For the PP2 and selected DPA finalists, use a new campaign with
`--phase repeat --profiles pp2 dpaN --repetitions 2` to obtain three total
baseline repetitions. Then use another campaign with
`--phase hicache --profiles pp2 dpaN --repetitions 3`.
Select `dpaN` using long benchmark results and capacity evidence, not short
throughput alone. If a profile cannot load or cannot sustain C40, report that
boundary rather than silently lowering the concurrency or changing kernels.

## Concurrent binary log collection

Start this read-only auxiliary collector in another terminal before admission
and measurements, using the orchestrator's same private campaign directory:

```bash
python3 deploy/glm53-stg-topology/collect_follow_logs.py \
  --kubeconfig /absolute/private/inf-glm53-stg.kubeconfig \
  --campaign-root /absolute/private/runs/CAMPAIGN_ID \
  --phase baseline --expected-profiles 4
```

For the two finalists, use `--phase repeat --expected-profiles 2` or
`--phase hicache --expected-profiles 2`. It follows one named serving container
at a time through ordinary Kubernetes log GETs, without changing any workload.
It preserves binary output, embedded CR characters, source snapshots and
checksummed metadata in `server-follow.*` files; keep these artifacts private.

Every connection has separate byte offsets and CRI timestamp bounds.
Reconnects are never joined to claim continuity. An observed measurement
window must fit inside one segment without framing errors; this establishes
no observed transport interruption in that window, not infallible kubelet log
retention. API and stream failures remain explicit in metadata and the exit
status. The collector stops when all expected profile attempts are recorded
and serving pods are gone. If stopping it manually, wait until experimental
GPU pods have quiesced, then send Ctrl-C. It does not stop GPU workloads itself.

## CPU client validation

`client_harness.py` runs the actual pinned vLLM client and checkpoint tokenizer
against a strict synthetic server on container loopback. It checks admission,
400 short requests,300 cold long requests and300 repeated long requests, exact
catalog/body joins, usage, finish reasons and cache-flush order. This uses4CPU
and16GiB, with no GPU request or Service. The existing `sglang-w4fp8-pvc` and
`glm53-topology-results` PVCs are prerequisites; the model mount is read-only.

Render from the committed tooling checkout, supplying the node privately:

```bash
python3 deploy/glm53-stg-topology/render_client_validation.py \
  --node VERIFIED_NODE \
  --run-id client-validation-UNIQUE_ID \
  --tooling-commit TOOLING_COMMIT > /absolute/private/client-validation.json
```

Replace the placeholders with a verified node, a unique lowercase DNS label
and its full40-character tooling commit. The renderer verifies the local source
against that commit and emits a separate immutable ConfigMap plus CPU Job.
It does not contact Kubernetes. Keep the rendered manifest private; use the
same authorized Kubernetes workflow to create the resources.

Results live under `/results/client-validation-UNIQUE_ID` with
`purpose=validation`, a ConfigMap SHA and explicit synthetic provenance.
`client-validation-summary.json` must report `status=passed`; the harness
refuses an existing nonempty result directory. All child timings and output
tokens are synthetic protocol fixtures. They establish no model performance,
KV capacity, cache-hit behavior or server C40, and must remain outside model
comparisons. Keep raw artifacts local; `export.py` can export the dedicated
directory through checksummed log chunks using the same scripts ConfigMap.

## Workload and evidence contract

The CPU benchmark Job uses `vllm bench serve`, OpenAI chat and model `GLM-5.3`:

- Short: random1000 input/up to1000 output,400 prompts,C40,rateinf,seed0,
  natural EOS, original unspecified temperature/thinking, percentiles50/90/95/99.
- Long: prefix60000+suffix15000,output1000,20 prefixes,300 prompts,C40,rateinf,
  seed0,warmup1,ignoreEOS,temperature0.3,thinkingtrue, percentiles50/95/99.

The original short command omits `--ignore-eos`, but vLLM0.23.0 itself enables
it for random datasets. The runner explicitly restores natural EOS to implement
the accepted experiment plan; this distinction is recorded in provenance.
The client also needs request-ID instrumentation for prefix-repetition samples
so warmup requests cannot be mistaken for measured completions. Neither change
alters prompts, request order, RNG state, temperature, or thinking settings.

Startup, correctness probes and JIT preparation are outside measured runs.
The checkpoint's chat template opens `<think>` even when `enable_thinking=false`.
The short correctness probe therefore sets `enable_thinking=true` so the `glm45`
parser separates reasoning from the final answer. It still requires exact nonce
content and `finish_reason=stop`, without stripping reasoning tags. This probe
allows2048 tokens to finish reasoning. Its tokenization runs outside the HTTP
event loop, and its two POSTs use separate connections. These probe settings do
not change the 400-request short benchmark's sampling or measured transport.
Each repetition has a short run, a long run after cache flush (followed by the
original single-request warmup), and an immediate repeated long run without
flush. Label these `cold` and `warm`; cold does not mean radix is disabled.

A loopback-only recording proxy in the benchmark container streams each request
to the internal Service without altering prompts or sampling. It preserves
per-request completion usage, finish reasons and cache observability, which
vLLM's detailed report alone does not reliably expose. It is identical for all
profiles and is never published as a cluster/external service.

Before each benchmark suite, the supervisor builds an offline metadata-only
catalog of the seeded long dataset using the same vLLM/tokenizer. It saves
`dataset-catalog.json` and its SHA256 in provenance before starting the measured
vLLM subprocesses; their RNG state is independent. The catalog identifies
prefix groups and exact request-body hashes without storing prompts.
Telemetry includes native `prefill_effective_tokens` and `load_back_tokens`
counters for observing host-cache activity.

Read-only inventories of compiler artifacts are taken before and after each
workload, outside timed requests. Preserve DeepGEMM warmup log messages and any
new or changed compiler artifacts. A timed attempt with further preparation is
retained as preparation and repeated, independently of its performance result.
Its original measurement intent remains recorded; only mechanically observed
preparation with all other checks satisfied is excluded from comparison, so a
clean repeat can qualify. Missing evidence and failed measurements remain blocking.
Stable artifacts and absent warmup messages mean no compilation was observed;
they are not proof that every in-memory JIT path is observable.

The launcher also enables native JSON request logging at level 0. Finished
events retain request IDs, DP rank and cached-token metadata while excluding
prompt/output text and token IDs. The analyzer joins recorder response IDs to
these native IDs and verifies exact catalog body hashes before assigning long
requests to prefix groups. Missing fields or incomplete joins remain
unobservable; per-group cache totals alone do not prove prefix placement.

Keep raw vLLM output, request metadata, server-info, metric time series,
GPU/RAM telemetry, manifest, source commit, image IDs, pod UID, and timestamps.
Long results require 300 successful completions, 1000 output tokens per request,
and `finish_reason=length`. Natural EOS in the short workload is valid and its
actual length is reported. Missing `cached_tokens` is unobservable, not a hit.

Client C40 is not server C40. `/v1/loads` contains independently timestamped DP
slots: their sum and timestamp skew are diagnostics, never a simultaneous C40
claim. DPA's observer reads surviving requests after the stock result handler
every 15 logical forwards, on attention-TP0/CP0 of each group. The existing MLP
metadata collective supplies IDLE batches to empty groups; `run_batch` advances
`forward_iter` once on every group, and overlap copies preserve it. The analyzer
requires the complete DP rankset on the same step, a verified measured-response
ID set, disjoint IDs across groups, and the audited synchronization conditions.
No barrier is added. Unsupported synchronization configurations fail observation.
CPU result timestamps can differ; the claim concerns participation in one shared
forward, not simultaneous HTTP or log delivery. Finished, retracted, queued,
health and foreign requests do not count. PP records the live microbatch union
on PP0/TP0. Missing observer data remains an evidence limitation.
The analyzer reports observed C40, sample coverage, sustained occupancy,
decode progress and request turnover separately. A fast turnover of requests
does not require the same 40 IDs to remain active throughout 30 seconds. Snapshot
evidence does not establish uninterrupted occupancy between observations.
HiCache is judged separately for capacity, correctness, cold/warm latency and
throughput. Historical CP/DCP numbers from the other cluster are reference-only.
About2000 outputtok/s is an investigation target, not a promised PASS criterion.

After exporting the Job results, generate the publishable aggregate locally:

```bash
python3 deploy/glm53-stg-topology/analyze_results.py \
  --root /absolute/path/runs/s0913a \
  --catalog /absolute/path/runs/s0913a/results/RUN_ID/dataset-catalog.json \
  --output /absolute/path/aggregate.json
```

Replace `RUN_ID` with the exported suite directory. Catalog generation makes no
HTTP requests. Keep the catalog, raw logs, requests and infrastructure evidence
local; publish only the analyzer's allowlisted aggregate, which excludes raw
request IDs, prompts, credentials, host addresses and GPU/pod identifiers.

## Separate HiCache RAM gate

Before HiCache, use the independent [RAM preflight](HICACHE_PREFLIGHT.md) to join
completed baseline serving cgroup telemetry with a fresh whole-host RAM sample.
The CPU Job reads the model PVC without modifying it, verifies the selected node,
and binds its short-lived report to the exact serving manifest. It treats the
32 GiB operational margin, actual KV plus DSA indexer allocation, native allocator
admission bound, and configured 640 GiB serving limit as separate quantities.
Missing evidence blocks admission; RAM PASS does not establish cache effectiveness
or replace the existing smoke and benchmark checks. This helper does not modify
the frozen benchmark or orchestrator and never applies cluster resources itself.

## Local verification

```bash
python3 -m unittest discover -s deploy/glm53-stg-topology/tests -v
python3 deploy/glm53-stg-topology/install.py --package-root python/sglang
python3 deploy/glm53-stg-topology/render.py --kind base
```

These checks establish manifest isolation, launch contracts, response handling,
and source identity. Live model loading and full C40 evidence are separate gates.
