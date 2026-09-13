# GLM-5.3 STG: PP2 and DP-attention on eight H200

This experiment compares full `PhalaCloud/GLM-5.3-W4AFP8` with PP2 and
DPA2/DPA4/DPA8 in namespace `inf-glm53`. Long C40 traffic is primary; short
traffic diagnoses decode. It does not change the independent CP/DCP work.
Only ClusterIP is exposed. There is no Envoy, ingress, or public serving API.

The runtime is pinned to upstream
`0bcd822377da7b5718e674eaf9c870d349424dd1`. The installer verifies critical
runtime files against this commit and refuses a different base; it does not
patch scheduling, attention, quantization, or cache policy. The optional PP
observer samples CPU request metadata without changing scheduler decisions.

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

For the PP2 and selected DPA finalists, use a new campaign with
`--phase repeat --profiles pp2 dpaN --repetitions 2` to obtain three total
baseline repetitions. Then use another campaign with
`--phase hicache --profiles pp2 dpaN --repetitions 3`.
Select `dpaN` using long benchmark results and capacity evidence, not short
throughput alone. If a profile cannot load or cannot sustain C40, report that
boundary rather than silently lowering the concurrency or changing kernels.

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
Each repetition has a short run, a long run after cache flush (followed by the
original single-request warmup), and an immediate repeated long run without
flush. Label these `cold` and `warm`; cold does not mean radix is disabled.

A loopback-only recording proxy in the benchmark container streams each request
to the internal Service without altering prompts or sampling. It preserves
per-request completion usage, finish reasons and cache observability, which
vLLM's detailed report alone does not reliably expose. It is identical for all
profiles and is never published as a cluster/external service.

Keep raw vLLM output, request metadata, server-info, metric time series,
GPU/RAM telemetry, manifest, source commit, image IDs, pod UID, and timestamps.
Long results require300 successful completions,1000 output tokens per request,
and `finish_reason=length`. Natural EOS in the short workload is valid and its
actual length is reported. Missing `cached_tokens` is unobservable, not a hit.

Client C40 is not server C40. DPA load samples count independent groups once;
PP observer records union of live microbatch request IDs on PP0/TP0, excluding
queued/finished/retracted requests. Preserve the time series and progress, not
just one maximum count. Missing observer data is an evidence limitation.
HiCache is judged separately for capacity, correctness, cold/warm latency and
throughput. Historical CP/DCP numbers from the other cluster are reference-only.
About2000 outputtok/s is an investigation target, not a promised PASS criterion.

## Local verification

```bash
python3 -m unittest discover -s deploy/glm53-stg-topology/tests -v
python3 deploy/glm53-stg-topology/install.py --package-root python/sglang
python3 deploy/glm53-stg-topology/render.py --kind base
```

These checks establish manifest isolation, launch contracts, response handling,
and source identity. Live model loading and full C40 evidence are separate gates.
