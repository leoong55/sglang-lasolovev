# GLM-5.3: DFlash2 + CP8/DCP4 + GPU/RAM HiCache

## v9.7: producer-only HiCache index storage and bounded draft continuity

Image tag: `glm53-hicache-v9.7-0bcd822377da`. Based on the complete v9.6
stack (`640535efa1896354992191573e0f137c2fba7fa9`); CP8/DCP4, Q8 prefill,
FP8 target KV, DFlash2/FA4 and write-through HiCache are retained.

The launcher enables two independently reversible optimizations for the
corresponding selected features. Set either environment variable to `0` to
compare against the v9.6 behavior on the same image:

| Environment variable | Default through this launcher | Effect |
| --- | --- | --- |
| `SGLANG_GLM53_HICACHE_INDEX_ELISION` | `1` with HiCache | Allocate and transfer only producer DSA index layers |
| `SGLANG_GLM53_BOUNDED_DRAFT_FASTPATH` | `1` with bounded draft window 2048 | Reuse a validated continuous-decode window mapping |

Ordinary SGLang entrypoints default both variables to `0`. Explicit values
are respected; disabling HiCache or bounded draft disables its optimization.
These settings take effect on process restart and are printed by the launcher.

HiCache retains all 78 logical layer-completion events while storing only
the 21 GLM53 producer index layers. The GPU sizing predicate and allocator
use the same eligibility rule; host DMA lists contain only real buffers.
Main MLA KV and draft state keep their existing representations. The startup
message should include `DSA indexer HiCache stores 21/78 target layers`.

At the same target-KV memory budget, the calculated cost decreases from
`78 * (656 / 4 + 132) = 23088` to `78 * 656 / 4 + 21 * 132 = 15564`
bytes per logical token per GPU. This predicts about 2.97 million slots from
a previous 2 million, before small auxiliary allocations/alignment and actual
startup budgeting. It does not predict a decode speedup. The larger logical
capacity also increases the bounded draft's CPU backing size.

Bounded draft skips the full virtual-ID/version scan only for requests with
a proven continuous accepted append after a validated window. Request owner,
prefix tensor, protected-prefix length, request slot, prefill, retraction and
HiCache restore/clear invalidate that proof. The fallback still validates and
refills from CPU. Its CPU FA4 planning bound uses the pool's logical DCP page
size; the exact device-side attention lengths are unchanged.

The fast path prints `GLM53 bounded draft windows: reused=..., validated=...`.
Durable CPU backing writes and the HiCache completion fence remain synchronous.
This patch does not add a new FlashMLA 32-head kernel or change DCP collectives.
GPU startup, DMA/replay correctness and end-to-end speed still require the H200
run; CPU tests cannot certify those properties.

`manifest-bounded-48.yaml` uses mem fraction 0.80, running limit 48, 16k prefill
and decode buckets 1/2/4/8/16/32/48, matching the last supplied operator geometry.
The launcher does not add bucket 40 or alter arguments in an existing manifest.

The supplied 60k-prefix/15k-suffix/1k-output, 20-prefix, concurrency-40 baseline
is recorded in [operator-v9.6-c40-prefix20.json](benchmarks/operator-v9.6-c40-prefix20.json):
512.75 output tok/s, mean TPOT 66.81 ms, P95 TTFT 68.42 s, 300/300 successful.
See [the repeatable A/B procedure](benchmarks/V9.7-VALIDATION.md).

Build the cumulative overlay with `make_bundle.py`, then run `bash build.sh`
inside the resulting bundle (`--push` also publishes the image). The build
uses the pinned upstream base rather than overlaying an already patched image.
`REVISION.json` records the exact source commit. All earlier chunk/DFlash/HiCache
patches are included; experimental DeepEP patches remain outside this stack.

The sections below describe earlier revisions and their original measurements.

Third stage, based on the DFlash PR. Full target model
`PhalaCloud/GLM-5.3-W4AFP8`, draft `incoai/GLM-5.3-DFlash2`,
8xH200, TP8/EP8/CP8 interleave, DCP4 ag_rs, FP8 target KV and
FA4 draft KV in its upstream compute dtype. Initial chunk: 16384.

## v9.6: fix batch 33 and optionally bound draft GPU KV

The 21:51 operator log fails in `_SelectorDraftSampler.stage_sampling_params`:
32-row graph buffers are updated from a 33-request batch before eager dispatch.
`torch.clamp(out=...)` attempts to resize a view, then `greedy_mask.copy_` raises
`32 != 33`. SIGQUIT and cancelled HTTP requests follow that exception. This is
not an allocation failure. v9.6 skips graph-parameter staging above capture
capacity; the existing eager selector uses the live batch sampling parameters.
Graph addresses remain stable. Running 48 requests with graphs through 32 is
supported by this fix; expanding graph capture is optional and costs memory.

The new launcher option `--glm53-draft-cache-window 2048` enables a physically
bounded draft GPU pool. Its default is **0**, preserving the full draft pool.
It is separate from upstream `--speculative-draft-window-size`, which only
changes the attention view in this pinned implementation. Do not combine them.

* Each request has a 2560-row ring: the native 2048-token window, page alignment
  slack and scratch space. Six BF16 K/V layers retain the model's dtype.
* Target virtual IDs and verification locations stay unchanged. Draft attention
  gets its own page-aligned request table and physical scratch locations.
* All committed draft K/V is retained in a pinned CPU backing store. Shared L1
  prefixes and retracted requests can refill the ring without stale rows.
  Virtual-ID versions detect allocator reuse; rejected proposals are excluded.
* HiCache DRAFT sidecars copy between CPU backing and their existing L2 host
  pool. A restore invalidates ring tags, and refill waits for all six layers.
  Target FP8 KV and DSA transfers retain their existing paths and write-through
  producer fence. The backing store is additional to the HiCache L2 allocation.
* The memory solver reserves the fixed ring and per-virtual-token version table
  instead of charging full GPU draft K/V per target token.

At 48 requests, K/V tensors including padding require about **0.360 GiB/GPU**,
plus tags and a 4-byte version entry per logical target slot. At 2,000,000 slots,
the previous K/V alone used about **5.72 GiB/GPU**. CPU backing then needs another
5.72 GiB per rank, about 45.8 GiB across eight ranks, before the existing L2 pool.
These are tensor-size calculations, not measured post-startup memory figures.

[manifest-bounded-48.yaml](manifest-bounded-48.yaml) matches the supplied
0.80 memory fraction, 48-request limit, 16k prefill, decode graphs through 32,
DFlash2 and write-through HiCache. It uses image
`glm53-hicache-v9.6-0bcd822377da` and adds the bounded-cache option. Existing
manifests keep the full draft pool for comparison. Setting the option to 0
restores full allocation; removing all speculative options disables DFlash.
All of these changes require a pod restart.

This opt-in implementation uses blocking D2H materialization and explicit
window-miss synchronization. Their latency/throughput cost is **unmeasured**.
CPU regressions execute ring wrap, reuse, last-page indices, actual worker
projection/accepted-prefix filtering, solver/builder integration, HiCache
relocation, and stable sampler buffers at 32/33/48/64 requests. H200 startup,
CUDA graph replay, greedy parity versus the full pool, forced L1/L2 eviction,
retraction/cancellation and the 300-request benchmark are still required.
The PR remains draft pending these hardware checks.

Build tag: `glm53-hicache-v9.6-0bcd822377da`.

## v9.5: independent launch controls and recorded v9.4 result

The operator reports a successful v9.4 run: 300 completed requests, no errors,
549.67 output tok/s over 545.78 seconds. The full metrics and comparison with
seven previous measurements are in [benchmarks/COMPARISON.md](benchmarks/COMPARISON.md).
The best previous aggregate rate was 474.86 tok/s; v9.4 is 15.75% higher,
but median TTFT rose from 10.56 to 17.70 seconds. This measures the combined
configuration, not DFlash's isolated contribution. The actual pod arguments,
image digest and detailed client JSON were not attached to this result.

The old experiment launcher required DFLASH, HiCache and exactly 32 running
requests. The DSA runtime guard also incorrectly coupled HiCache to DFLASH.
These restrictions now follow the selected features:

| Control | How to select it | Behavior |
|---|---|---|
| Speculation off | Remove all `--speculative-*` options **and their values** | No draft worker; target/indexer HiCache can remain enabled |
| Speculation on | Keep the four DFLASH options from `manifest.yaml` | Full GLM-5.3 DFlash2, FA4, unquantized draft |
| Request limit | Any positive `--max-running-requests` | No launcher ceiling of 32; available KV still limits actual admission |
| Decode capture | `--cuda-graph-max-bs-decode` and optional `--cuda-graph-bs-decode` | Independent of request limit; larger batches can run outside the captured range |
| Decode graphs off | `--cuda-graph-backend-decode disabled` | Capture sizes may be removed |
| Prefill graphs off | `--cuda-graph-backend-prefill disabled` | Capture sizes may be removed; chunk size remains explicit |
| HiCache off | Remove `--enable-hierarchical-cache` | L2 is disabled; ordinary unified L1 radix caching remains |
| Prefill chunk | 8192, 16384 or 32768 | With breakable capture, the largest bucket and max prefill BS must equal the chunk |

These are **startup settings**, requiring a new image and pod restart; they
are not live HTTP controls. Feature environment variables are derived from
the CLI, so stale `SGLANG_GLM53_DFLASH_DCP=1` cannot turn a removed draft back on.
The launcher prints the selected settings and complete effective argv.

Full alternative manifests preserve the PVCs, Service, 16k chunk, 0.75 memory
fraction and HiCache settings. Apply one to the existing Deployment at a time:

| Manifest | DFlash | Request limit | Maximum decode capture BS |
|---|---|---:|---:|
| [manifest.yaml](manifest.yaml) | on | 32 | 32 |
| [manifest-nospec-32.yaml](manifest-nospec-32.yaml) | off | 32 | 32 |
| [manifest-nospec-64.yaml](manifest-nospec-64.yaml) | off | 64 | 64 |
| [manifest-dflash-64.yaml](manifest-dflash-64.yaml) | on | 64 | 64 |

The 64 variants include decode buckets `1 2 4 8 16 32 48 64`. Compare on/off
at 32 first, then change the request limit; retain client concurrency 40 to
compare the existing workload. Testing actual concurrency 64 requires a
separate client run with that concurrency. Larger capture sets consume GPU
memory; this patch removes a hard limit, not the physical capacity constraint.

The existing target FP8, DSA/CP8/DCP4, `write_through`, `layer_first`, `direct`
and full-model checks remain. Other speculative algorithms and the experimental
DeepEP patches are outside this profile. HiCache without speculation uses the
target/indexer pools and does not create a draft sidecar.

CPU regressions cover all four DFlash/HiCache combinations, graph opt-outs,
stale environment variables, larger request/graph capacities and retained
unsupported-profile rejection. The compiler gate now covers 34 specializations,
including CP request dimensions 64 and 96. v9.5 has not been run on H200 here.

Build tag: `glm53-hicache-v9.5-0bcd822377da`.

## v9.4: compiler gate and short-prefill LSE correctness

The next H200 log reaches actual 16k breakable prefill capture, then Triton
rejects the CP split loop: its carry starts as int32 but captured lengths are
int64. The carry and loaded operands now remain int64; caller output dtypes
are preserved. Both CP-v2 and legacy callers are covered with real kernels.
No memory-fraction adjustment can fix this compiler error.

The audit also found a separate numerical bug on short prefill tails that
cannot use CP. Dispatch falls back from Q8 to FlashMLA KV, but the DCP reducer
selected the LSE base from the configured Q8 backend. It now uses the same
per-batch selector as metadata and attention. This avoids applying exp2 to
natural-log LSE. See [AUDIT.md](AUDIT.md) for scope and remaining hardware gates.

`kernel_preflight.py` compiles 26 CP/DCP kernel specializations to SM90 PTX
and cubin without a GPU, model weights or CUDA context. Docker builds run it
against the installed image's Triton after patch verification; CI additionally
runs it with Triton 3.1.0. Numeric interpreter tests exercise CP splitting on
all eight ranks and actual AG+RS/A2A LSE kernels. Compiler and interpreter
checks complement one another; neither is a full H200 integration run.

Build tag: `glm53-hicache-v9.4-0bcd822377da`. Keep 16k and the current
`mem-fraction-static 0.75` for the next comparison.

## v9.3: causal verify capture and the model-stage prefill graph rule

The v9.2 H200 log passes KV allocation and attention-backend profile validation,
then fails while warming the target verify graph. Capture builds a DFlash input
through a backend mask policy that did not recognize the opt-in DSA/DCP path;
it supplied the generic custom-mask buffer, unlike live DFlash's causal input.
The policy now returns no custom mask for a `DeepseekSparseAttnBackend` whose
`glm53_dflash_dcp` profile was validated. Wrapper unwrapping is preserved.
Other backend profiles keep their existing policy, and Eagle retains tree masks.
The causal-layout validator still rejects custom and ragged masks.

This log also confirms an actual disabled prefill graph mode. Beyond the early
logging issue fixed in v9.2, `handle_model_specific_adjustments` unconditionally
disabled DSA CP prefill after CLI graph resolution. That later rule now preserves
the selected `breakable` mode for the opt-in GLM53 profile. Disabled graphs remain
disabled; other DSA CP models/topologies retain the old disable rule.

Five CPU regressions execute the actual capture-input builder, mask policy,
causal validator, CLI graph parser and DSA model rule. The old source reproduces
both startup failures. Coverage includes verify batches 1/2/4/8/16/32, prefill
8k/16k/32k, wrapper handling, graph opt-out and retained unsupported-mode checks.
These tests do not execute CUDA capture. In the next H200 log, check actual
prefill capture and completion of target/draft verify capture before inference.

Build tag: `glm53-hicache-v9.3-0bcd822377da`. Keep the existing 16k profile and
`mem-fraction-static 0.75` for the first retry; replace the image only.

## v9.2: recognize DSA dispatch and its internal kernels separately

The next H200 log confirms the v9.1 pool fix: one draft KV head per GPU,
2.42 GiB each for K and V, with about 34 GiB still free after allocation.
Startup then failed the DFlash/HiCache profile check. Both the DFlash and
prefill-graph gates incorrectly compared generic attention backends against
internal DSA kernel names. The generic pair is `dsa` / `dsa`; the separate
DSA selectors are `flashmla_sparse_q8` / `flashmla_kv`.

Both gates now validate those fields separately through resolved views.
The graph change makes the CP/DCP support and capture-routing predicates
recognize this DSA profile. The early DSA CP log previously printed a literal
`backend=disabled` before graph resolution; it was not evidence of the final
graph mode. That message now says graph resolution happens later. Explicit
`breakable` bypasses the generic default compatibility policy, while runtime
CP capture still needs the corrected profile predicate. HiCache still requires
unified radix, `cache`, `write_through`, `layer_first` and `direct`.

Six CPU regressions execute the actual resolution views, backend selector,
DSA startup gate and graph-compatibility hook. They reproduce the old startup
exception and graph disable decision, and check both corrected paths and
rejection of other backends, models, topologies and cache policies. Existing
graph tests now use the real generic backend selector instead of a mock
returning internal DSA names. Full CUDA graph capture is still unvalidated.

Build tag: `glm53-hicache-v9.2-0bcd822377da`. The example manifest now uses
`mem-fraction-static 0.75`, matching the latest operator log. For the next
retry, change only the image in the existing deployment.

## v9.1: fix draft KV allocation under target prefill CP

The first H200 startup log exposed an allocation mismatch: DFlashAttention,
FA4 and the draft memory budget use the full TP group (one KV head per GPU
with eight total heads and TP8), but the MHA pool used the target attention
group (eight heads with CP8). At 493312 target slots and DCP4, that inflated
the six-layer BF16 draft K/V buffers from about 5.65 GiB to 45.17 GiB per GPU.
The OOM happened during draft pool construction, before HiCache or graphs.

The DFlash-family draft MHA pool now uses full TP for head sharding, while
retaining DCP's widened virtual token and page capacity. Other MHA pool paths
keep their existing geometry. A startup log prints heads, tokens, page size,
layers and dtype; this profile must report `heads=1` and `layers=6`.
Four CPU regression tests execute the actual sizing, shape and budget methods;
the old code fails them. They cover the reported geometry, CP/DCP combinations
and preservation of the other model paths. This is not an H200 startup test.

FA4 draft KV stays BF16; target KV stays FP8.

## What changed and why

* DSA indexer host buffers now cover the anchor's **logical_size**. Target
  MLA KV is distributed across DCP ranks; indexer KV is replicated across
  that widened token-ID space. With page64/DCP4, one 256-token radix page
  transfers 64 MLA rows per rank and four 64-token indexer pages per rank.
* The existing full draft sidecar is sized from the target host's logical
  capacity and shares its virtual device/host IDs. It must not use MLA's
  owner filtering. DFlash materializes prompt and committed verify-input
  KV before the worker returns; the unmaterialized bonus token stays out
  of the cached prefix.
* Device-to-host write-through explicitly waits for the DFlash producer
  stream, including materialization after early sequence-length publication.
* Unified radix submits KV, INDEXER and DRAFT as one L2 operation. Its
  completion event is recorded after every pool. Pending nodes remain
  locked until completion, including splits; every rank participates in
  the completion-count collective even with an empty local queue. Load
  completion for each layer follows all corresponding sidecar copies.

The last point is verified against the existing implementation, not inferred
from the policy name. Tests execute the real pool mapping and state methods.
They replace CUDA transfer primitives with byte copies/event recorders.

## Supported first experiment

The manifest explicitly selects unified radix, host memory mode `cache`,
`write_through`, `layer_first`, `direct`, L1/L2 only and `hicache-ratio 1.0`.
Both runtime guards and launcher enforce this profile. L3 storage, LMCache,
HiSparse, `write_back` and `buffer_only` remain outside this patch.

Ratio 1.0 applies to target KV sizing, **not total RAM**: replicated indexer
and draft sidecars add memory. It is an initial test setting; inspect total
pinned RAM across all eight processes and actual device KV capacity.

## Build from a clean Git checkout

```bash
python3 deploy/glm53-hicache-v9/make_bundle.py ../glm53-hicache-v9-release
cd ../glm53-hicache-v9-release
bash build.sh --push
kubectl apply -f manifest.yaml
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=120m
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

The generator produces a cumulative runtime.patch, verified overlay,
base-files.json, v8-to-hicache.patch, revision metadata and SHA256SUMS.
The image starts from pinned upstream `0bcd822377da7b5718e674eaf9c870d349424dd1`.
The commands above are operator instructions; no image push or cluster
rollout was performed while preparing this PR.

## Validation and GPU acceptance gate

```bash
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover -s deploy/glm53-hicache-v9/tests -p 'test_host*.py' -v
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover -s deploy/glm53-hicache-v9/tests -p 'test_chunk*.py' -v
```

New CPU tests cover DSA host-tail capacity and relocated page round trips,
byte-exact target KV transfer on all four DCP owners, replicated draft
capacity and alias ownership, DMA/sidecar layer ordering, the producer
fence, split-node acknowledgement and an empty-rank collective.
The shared workflow also checks the actual causal Triton index kernel
with the CPU interpreter. NumPy 1.26 is pinned for Triton 3.1's interpreter.

The v9.4 operator benchmark above establishes startup and completion of that
workload on the operator's deployment. It does not establish numerical parity
or forced GPU/RAM eviction and recovery. v9.5 controls still need an H200 run.
For the remaining hardware validation:

1. Re-run the DFlash profile with HiCache off; record greedy token IDs,
   request errors, acceptance, KV capacity and TPOT/ITL/throughput.
2. Enable this manifest. Repeat prompts spanning 8k/16k/32k, mixed tails
   and shared-prefix branches; require the same greedy token IDs.
3. Insert enough distinct long prefixes to force GPU eviction. Revisit an
   earlier prefix and confirm RAM load-back counters increase for target,
   indexer and draft, without new errors or output differences.
4. Repeat with a split shared prefix while writes are pending; then apply
   RAM pressure and request cancellation. Check occupancy returns, no
   stuck pending writes, and continued progress on every rank.
5. Compare sampling distribution/quality and run the established 300-request
   load. Report cold versus warm separately. Do not infer a speedup from
   cache hits or draft acceptance alone.

The CPU suite establishes mapping and ordering contracts, not a measured
GPU/RAM round trip. Keep this PR draft until the hardware gate is recorded.
