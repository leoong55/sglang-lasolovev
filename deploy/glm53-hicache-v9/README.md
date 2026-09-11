# GLM-5.3: DFlash2 + CP8/DCP4 + GPU/RAM HiCache

Third stage, based on the DFlash PR. Full target model
`PhalaCloud/GLM-5.3-W4AFP8`, draft `incoai/GLM-5.3-DFlash2`,
8xH200, TP8/EP8/CP8 interleave, DCP4 ag_rs, FP8 target KV and
FA4 draft KV in its upstream compute dtype. Initial chunk: 16384.

## v9.2: recognize DSA dispatch and its internal kernels separately

The next H200 log confirms the v9.1 pool fix: one draft KV head per GPU,
2.42 GiB each for K and V, with about 34 GiB still free after allocation.
Startup then failed the DFlash/HiCache profile check. Both the DFlash and
prefill-graph gates incorrectly compared generic attention backends against
internal DSA kernel names. The generic pair is `dsa` / `dsa`; the separate
DSA selectors are `flashmla_sparse_q8` / `flashmla_kv`.

Both gates now validate those fields separately through resolved views.
This also fixes prefill graphs being disabled by CP/DCP compatibility checks
despite an explicit `breakable` request. The HiCache requirements remain
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

No full-model H200, CUDA DMA/graphs or performance validation is claimed.
On the built 8xH200 deployment, require these checks in order:

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
