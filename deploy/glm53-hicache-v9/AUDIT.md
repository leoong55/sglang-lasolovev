# GLM53 runtime audit — 2026-09-11

## v9.7: HiCache producer layers and continuous bounded draft decode

The new patch is based on v9.6 `640535efa1896354992191573e0f137c2fba7fa9`.
The operator's prefix20 baseline is recorded separately; no GPU profile was
provided, so this patch makes no measured throughput or latency claim.

HiCache index elision is guarded by the same predicate in the GPU allocation
and memory solver. For the supported direct/layer_first/write_through CP8/DCP4
profile, the host packs producer layers and filters zero-row DMA pointers.
Logical callback IDs and completion events are retained. Tests execute the
actual solver, factory and host methods with byte-copy substitutes for DMA;
they cover all four DCP ranks, relocated pages, skipped layers and MTP tails.

Draft continuity was checked independently against scheduler overlap and radix
cache lifecycle. A late prefill result can repoint a prefix after the first
decode forward; owner identity alone is insufficient. The fast path also
checks prefix-tensor identity and protected-prefix length. Retraction, slot
reuse, prefill, clear and L2 restore invalidate continuity; the restore
completion fence executes before the epoch check. CPU FA4 planning uses a
monotonic bound capped at 2303 for the logical DCP page256, while GPU lengths
remain exact. The physical CLI page64 must not determine that bound.

Local validation: **192 CPU/interpreter tests passed** across the v7/v8/v9
build kits, including 8 new producer-index tests, 6 new bounded-continuity tests
and 1 independent-switch test. **34 existing SM90 bridge specializations
compiled** with PyTorch 2.5.1+cpu / Triton 3.1.0. The continuous ring test covers
560 iterations with variable acceptance and request reordering. Compilation
and CPU substitution do not validate GPU DMA, FlashMLA/FA4 execution or replay.

Synchronous durable draft backing and accepted-row filtering still exist.
This revision does not change the speculative verification algorithm or add
32-head FlashMLA/alternative DCP collectives. H200 output parity under cache
eviction/restore and the supplied benchmark remain required hardware checks.

Earlier audit entries below describe their respective original revisions.

## v9.6: concurrency and bounded draft

The uploaded 21:51 log establishes a selector graph-buffer overrun at batch 33
with capture max 32. The patch preserves graph buffer addresses and skips their
staging for eager batches. The opt-in bounded draft design, allocation formula,
CPU backing/HiCache behavior and remaining hardware gates are documented at
the top of README.md. No H200 execution or image build is claimed for v9.6.

## Earlier v9.4 audit

Scope: the cumulative runtime changes from upstream `0bcd822377da` through
v9.3 `80b93b7f`, the operator's four startup logs, and the CP/DCP, DFlash and
HiCache paths reached by the current TP8/CP8/DCP4 profile. This is a source
audit with CPU/compiler regression checks, not a completed H200 inference run.

## Confirmed and fixed

| Finding | Evidence and effect | Repair / regression |
|---|---|---|
| CP split loop changes int32 to int64 | Latest log reaches **actual breakable 16k prefill capture** with roughly 27 GiB free/GPU. `capture_prepare`/static fields use int64 lengths; the shared Triton kernel initializes `extra_seq` with an int32 Python zero. Compilation rejects loop-carried type promotion. This exception is unrelated to `mem-fraction-static`. | Int64 carry and operands, with caller output dtype preserved. Real kernel + CP-v2 and legacy callers versus independent absolute-position ownership, both integer dtypes, all eight ranks, empty/short/mixed requests, 8k/16k/32k, and lengths above int32 range. |
| Short prefill uses the wrong LSE base | `_dsa_impl_for_batch` switches a non-CP Q8 prefill to `flashmla_kv`; `is_dsa_dcp_lse_base_on_e` previously consulted the configured Q8 string. DCP AG+RS would use `exp2` on natural-log LSE. A two-owner example gives about 4.73 instead of 5.0. | Use the same per-batch dispatch selector as metadata and attention. Test the actual selector + base decision + weighted combine, including empty owners. Separate actual Triton tests compare AG+RS correction and A2A combine against a numerical reference. |

The signed carry arithmetic was checked separately against absolute-position
ownership. No independent valid-input arithmetic bug was established, so it
was retained. The dtype fix does not narrow lengths or change the splitting rule.

## Cross-path checks

| Path | Source checks and existing regression coverage | Remaining boundary |
|---|---|---|
| Argument resolution → allocation | Generic DSA selector is separate from inner kernel selectors. Full TP sharding gives one draft KV head/GPU. DCP retains the widened virtual slot/page space. Target FP8 and FA4 draft BF16 are distinct contracts. Earlier operator logs confirm the allocation and profile fixes. | Peak memory during all captures and real load remains a measurement on H200. |
| CP prefill capture → replay | Exact 8k/16k/32k buckets only; unmatched tails run eagerly. Live CP/DCP plans are rebuilt per batch. Attention executes across an eager break with refreshed lazy QKV context and a fresh scratch allocator. Gather workspaces are batch/stream scoped; returned hidden states do not alias transport scratch. | CUDA/NCCL capture and replay cannot be established by host mocks. Change batch partition, prefix lengths and request slots between replays of the same bucket. |
| Prefill → DFlash draft | Both ordinary hidden states and packed/list auxiliary hidden states are gathered into global token order before draft KV materialization. Draft sidecar uses full virtual slot IDs. Prompt materialization finishes before the worker returns. | FA4 forward and fused draft append kernels require real GPU execution. |
| Verify capture → live verify | Causal custom-mask policy is shared with the validated DSA/DCP profile. Width/row guards remain enabled. Per-query causal bounds mask future/out-of-range top-k positions before DCP owner translation. Metadata has one row per query; FlashMLA head padding and split counts agree. CPU length mirrors are restored even on preparation exceptions. | Compare actual accepted token IDs and generated output against target-only greedy inference, including page boundaries and batches 1/2/4/8/16/32. |
| Prefill Q8 → short-tail FlashMLA | Both kernel and metadata use `_dsa_impl_for_batch`; LSE base now follows it as well. Empty DCP owners receive zero output and -inf LSE before reduction. | Exercise 1–7 token extends/tails, the CP threshold, and long cached prefixes. |
| GPU → RAM write-through | Unified radix includes KV, INDEXER and DRAFT in one transfer. Target rows are owner-sharded; DSA/draft use the full logical ID space. D2H waits on the forward producer stream beyond early length publication. Completion follows all pools; locked pending nodes cannot be evicted before acknowledgement, including split nodes. | Real asynchronous DMA/event ordering and allocator reuse under load need an H200 eviction/restore run. |
| RAM → GPU restore | Host indexer allocation covers `anchor.logical_size`; copies include all virtual indexer pages. Draft state uses matching virtual IDs. Per-layer completion follows sidecar copies; load-back waits behind in-flight forwards. Existing tests move data to different destination pages, check byte equality and pending-event handling. | Force eviction and prove RAM restore using counters plus output parity. Merely observing a cache hit is insufficient. |

An additional inherited behavior to watch: `dflash_utils` catches failure to
import sampling kernels; non-greedy DFlash can then warn and fall back to greedy
verification. No such import failure is established by these logs. Sampling
acceptance is therefore still a hardware gate: fail that test if the warning
`non-greedy verification is unavailable` appears; do not count greedy output
as a successful sampling test.

## Stronger pre-build checks

Previous AST/CPU tests missed compiler typing rules. `kernel_preflight.py`
now compiles **26 specializations** of CP split, DCP KV index/page builders,
prefill/verify/decode index transforms and LSE correction/combine to SM90 PTX
and cubin. It does not initialize CUDA or require weights. Docker runs it after
patch installation using the **image's installed Triton** and reports versions.
The CPU workflow also runs it on Triton 3.1.0. The latter version alone is not
claimed to match the operator's image.

The Triton interpreter checks numerical behavior separately; it cannot prove
that CUDA compiler/type/layout lowering succeeds. Neither compilation nor the
interpreter runs FlashMLA Q8, FA4, DeepGEMM, NCCL or hardware cache transfers.
Source and runtime files remain pinned and checksum verified in the bundle.

## Next H200 gate

Keep the current 16k and memory fraction for the next comparable startup.
Require target prefill, target verify and draft graph capture to finish, then:

1. Run greedy prompts with DFlash/HiCache disabled as the target reference;
   compare with DFlash enabled and HiCache off, then with HiCache on.
2. Exercise single/mixed requests, 1–7 token tails, 63/64/65 and 255/256/257
   boundaries, exact 8k/16k/32k buckets, and repeated 16k replays with changing
   prefixes/request slots. Use explicit graph buckets matching each experiment.
3. Force GPU eviction with distinct long prefixes, revisit an earlier prefix,
   and require RAM restore counters to increase with greedy token parity.
4. Repeat split-prefix, cancellation and host-pressure cases while transfers
   are pending. Require forward progress, released locks and reclaimed slots.
5. Verify non-greedy sampling actually uses its sampling kernel; then measure
   the established workload with cold/warm cache results separated.

Do not treat `/health`, successful compilation, or startup completion as proof
of inference correctness or HiCache round-trip correctness. No Docker image
was built/pushed and no cluster operation was performed during this audit.
