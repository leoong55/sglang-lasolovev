# First H200 validation, 2026-09-26

Image supplied by user: `i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:v0.5.20`, observed digest `sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4`. Eight H200, driver580.105.08, torch2.13.0+cu130. Original source08391453b installed all61 overlay files and passed pinned dependency checks.

- Compact indexer: all3 numerical tests passed (4,37,256 Q rows).
- Q-stream eager: both distributed cases passed on all4 ranks, including empty shard, invalid indices, nativeQ8 and FP32 comparison.
- Q-stream graph: both cases failed on all4 ranks. First failure is `topk_length` device-to-host `.item()` validation in the native Q8 Python wrapper during capture. Later capture-invalidated errors are consequences.

Targeted correction adds an opt-out for this value check only, with the default validation preserved. Q-stream uses it because its compaction producer computes the sum of at most `topk` boolean entries, guaranteeing `0 <= length <= topk`. Shape, dtype, device, and contiguity checks remain. This also removes the repeated host synchronization from eager Q-stream tiles. Existing distributed graph tests are the regression gate.

The initial correction was CPU-checked before the second GPU attempt. The GPU retest outcome is recorded below; full-model P6 remains unvalidated. The first P2 serving attempt uses original08391453b, legacy indexer, KV gather, sync HiCache and attention graph off. No speed or loaded-model correctness result is claimed here.

First P2 full-model startup read all41 target shards, loaded target and draft,
allocated KV and began prefill capture. It then failed before readiness because
`_zero_dsa_dcp_padding` still used removed `ForwardBatch.num_token_non_padded_cpu`.
The 0.5.20 invariant host field is `global_num_token_non_padded_cpu`; the scattered
path continues to use `extend_num_tokens`. Four CPU regressions cover global,
scattered, absent and full-length counts. No inference benchmark completed in
this failed startup. Existing compile caches are preserved for the retry.

## Second attempt: source 93a6150ed

The 4-rank distributed Q-stream test completed with exit 0: all four unique
cases passed on each rank (eager/graph, empty/nonempty shards). Existing native
Q8 and FP32 tolerances were unchanged. CPU regressions total 56 passed plus
20 subtests. This is kernel-level evidence, not full-model P4/P6 validation.

The P2 serving profile completed target prefill graph capture, target verification
capture and full DFlash verification capture. Physical target KV was capped at
600000 tokens, corresponding to 2400000 logical tokens under DCP4. Free GPU
memory after capture was 19.36–20.54 GiB per card; inference peak headroom has
not been measured.

Startup then failed before readiness while allocating the MLA host cache:
`Requesting 96.00 GB but only have 92.32 GB free`. These are decimal GB **per
rank**, after the host memory reserve and division across eight processes.
The sidecar index cache had not yet been allocated. For the actual weight
configuration (78 layers, 21 index producers, FP8 MLA 656 bytes per token/layer,
132-byte index records, DCP4 logical capacity), the configured host pools need
approximately 96.003 GB MLA + 20.804 GB index per rank, or 934.45 GB total,
before controller/staging overhead. A CPU diagnostic after server exit observed
781.15 GB host MemAvailable. This is a host RAM admission failure, not CUDA OOM.

No model requests or performance measurements have run. Baseline was explicitly
skipped. The user was asked to free host RAM or authorize one reduced HiCache
capacity consistently across all compared profiles; no memory guard was bypassed.
Six closed-attempt artifacts were exported with remote SHA256 verification;
preflight and the first attempt contribute another fourteen verified artifacts.
