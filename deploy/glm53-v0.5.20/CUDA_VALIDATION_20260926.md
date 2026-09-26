# First H200 validation, 2026-09-26

Image supplied by user: `i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:v0.5.20`, observed digest `sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4`. Eight H200, driver580.105.08, torch2.13.0+cu130. Original source08391453b installed all61 overlay files and passed pinned dependency checks.

- Compact indexer: all3 numerical tests passed (4,37,256 Q rows).
- Q-stream eager: both distributed cases passed on all4 ranks, including empty shard, invalid indices, nativeQ8 and FP32 comparison.
- Q-stream graph: both cases failed on all4 ranks. First failure is `topk_length` device-to-host `.item()` validation in the native Q8 Python wrapper during capture. Later capture-invalidated errors are consequences.

Targeted correction adds an opt-out for this value check only, with the default validation preserved. Q-stream uses it because its compaction producer computes the sum of at most `topk` boolean entries, guaranteeing `0 <= length <= topk`. Shape, dtype, device, and contiguity checks remain. This also removes the repeated host synchronization from eager Q-stream tiles. Existing distributed graph tests are the regression gate.

At commit time the correction is CPU-regression checked but NOT GPU-retested. Do not mark P6 validated or enable it in a serving profile yet. The first P2 serving attempt uses original08391453b, legacy indexer, KV gather, sync HiCache and attention graph off. No speed or loaded-model correctness result is claimed here.
