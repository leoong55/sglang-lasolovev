# GLM-5.3: CP8/DCP4 prefill chunks 8k / 16k / 32k

Based on graph v5, commit `3534a03e40cbdbc59f811dac16dd474847a47d81`,
with upstream runtime `0bcd822377da7b5718e674eaf9c870d349424dd1`.
DeepEP, speculation and HiCache remain disabled in this first stage.

The default manifest uses **16384 global tokens**, with graphs for 8192 and
16384 tokens. CP8 uses 1024 / 2048 / 4096 rows per rank for the supported
8192 / 16384 / 32768 buckets. Exact totals only: tail batches remain eager.
A batch cannot replay a larger captured graph or a graph with mismatched
local geometry. Live CP/DCP plans and eager attention boundaries are preserved.

## Build

From a clean checkout of this PR:

```bash
python3 deploy/glm53-chunks-v7/make_bundle.py ../glm53-chunks-v7-release
cd ../glm53-chunks-v7-release
bash build.sh --push
kubectl apply -f manifest.yaml
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=120m
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

The cumulative overlay builds from the original pinned upstream image. Input
and output hashes are checked before installation; a mismatch stops before
any source change. The same verification runs on container start.

## Explicit YAML choices

| Chunk | Prefill graph sizes | Prefill graph maximum | Local rows at full chunk |
|---|---|---|---|
| 8192 | 8192 | 8192 | 1024 |
| 16384 (default) | 8192, 16384 | 16384 | 2048 |
| 32768 | 8192, 16384, 32768 | 32768 | 4096 |

Change `--chunked-prefill-size`, `--cuda-graph-bs-prefill` and
`--cuda-graph-max-bs-prefill` together. A smaller explicit subset of graph
sizes is allowed, provided its maximum equals the chunk. The launcher preserves
all supplied arguments, rejects duplicates and conflicting settings, and prints
the effective arguments and global/local bucket geometry.

TP8/EP8/CP8/DCP4/DP1, FP8 KV/page64, Q8 prefill, full decode graphs through
batch32, max-running32 and mem-fraction0.80 are retained. Larger chunks can
consume more activation/graph memory and lengthen pauses between decode steps.
Only H200 runs can establish whether 16k or 32k improves the workload.

## Validation

CPU-only checks (standard library):

```bash
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover -s deploy/glm53-chunks-v7/tests -p test_chunk_geometry.py -v
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover -s deploy/glm53-chunks-v7/tests -p test_host_launch.py -v
python3 -m unittest discover -s deploy/glm53-chunks-v7/tests -p test_host_install.py -v
```

Inherited `test_host*.py` attention/KV checks require CPU PyTorch. Geometry
tests execute the real CP replay selection method and cover mixed request
lengths, tails, missing captures and mismatched local rows. Installer tests
exercise mismatch refusal, rollback and idempotence. No CUDA capture, full-model
correctness, image build or performance result is claimed from these tests.

On H200, first test 16k with short requests, repeated long prefixes, tail chunks
and batch sizes 1/8/32. Check startup bucket logs, replay counters, successful
completion and KV capacity before the full 300-request benchmark. Compare with
graph-v5 631.77s under comparable initial cache conditions. Then test 32k.

Rollback: restore graph-v5 image and the complete previous 8192-argument profile.
An image-only rollback with a 16384 YAML is invalid for the old launcher.
