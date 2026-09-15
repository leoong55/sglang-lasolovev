# GLM-5.3 FP8 CP8/DCP4 candidate

Separate fork of v9.14 commit
`219206f7db9f7ffb24dad2605edf72af62d4939e`, on branch
`work/glm53-fp8-cp8-dcp4-v1`. The workflow publishes an immutable image to
`ghcr.io/leoong55/sglang-lasolovev:glm53-fp8-cp8-dcp4-v1-<source SHA>` and
uploads `fp8-cp8-dcp4-c40.yaml`, `image.json`, and the checksummed source kit.
Use the image digest recorded in `image.json`.

This candidate adapts the user's CP8/DCP4 configuration to the official
[zai-org/GLM-5.3 FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json).
It does not include the separate H200 archive TP/PP experiment.

## Effective profile

| Setting | Value |
| --- | --- |
| Deployment and Service | `sglang-glm53-fp8-dcp4`, namespace `inf-glm53` |
| Model path / PVC | `/mnt/model-pvc-fp8` / `sglang-fp8-pvc` |
| Served model | `alpha-fm` |
| Parallelism | TP8, EP8, DP1, PP1, interleave CP8, DCP4 `ag_rs` |
| Target quantization | Autodetected serialized FP8, dynamic activations, 128×128 weight blocks |
| MoE | Native `auto`, A2A `none`; this source selects Triton for FP8 on CUDA |
| Shared-expert fusion | Native default; no disable or enforce flag |
| Attention | `flashmla_sparse_q8` prefill, `flashmla_kv` decode |
| KV | `fp8_e4m3`, page size 64 |
| Capacity | `max-running-requests=40`, `mem-fraction-static=0.78` |
| Prefill | Chunk 16384, interval 1, breakable graphs at 8192/16384 |
| Decode graphs | Full, batch buckets 1/2/4/8/16/32/40 |
| HiCache | Size 96 per rank; write_through/direct/layer_first/cache |
| Speculation | DFLASH, `incoai/GLM-5.3-DFlash2`, unquant draft, FA4, block 8 |
| Draft cache | Window 2048, bounded fast path enabled |
| Graph fallback | `warn`, matching the supplied configuration |
| MQA logits budget | `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.05` |
| Request template | `reasoning_effort=low` |

The configuration retains ordinary SGLang collectives. Removing
`--disable-shared-experts-fusion` does **not** enable all-reduce: that flag
controls merging the shared expert into the routed-expert kernel. On H200
with EP8 the native model decision currently declines that fusion. No
`--enforce-shared-experts-fusion` is added. The original
`--enforce-disable-flashinfer-allreduce-fusion` and custom CP decode fusion
`off` are retained: they disable fusion optimizations, not ordinary all-reduce.

All Humming environment entries and the target `--quantization w4afp8` and
`--moe-runner-backend humming` arguments are removed. Existing Humming source
remains in the inherited overlay but is not selected or required by this profile.
The launcher reports the requested backend accurately instead of calling `auto`
"cutlass". FP8 linear kernels remain under their native selection rules.

## Adaptation boundaries

The new FP8 entry point reuses v9.14 validation with an explicit FP8 contract.
The legacy entry point still requires W4 and its explicit shared-expert flag.
BCG and DFlash accept native FP8 only after checking checkpoint quantization,
MLA dimensions and the auto/Triton MoE path; their existing topology, DSA,
page-layout and model-architecture checks remain in place. HiCache's final
backend gate inherits this contract. Its early layout-only gate remains
independent of weight quantization, as before.

CP/DCP metadata, causal verification, bounded-draft indexing, cache I/O and
collective kernels exchange activations or KV, not packed expert weights.
They are retained. Native `Fp8MoEMethod` supplies block-FP8 MoE computation.
No INT4-to-FP8 conversion is inserted and no new FP8 GEMM kernel is introduced.

## Build and deploy

The workflow installs the cumulative source overlay on the pinned public
`lmsysorg/sglang:v0.5.19-cu130` image, verifies each file hash, and checks the
actual installed command-line parser. The source kit can also build a Harbor
image with `IMAGE=<your-tag> bash build.sh --push` on a machine authenticated
to the destination registry. `GLM53_BASE_IMAGE` can override the base only if
the installer's source hash checks accept it.

Download the ready manifest from the image workflow artifact, then:

```bash
kubectl apply -f fp8-cp8-dcp4-c40.yaml
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-fp8-dcp4 --timeout=7200s
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-fp8-dcp4
```

The deployment is separate from the old workload and needs eight available
GPUs. The original compile-cache PVC `sglang-glm53-dcp4-compile-cache` is
retained, with a new `/mnt/cache/glm53-fp8-cp8-dcp4-v1/` directory. If that PVC
is RWO, sequential tests must obey its node attachment constraints. Existing
workloads are not scaled or changed by this kit. The checkpoint must have
`config.json` at the mount root. The draft is downloaded through Hugging Face
using the pod's existing network/authentication environment.

## Validation and comparison

Host tests exercise actual resolved-view functions and backend guards for
FP8 autodetection and explicit FP8, invalid quantization/block sizes/backends,
W4 compatibility, native FP8 runner selection, shared-expert defaults and the
ordinary post-expert all-reduce decision. Image checks repeat these against
installed source and run inherited cancellation regressions.

**No H200 execution, full checkpoint loading, CUDA graph replay, output quality
or throughput is certified by those CPU tests.** Record startup logs, loaded
weight memory, KV capacity, graph capture success and DFlash acceptance before
comparing performance. With FP8, the same memory fraction leaves less KV space
than W4. Keep prompts, output lengths, concurrency and sampling identical;
report OOM/retractions and eager fallbacks rather than hiding failed points.
This W4-versus-FP8 comparison changes both storage precision and MoE kernels;
it cannot by itself attribute a result to one patch or to PP.
