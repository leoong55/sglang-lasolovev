# GLM-5.3 CP8/DCP4: DeepEP optimizations v6 + prefill graphs

This experiment starts from prefill-graph v5 (`3534a03e40cb`) and keeps its
breakable prefill graph at **8192 global / 1024 local tokens**, with eager
attention/indexer/CP-DCP metadata and eager tail batches. Decode keeps full
CUDA graphs at 1, 2, 4, 8, 16 and 32 requests.

Runtime base: upstream `0bcd822377da7b5718e674eaf9c870d349424dd1`.

## Changes

- Stock `--moe-a2a-backend deepep --deepep-mode auto` (normal prefill, LL decode).
- Prefill W4AFP8: invalid routes sort last, avoiding the GPU-scalar Python
  slice; fused packing + static FP8 quantization; fused SiLU + quantization
  limited to valid rows; preserve the original BF16 product rounding boundary
  and FP8 saturation; omit output zeroing where invalid maps prevent reads.
- CP decode: partition source tokens before router/shared expert/DeepEP and
  gather the completed MoE output back in original order before attention.
  Padding and the live non-padded count remain valid for replay at smaller
  batches, including batch < EP size. The TP communicator is used for gather:
  TP and CP have identical rank order here, and TP is enabled during capture.
- The optimized paths are gated by `SGLANG_GLM53_DEEPEP_OPT=1`. The launch
  profile sets this flag and `SGLANG_GLM53_PREFILL_BCG=1`, and prints effective
  arguments. Backend/graph arguments must agree with the supplied manifest;
  conflicting arguments fail instead of being silently replaced.

DeepEP NORMAL remains an eager graph break. Graphs still capture surrounding
fixed-shape segments; this does not preserve identical graph coverage or
promise the same graph speedup as the non-DeepEP v5 image.

No changes to model weights, chunk8192, TP8/EP8/CP8/DCP4/DP1, FP8 KV,
max-running32 or memory fraction0.80. HiCache and speculation remain disabled.
LL capacity and communication parameters stay at stock defaults to avoid
mixing another tuning change into this experiment.

## Build and launch

Use the release archive, not the source-kit directory: the archive includes
verified overlays, hashes, the cumulative upstream patch and incremental v5
patch. `REVISION.json` identifies the exact source commit.

```bash
bash build.sh --push
kubectl apply -f manifest.yaml
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=120m
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

Image:
`i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-deepep-opt-v6-0bcd822377da`

The Dockerfile uses the original image
`v0.5.19-latest-0bcd822377da`, not a previous custom image. Overlay input
hashes prevent installation onto a different runtime. Build dependency checks
never import DeepEP or require a GPU. GPU/DeepEP API checks run in the pod.

`manifest.yaml` contains the compile-cache PVC, Deployment and Service in
existing namespace `inf-glm53`; it mounts existing model PVC
`sglang-w4fp8-pvc` at `/mnt/model-pvc-w4fp8`, uses `harbor-pull`, requests
8 GPUs, 112 CPUs and 1536Gi memory. Apply it as the complete deployment config.

## Validation

- Host tests cover unique decode ownership and order, every raw count within
  capture sizes 1/2/4/8/16/32, counter restoration on exceptions, and empty
  inputs. CPU reference kernels compare old/new normal W4AFP8 outputs with
  invalid and skewed routes, preserving rounding and routed scaling.
- Inherited CP/DCP, FP8 KV, Q8 prefill, live graph metadata and installer tests
  are included.
- `tests/gpu_smoke.py` is an optional Hopper kernel and local graph check.
  It does not load the model or exercise distributed DeepEP; it has not been
  run in the development environment.
- No CUDA GPU or Docker daemon was available during development. The image
  has not been built or pushed here; no GPU throughput or full-model numerical
  equivalence is claimed. Host GEMM/collective mocks cannot establish that.

For local host checks from a checkout:

```bash
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover \
  -s deploy/glm53-deepep-opt-v6/tests -p 'test_host*.py' -v
```

For a quick optional GPU check after building (run on a GPU host):

```bash
docker run --rm --gpus 'device=0' \
  -v "$PWD/tests:/checks:ro" --entrypoint python3 \
  i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-deepep-opt-v6-0bcd822377da \
  /checks/gpu_smoke.py
```

Compare the same 300-request benchmark against graph-v5 (631.77s) and stock
DeepEP-v5 (728.25s). Keep cache starting conditions comparable; changed kernel
shapes can compile on first use. A gain is not guaranteed by route-count or
buffer-size reductions.
