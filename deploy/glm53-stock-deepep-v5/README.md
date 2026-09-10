# GLM-5.3 stock-deepep v5

Independent experiment based on v3 (`258df599b5a53c756125f40cb61b08739dedb908`). Do not stack with the other v5 variant or v4.

## Build and replace the image

Build context is this extracted archive, not the GitHub deploy directory (run make_bundle.py when building from git).

```bash
bash build.sh --push
kubectl -n inf-glm53 patch deployment sglang-glm53-dcp4 --type strategic --patch-file deployment-image-patch.yaml
```

The patch changes only container `sglang` image. Existing command must invoke `python3 /opt/glm53-cp8-dcp4-v1/launch.py`. PVC, resources, model path and Service are preserved. Launcher prints the full effective argument list and overrides only the options listed below. Record the new image tag in your source Deployment YAML too, to prevent a later apply restoring the old image.

Image: `i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-stock-deepep-v5-0bcd822377da`.
Base image remains upstream `v0.5.19-latest-0bcd822377da`; do not build FROM v4.

## Effective options

```text
--moe-a2a-backend deepep
--deepep-mode auto
--cuda-graph-backend-prefill disabled
```

Runtime source is EXACT v3. This is a configuration/build-kit patch; no custom MoE, dispatcher or second shared expert. Upstream chooses normal dispatch for extend and low_latency for decode. W4AFP8 uses the upstream CUTLASS W4A8 implementations. `--moe-runner-backend deep_gemm` does not make these particular W4A8 matmuls DeepGEMM.

DeepEP auto uses its low-latency transport at decode; CUDA/NVSHMEM/RDMA prerequisites must be available inside the inference pod. A GPU-less Docker build only checks package presence and never imports DeepEP. Startup checks the real imports/APIs; it cannot prove the NIC/device-plugin setup or collective correctness. A missing transport must be fixed in the pod/cluster; this kit does not silently change auto to normal or add privileged access.

## Preconditions and validation

TP8 EP8 CP8 interleave DCP4 ag_rs DP1; Q8 prefill, flashmla_kv decode; FP8 KV/page64; chunk8192; max-running32; decode full/max32; SGLANG_ENABLE_CP_V2=1; shared-expert fusion disabled. No HiCache/speculation/overlap/EPLB. Launcher rejects incompatible comparison settings before weight loading.

`runtime.patch` is cumulative from upstream; `v3-to-*.patch` includes the independent delta and build kit. `base-files.json` plus installer reject unexpected source revisions before writing files. Build and push stop on the first error. GPU-less host tests cover source invariants and installation, not CUDA capture, numerical parity or cluster transport. The binary Docker image has NOT been built or pushed from this workspace.

Rollback: restore the v3 image tag in the existing Deployment (use the exact tag you tested). Do not run `rollout undo` blindly after unrelated Deployment changes.
