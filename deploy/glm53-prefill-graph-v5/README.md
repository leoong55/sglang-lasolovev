# GLM-5.3 prefill-graph v5

Independent experiment based on v3 (`258df599b5a53c756125f40cb61b08739dedb908`). Do not stack with the other v5 variant or v4.

## Build and replace the image

Build context is this extracted archive, not the GitHub deploy directory (run make_bundle.py when building from git).

```bash
bash build.sh --push
kubectl -n inf-glm53 patch deployment sglang-glm53-dcp4 --type strategic --patch-file deployment-image-patch.yaml
```

The patch changes only container `sglang` image. Existing command must invoke `python3 /opt/glm53-cp8-dcp4-v1/launch.py`. PVC, resources, model path and Service are preserved. Launcher prints the full effective argument list and overrides only the options listed below. Record the new image tag in your source Deployment YAML too, to prevent a later apply restoring the old image.

Image: `i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-prefill-graph-v5-0bcd822377da`.
Base image remains upstream `v0.5.19-latest-0bcd822377da`; do not build FROM v4.

## Effective options

```text
--moe-a2a-backend none
--deepep-mode auto                 # unused with a2a=none
--cuda-graph-backend-prefill breakable
--cuda-graph-max-bs-prefill 8192
--cuda-graph-bs-prefill 8192
SGLANG_GLM53_PREFILL_BCG=1          # set by launcher
```

8192 is the GLOBAL token bucket, not request count. Capture is limited to an exact 8192-token interleave batch (1024 local rows per CP rank). Tail chunks run eagerly. CP sharding, DCP prefix planning, attention/indexer/KV exchange and final gather/logits remain eager with live metadata; graphs capture the compatible transformer segments between attention calls. Decode retains v3 full graphs.

This is an experimental runtime implementation, not simply removal of compatibility guards. No promise of an improvement or GPU readiness is made before the first Hopper run. Graph bridge buffers and graph pools consume memory; capture OOM must not be confused with a model-weight problem. The kit preserves your mem-fraction-static to avoid silently changing KV capacity between variants.

## Preconditions and validation

TP8 EP8 CP8 interleave DCP4 ag_rs DP1; Q8 prefill, flashmla_kv decode; FP8 KV/page64; chunk8192; max-running32; decode full/max32; SGLANG_ENABLE_CP_V2=1; shared-expert fusion disabled. No HiCache/speculation/overlap/EPLB. Launcher rejects incompatible comparison settings before weight loading.

`runtime.patch` is cumulative from upstream; `v3-to-*.patch` includes the independent delta and build kit. `base-files.json` plus installer reject unexpected source revisions before writing files. Build and push stop on the first error. GPU-less host tests cover source invariants and installation, not CUDA capture, numerical parity or cluster transport. The binary Docker image has NOT been built or pushed from this workspace.

Rollback: restore the v3 image tag in the existing Deployment (use the exact tag you tested). Do not run `rollout undo` blindly after unrelated Deployment changes.
