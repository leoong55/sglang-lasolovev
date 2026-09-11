# GLM-5.3: DFlash2 with CP8/DCP4 and FP8 target KV

Stage 2, based on the 8k/16k/32k graph profile. Draft:
`incoai/GLM-5.3-DFlash2`, unquantized, FA4. The draft block size is inferred by
upstream from its config. The full GLM target remains W4AFP8 with packed FP8 KV.
FA4 draft KV uses its own compute dtype through the existing dtype resolver.
DeepEP and HiCache are disabled in this stage.

## Integration

- Fix CP breakable-prefill output: gather both final hidden states and DFlash
  auxiliary features into global token order before the logits processor.
  The old graph path only gathered final hiddens; draft features stayed CP-local.
- Enable DSA DCP verification only for an explicit full GLM-5.3 TP8/EP8/CP8/DCP4
  profile gated by `SGLANG_GLM53_DFLASH_DCP=1`. Other CUDA speculative algorithms
  retain the existing rejection. Fixed-width linear verification only.
- Flatten target verification to B*W independent FlashMLA query rows, retaining
  local KV, Q gathering and cross-rank LSE reduction. Verify row counts for Q,
  causal lengths, sparse indices and scheduler splits before launching kernels.
- Fuse per-query causal bounds into sparse-index translation BEFORE reading the
  page table; then map widened virtual locations to the owning DCP rank.
  Invalid, future, out-of-table, non-owned and padded entries become -1.
- Preserve upstream acceptance/rejection, bonus-token handling, virtual-index
  allocation and draft KV materialization. Restore temporary host sequence
  lengths with finally when target metadata preparation fails.
- Preserve target prefill graphs and target/draft TARGET_VERIFY graphs. This
  does not assert that an ordinary decode graph can be reused for verification.

The existing DFlash worker writes draft KV for prompt tokens before returning
to the scheduler and writes only committed target-input rows after verification.
Its replicated draft pool is sized for the target allocator's widened virtual
location space. The memory budget already accounts for that larger draft pool.
The shared GPU radix cache keeps its normal allocation/refcount semantics.

## Build and run

```bash
python3 deploy/glm53-dflash-v8/make_bundle.py ../glm53-dflash-v8-release
cd ../glm53-dflash-v8-release
bash build.sh --push
kubectl apply -f manifest.yaml
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=120m
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

The complete YAML starts with chunk16384 and preserves max-running32 and
mem-fraction0.80 for comparison. Draft weights and extra graph buffers can
reduce KV capacity or exhaust this memory budget; startup must be measured.
The image builds from pinned upstream, not from an assumed local custom image.

## Validation

Standard-library contract and launcher tests can run without PyTorch.
Inherited attention/cache tests and `test_host_dflash.py` require CPU PyTorch.
The CI workflow also runs the actual Triton index kernel through its CPU
interpreter against an independent scalar reference (DCP4, page boundaries,
causal limits, padding and widths 2/8/16). This is not GPU kernel validation.

```bash
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover -s deploy/glm53-dflash-v8/tests -p 'test_host*.py' -v
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover -s deploy/glm53-dflash-v8/tests -p 'test_chunk*.py' -v
# On a CUDA host with the built image, without TRITON_INTERPRET:
SGLANG_SOURCE_ROOT="$PWD" python3 deploy/glm53-dflash-v8/tests/test_kernel_verify.py -v
```

No full-model GPU run, CUDA graph replay, image build or speedup is claimed.
On H200, check short/long prompts, repeated prefixes, tails and batch1/8/32;
then compare greedy outputs with the no-spec profile and run the usual load.
Inspect speculative acceptance/progress metrics, target graph replay logs,
TPOT/ITL, output throughput and actual KV capacity. Acceptance alone is not
a performance result. Sampling mode must also be checked on the actual image;
upstream can warn and fall back when its sampling verification is unavailable.

HiCache requires the next stage. Do not enable it just by removing a guard.
