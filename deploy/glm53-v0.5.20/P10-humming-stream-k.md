# P10: controlled Humming Stream-K disable

The EP8 arithmetic gate exposed a repeatability failure at layer3/n2048 with
concentrated expert routes, including the original identity expert layout.
The subsequent diagnostic measured native eager/eager, graph/graph, changed-route
capture, fixed-route capture, frozen sorting, and Stream-K-off independently.

All64 Stream-K-off rank/case observations were bitwise identical across both
eager and graph repeats. Native modes passed the existing graph/eager tolerance
in33/64 cases; freezing sorting improved that to57/64. Thus route order contributes,
but is not the only source. Humming0.1.12 reduces partial outputs in BF16 across
Stream-K CTAs; more than three slices use atomics. This matches the measured
removal of variation when Stream-K alone is disabled. It does not prove
full-model accuracy or explain every logprob difference.

`SGLANG_GLM53_HUMMING_DISABLE_STREAM_K=1` disables Stream-K in both W4AFP8
expert GEMMs, retaining the same shape bands, tile sizes, quantization,
FP32 accumulation choice, routing, weights and backend. Default0 keeps the
control path. The W4AFP8 layer opts in; unrelated Humming layers are unchanged.
No shared default tuning dictionaries are mutated.

Two CPU regressions execute the actual configuration method; one fails on the
previous source and both pass with the change. The three-layer real-weight gate completed with unchanged rtol0.001/atol0.002
and FP32-reference relativeL2<=0.08, as detailed below. Cross-layout strict parity
is recorded separately. P10 has no serving result or full-model acceptance yet.

At16k, the diagnostic rank-max local MoE times fell4.9–6.5%; at2k they ranged
from1.4% faster to3.3% slower. These are ten replays of one layer with synthetic
activations, not end-to-end speedups or decode acceptance evidence. The test
contains no attention, KV or DFlash work.

Evidence: `humming-drift-r1-summary.json`; full archive contains all8rank outputs
and13files verified bySHA256. Implementation references:
[Humming BF16 epilogue](https://github.com/inclusionAI/humming/blob/main/humming/include/humming/epilogue/gmem_writer.cuh),
[Humming kernel constraints](https://github.com/inclusionAI/humming/blob/main/humming/kernel/humming.py).
The local diagnostic uses pinned0.1.12 sources; upstream main can change.


The three-layer gate completed on sourcefe1dcba13:36cases ×8ranks, two layouts.
Every graph/eager and FP32-reference gate passed unchanged; maximum local
referenceL2=0.048313 and EP8referenceL2=0.045988. The archive and14files are
SHA256 verified. Cross-layout strict parity did not pass for nonzero cases:
maximumL2=0.002368 with FP32 EP summation and0.004875 with BF16 summation.
These smaller residual differences do not establish full-model equivalence.

The prepared serving pair uses the same sourcefe1dcba13, fixed expert placement,
padding repair, CP8/DCP4/EP8, full DFlash, HiCache64 and KV600000. P9control
retains Stream-K; P10candidate disables it. Only that compute switch differs.
P9 is explicitly a known-nondeterministic experimental control, supported by
its64 native FP32-reference diagnostic points within the unchanged bound; its
failed repeatability gate is not relabeled passed. The startup evidence records
that distinction. Neither arm is a production-approved profile. P9 completed627 performance requests and its matched-cache diagnostic;
P10 has been created, with no P10 performance measurements yet.


## Paired resident/host-cache diagnostic

After each P9/P10 performance run, `diagnostics/cache_matched_streamk_pair.py`
repeats the previous 29-request matched-cache workload with the same seed.
Three resident repeats precede 21 independent 128k prefixes, one verified host
reload and three resident repeats follow. Each comparison has the same
130816 cached tokens plus 256 new tokens, and compares first-token top-20
log probabilities and all 32 greedy output IDs exactly. No tolerance is added.

This distinguishes full-model resident repeatability from an additional
HiCache reload difference after the component Stream-K finding. The P9/P10
pair shares the expert layout, padding fix, runtime source and cache capacity.
The script requires its matching weight version, 15 completed performance
scenarios, and saved exact nonce/natural stop before it may flush cache.
Its prepared Job mounts only the results volume; the script refuses to overwrite
existing evidence.
These are sampled output comparisons, not full-logit equivalence or model
quality validation. P9 completed29 requests: resident repeats already differ
before eviction (shared first-top20 logprob difference up to0.488), and host
reload differs by0.197. P10 remains unmeasured at this revision.
