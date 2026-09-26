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
previous source and both pass with the change. The next check repeats the
three-layer real-weight arithmetic gate with the unchanged rtol0.001/atol0.002
and FP32-reference relativeL2<=0.08. Cross-layout strict parity is still recorded
separately. No new serving result or full-model acceptance exists yet.

At16k, the diagnostic rank-max local MoE times fell4.9–6.5%; at2k they ranged
from1.4% faster to3.3% slower. These are ten replays of one layer with synthetic
activations, not end-to-end speedups or decode acceptance evidence. The test
contains no attention, KV or DFlash work.

Evidence: `humming-drift-r1-summary.json`; full archive contains all8rank outputs
and13files verified bySHA256. Implementation references:
[Humming BF16 epilogue](https://github.com/inclusionAI/humming/blob/main/humming/include/humming/epilogue/gmem_writer.cuh),
[Humming kernel constraints](https://github.com/inclusionAI/humming/blob/main/humming/kernel/humming.py).
The local diagnostic uses pinned0.1.12 sources; upstream main can change.
