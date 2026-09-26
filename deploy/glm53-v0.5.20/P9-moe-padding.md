# P9: preserve inactive MoE expert IDs

The standard EP dispatcher indexes its mapping table with router IDs. A padded
ID of -1 therefore selects the last physical expert instead of remaining invalid.
The EP8 real-weight arithmetic gate reproduced a nonzero output on rank7 for a
zero-reference padded row in the original identity layout. This does not establish
corruption of valid full-model outputs, because padded rows are normally discarded;
it does establish phantom expert work and breaks the expected sentinel contract.

SGLANG_GLM53_PRESERVE_MOE_PAD_IDS=1 uses one compiled gather/where operation
to preserve negative sentinels; default0 retains the control path. Existing
DP-padding masking and positive-ID translation stay intact. It is not enabled
in the already measured P2-P8 runs. Two CPU regression tests execute the actual
dispatch method; two subcases fail on the unpatched source and pass with the fix.
The EP8 retest passed the n1 all-padding and n40 partly padded cases, then
failed the original graph/eager gate on an identity-layout concentrated route
at n2048 (layer3). Seven cases completed before failure. The padding repair
is not full arithmetic acceptance; the serving comparison below retains that limitation.

The first mismatch was 336672/12582912 elements (2.7%), max absolute difference
0.052734375 with unchanged rtol0.001/atol0.002. Other ranks reported NCCL peer
memory errors after the asserting rank exited. These secondary errors do not
establish an independent transport fault.

A separate diagnostic holds the real weights and runtime fixed, compares
eager/eager and graph/graph repeats, freezes the route-sort metadata, and
independently disables Stream-K in the probe only. This is not a replacement
acceptance gate and does not relax tolerances. The pinned Humming0.1.12
`gmem_writer.cuh` performs BF16 partial additions, using atomics when more than
three slices contribute. Disabling Stream-K alone removed the measured repeat variation in64/64
rank/case observations; freezing sorting alone passed57/64. See P10 for the
subsequent three-layer component gate and remaining full-model limits.

Primary implementation references: [Humming epilogue](https://github.com/inclusionAI/humming/blob/main/humming/include/humming/epilogue/gmem_writer.cuh),
[Humming kernel constraints](https://github.com/inclusionAI/humming/blob/main/humming/kernel/humming.py).
Local evidence uses the pinned0.1.12 package; upstream main can change.


## Serving result (source fe1dcba13, Stream-K retained)

P9 completed627 performance requests (15 scenarios, three paired seeds),
three reference requests,29 matched-cache diagnostic requests and two exact
nonce/natural-stop checks. The archive and all127 files passed SHA256 verification.
Against P8, the only intended runtime change is the padding sentinel fix;
P10 code is present but disabled. All weights, layouts and capacities are held fixed.

| Metric | P8 | P9 | Change |
| --- | ---: | ---: | ---: |
| Fresh128k TTFT |5.049s|5.002s|−0.9%|
| Mixed short p95 TTFT |4.182s|4.162s|−0.5%|
| Constant short p95 TTFT |1.526s|1.496s|−2.0%|
| Constant long completion |33.484s|35.158s|+5.0%|

No >=5% TTFT gain is demonstrated. Removing phantom padded-row work does not
rebalance valid expert work or change Stream-K arithmetic; this repair should
not be presented as a broad performance optimization. Under constant arrivals,
short queue p95 fell0.764→0.712s while prefill elapsed rose0.620→0.629s.
Long prefill elapsed remained6.68→6.70s, but its client-derived decode interval
rose0.102→0.110s. These are medians of three repetitions, not proof that padding
caused the decode difference; full-model numerical variation persists.

Matched-cache resident repeats differed before any host eviction (maximum
shared top-20 logprob difference0.488). The host reload difference was0.197;
first output IDs matched, but32-token sequences did not. This neither establishes
cache corruption nor passes full-model cache equivalence. Evidence:
`p9-padding-analysis.json`, `p9-padids-r1-summary.json`, and
`p9-cache-matched-comparisons.json`. P9 server was removed only after verified export.
