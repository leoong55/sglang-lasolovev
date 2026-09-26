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
is not full arithmetic acceptance; P9 has no serving measurements.

The first mismatch was 336672/12582912 elements (2.7%), max absolute difference
0.052734375 with unchanged rtol0.001/atol0.002. Other ranks reported NCCL peer
memory errors after the asserting rank exited. These secondary errors do not
establish an independent transport fault.

A separate diagnostic holds the real weights and runtime fixed, compares
eager/eager and graph/graph repeats, freezes the route-sort metadata, and
independently disables Stream-K in the probe only. This is not a replacement
acceptance gate and does not relax tolerances. The pinned Humming0.1.12
`gmem_writer.cuh` performs BF16 partial additions, using atomics when more than
three slices contribute. That is a plausible numerical-order mechanism, not
yet the measured cause.

Primary implementation references: [Humming epilogue](https://github.com/inclusionAI/humming/blob/main/humming/include/humming/epilogue/gmem_writer.cuh),
[Humming kernel constraints](https://github.com/inclusionAI/humming/blob/main/humming/kernel/humming.py).
Local evidence uses the pinned0.1.12 package; upstream main can change.
