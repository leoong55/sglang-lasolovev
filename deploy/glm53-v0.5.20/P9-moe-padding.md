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
The unchanged EP8 real-weight arithmetic gate is the next CUDA check. No
performance or full-model numerical acceptance is claimed for this patch yet.
