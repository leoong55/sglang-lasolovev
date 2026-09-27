# Exact configurations measured on H200

These profiles preserve the actual campaign argv/env, including HiCache 64 GB,
KV cap 600000, pinned draft snapshot and the isolated compile-cache directory.
Metadata records the measured source revision and the real comparison parent.
They supersede the early cumulative examples under `profiles/` for reproducing
the reported experiment. No default launch profile is changed by this report.

All nine profiles completed 627 performance requests each. No profile is
production-approved: full-model numerical/cache equivalence and absolute peak
memory headroom remain unresolved. See [the final report](../H200_FINAL_20260927.md).

The conservative candidate for another validation is `p6buckets-r1.json`:
adaptive HRRN, four prefill graph buckets, legacy indexer, KV-gather, sync
HiCache. P8/P9/P10 remain experimental, despite measured fresh/mixed gains.

For P8/P9/P10, mount `expert-layout.json` at `/scripts/expert-layout.json` as
specified by the saved argv. It is the measured calibration-specific placement,
not a guarantee for other text distributions. Model/cache mount paths are
environment-specific and must exist. Source revisions are intentionally not
rewritten to a newer report-only commit.

Inspect without launching:

```sh
python3 deploy/glm53-v0.5.20/launch_profile.py \
  deploy/glm53-v0.5.20/measured-20260927/p6buckets-r1.json --dry-run
```
