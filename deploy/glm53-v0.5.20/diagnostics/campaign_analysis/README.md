# Recompute campaign analysis

Set `GLM53_CAMPAIGN_ROOT` to the exported campaign directory containing
`collected/*-snapshot`, summary JSONs and the verified trace archives.
Scripts write derived analysis files there; they never contact a model or cluster.
Raw archives are retained outside Git. Their identities are in `../../final-archive-audit.json`.

Example:

```sh
export GLM53_CAMPAIGN_ROOT=/absolute/path/to/prefill16k-20260926
python3 deploy/glm53-v0.5.20/diagnostics/campaign_analysis/analyze_queue_fairness.py
python3 deploy/glm53-v0.5.20/diagnostics/campaign_analysis/analyze_request_native.py
python3 deploy/glm53-v0.5.20/diagnostics/campaign_analysis/analyze_trace.py p10-streamk-profile --all-ranks
python3 deploy/glm53-v0.5.20/diagnostics/campaign_analysis/analyze_moe_skew.py p10-streamk-profile
python3 deploy/glm53-v0.5.20/diagnostics/campaign_analysis/analyze_p10.py
```

Use the archived raw inputs. Read each output's limits: quantiles are per-run,
profiler kernel sums are not additive end-to-end latency, and matching collective
order requires the preserved eight-rank trace contract.
