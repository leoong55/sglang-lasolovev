"""Compare the matched P9/P10 pair and qualify the P8/P10 trace comparison."""
import os
import json
from pathlib import Path
from statistics import median

R = Path(os.environ["GLM53_CAMPAIGN_ROOT"]).resolve()
attempts = ['p2-r4', 'p6buckets-r1', 'p8-expertlayout-r1', 'p9-padids-r1', 'p10-streamk-r1']
fair = json.loads((R / 'queue-fairness-results.json').read_text())
native = json.loads((R / 'native-request-comparison.json').read_text())
metrics = {}
for attempt in attempts:
    data = json.loads((R / f'{attempt}-summary.json').read_text())
    result = {case + '_ttft_p95_s': median(x['ttft_p95_s'] for x in data['scenarios'] if x['scenario'] == case)
              for case in ['fresh', 'mixed', 'constant', 'burst', 'repeated']}
    for row in fair['summary']:
        if row['attempt'] == attempt:
            for key in ['short_ttft_p95_s', 'short_decode_interval_mean_s', 'long_ttft_s', 'long_e2e_s']:
                result[row['scenario'] + '_' + key] = row[key]
    metrics[attempt] = result
comparisons = {}
for control in attempts[:-1]:
    comparisons[control] = {key: 100 * (metrics['p10-streamk-r1'][key] / value - 1)
                            for key, value in metrics[control].items()}
traces = {}
for attempt in ['p8-expertlayout-profile', 'p10-streamk-profile']:
    skew = json.loads((R / f'{attempt}-moe-skew.json').read_text())
    trace = json.loads((R / f'{attempt}-trace-analysis.json').read_text())
    traces[attempt] = {'moe': skew['summary'], 'ranks': [
        {'rank': int(x['trace'].split('TP-')[1].split('-')[0]),
         'kernel_span_ms': x['kernel_span_ms'],
         'kernel_union_ms': x['kernel_union_ms'],
         'no_recorded_kernel_ms': x['kernel_span_ms'] - x['kernel_union_ms'],
         'humming_sum_ms': x['groups']['humming_gemm']['sum_kernel_ms'],
         'nccl_sum_ms': x['groups']['nccl']['sum_kernel_ms']}
        for x in trace['ranks']]}
out = {'metrics': metrics, 'p10_latency_change_percent': comparisons,
       'paired_runs': [x for x in fair['rows'] if x['attempt'] in attempts[-2:]],
       'native': [x for x in native['summary'] if x['attempt'] in attempts[-2:]],
       'trace': traces,
       'limits': ['Negative latency change is faster. Median of three per-run statistics, not a pooled percentile or confidence interval.',
                  'P9/P10 performance isolates Stream-K. P8/P10 traces additionally differ by padding repair.',
                  'Two profiled steps are diagnostic; profiling overhead/rank dispatch can change collective waits. They do not replace unprofiled TTFT.',
                  'Kernel sums are not additive TTFT, kernel union is not SM occupancy. Empty timeline intervals do not identify the CPU root cause.',
                  'Full-model numerical/cache parity remains inconclusive; no production acceptance.']}
(R / 'p10-streamk-analysis.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(comparisons, indent=2))
