"""Diagnose per-layer EP work and propose an offline, non-replicated layout.

Counts come from the native stat recorder, which sums across ranks. Ratios
are meaningful even if replicated CP/EP observations count tokens repeatedly.
An offline load prediction is not a latency benchmark or a correctness gate.
"""
import hashlib
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
raw = data['logical_count']['values']
shape = data['logical_count']['shape']
assert len(shape) == 3 and shape[2] % 8 == 0
layers, experts = shape[1:]
step_totals = [sum(map(sum, step)) for step in raw]
assert max(step_totals) > 0, 'Recorder returned no nonzero counts'
ids = [i for i, total in enumerate(step_totals) if total >= max(step_totals) / 4]
assert len(ids) >= 3, 'Too few comparable prefill steps'
split = max(1, len(ids) * 2 // 3)
train, heldout = ids[:split], ids[split:]


def aggregate(indices):
    return [[sum(raw[s][l][e] for s in indices) for e in range(experts)]
            for l in range(layers)]


weights = aggregate(train)
mapping = []
for counts in weights:
    bins, loads = [[] for _ in range(8)], [0] * 8
    # Same capacity-constrained longest-processing-time greedy idea as
    # deepseek-ai/EPLB balanced_packing. Stable expert-ID tie breaking.
    for expert in sorted(range(experts), key=lambda e: (-counts[e], e)):
        rank = min((r for r in range(8) if len(bins[r]) < experts // 8),
                   key=lambda r: (loads[r], r))
        bins[rank].append(expert)
        loads[rank] += counts[expert]
    row = [e for group in bins for e in group]
    assert sorted(row) == list(range(experts))
    mapping.append(row if sum(counts) else list(range(experts)))


def rank_loads(counts, layout):
    return [sum(counts[e] for e in layout[r * (experts // 8):(r + 1) * (experts // 8)])
            for r in range(8)]


def evaluate(indices, layout):
    ratios, critical, useful = [], 0, 0
    per_layer = []
    for l in range(layers):
        records = []
        for s in indices:
            loads = rank_loads(raw[s][l], layout[l])
            if not sum(loads):
                continue
            ratio = max(loads) / statistics.mean(loads)
            records.append(ratio)
            ratios.append(ratio)
            critical += max(loads)
            useful += sum(loads) / 8
        if records:
            per_layer.append(dict(layer=l, max_to_mean_median=statistics.median(records),
                                  max_to_mean_max=max(records)))
    assert ratios
    return dict(max_to_mean_median=statistics.median(ratios),
                max_to_mean_p95=sorted(ratios)[int(.95 * (len(ratios)-1))],
                weighted_utilization=useful / critical,
                sum_max_rank_count=critical, per_layer=per_layer)


trivial = [list(range(experts)) for _ in range(layers)]
summary = dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               shape=shape, retained_prefill_step_indices=ids,
               nonzero_step_totals={str(i):t for i,t in enumerate(step_totals) if t},
               train_step_indices=train, heldout_step_indices=heldout,
               scope='Native count ratios and predicted load, not measured speedup',
               recorder_average_utilization=data.get('average_utilization_rate_over_window'),
               caveats=['Synthetic random tokens; production routing may differ',
                        'Contiguous physical experts assumed in the observed control',
                        'Absolute summed counts may include duplicate rank observations',
                        'Holding out steps does not replace held-out real workloads'])
for subset, indices_ in [('all', ids), ('train', train), ('heldout', heldout)]:
    summary[subset] = dict(trivial=evaluate(indices_, trivial), balanced=evaluate(indices_, mapping))
    summary[subset]['predicted_critical_count_reduction'] = 1 - (
        summary[subset]['balanced']['sum_max_rank_count'] /
        summary[subset]['trivial']['sum_max_rank_count'])

(ROOT/'expert-count-analysis.json').write_text(json.dumps(summary, indent=2))
(ROOT/'expert-layout-candidate.json').write_text(json.dumps({'physical_to_logical_map': mapping}))
print(json.dumps({k:v for k,v in summary.items() if k not in ('all','train','heldout')}, indent=2))
for subset in ('all','train','heldout'):
    print(subset, json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='per_layer'}
                                if isinstance(v,dict) else v)
                             for k,v in summary[subset].items()}, indent=2))
