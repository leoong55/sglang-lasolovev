"""Summarize the bounded real-weight diagnostic; never mark it serving acceptance."""
import json,pathlib
R=pathlib.Path(__file__).resolve().parent
x=json.loads((R/'collected/humming-drift-r1-snapshot/drift/complete.json').read_text())
assert x['status']=='diagnostic_completed' and len(x['ranks'])==8
assert all(len(rows)==32 for rows in x['ranks'])
rows=sum(x['ranks'],[]);modes=[]
for mode in ['native_changed_routes','native_same_routes','frozen_sort','streamk_off']:
 a=[q for q in rows if q['mode']==mode]
 modes.append(dict(mode=mode,cases=len(a),strict_pass={key:sum(q[key]['strict_close'] for q in a) for key in ['eager_eager','graph_graph','graph_eager']},max_graph_eager_relative_l2=max(q['graph_eager']['relative_l2'] for q in a),max_fp32_reference_l2=max(max(q['graph_reference_l2'],q['eager_reference_l2']) for q in a),max_graph_eager_absolute_error=max(q['graph_eager']['max_abs'] for q in a)))
timings=[]
for n in [2048,16384]:
 for concentrated in [False,True]:
  for layout in ['identity','fixed']:
   a=[q for q in rows if q['rows']==n and q['concentrated']==concentrated and q['layout']==layout]
   t={mode:max(q['graph_mean_ms'] for q in a if q['mode']==mode) for mode in ['native_same_routes','streamk_off']}
   timings.append(dict(rows=n,concentrated=concentrated,layout=layout,rank_max_mean_graph_ms=t,latency_change_percent=100*(t['streamk_off']/t['native_same_routes']-1)))
result=dict(status='diagnostic_complete_not_acceptance',runtime_source='0b3735e42',cases_per_rank=32,ranks=8,modes=modes,timings=timings,conclusion='Native eager and graph both vary. Fixed sort partially helps; disabling Stream-K gives bitwise identical repeats across64cases. This localizes variation to Stream-K partial arithmetic, influenced by route order, rather than CUDA-graph metadata alone.',limitations=['One real layer3, synthetic hidden states and two routing distributions','Ten replay timings per point, not repeated full-model performance','No attention/KV/DFlash or full-model numerical acceptance','Fixed-sort uses immutable routes only as diagnostic; not a serving implementation'])
(R/'humming-drift-r1-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('Validated 256 component observations, Stream-K-off64/64exact; no full-model acceptance')
