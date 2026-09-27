"""Client-visible latency by request class; does not infer pure queue time."""
import os
import json, pathlib, statistics
R=pathlib.Path(os.environ["GLM53_CAMPAIGN_ROOT"]).resolve()

def quantile(values,q):
 v=sorted(values);p=(len(v)-1)*q;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
rows=[]
for attempt in ['p2-r4','p3-r1','p4-r1','p5-r1','p6graph-r1','p6buckets-r1','p8-expertlayout-r1','p9-padids-r1','p10-streamk-r1']:
 root=R/'collected'/(attempt+'-snapshot')/('bench-b' if attempt=='p2-r4' else 'bench')
 if not root.exists():continue
 for f in sorted(root.glob('*/requests.jsonl')):
  meta=json.loads((f.parent/'run.json').read_text());case=meta['scenario']
  if case not in ['mixed','constant']:continue
  req=[json.loads(l) for l in f.read_text().splitlines()];short=[q for q in req if q['input_tokens']<131072 and q['ok']];long=[q for q in req if q['input_tokens']==131072 and q['ok']]
  assert len(long)==1
  long=long[0];long_first=long['actual_start_s']+long['ttft_s'];gaps=[b[0]-a[0] for q in short for a,b in zip(q['token_updates'],q['token_updates'][1:])]
  rows.append({'attempt':attempt,'run':f.parent.name,'scenario':case,'short_count':len(short),'errors':sum(not q['ok'] for q in req),'short_ttft_p50_s':quantile([q['ttft_s'] for q in short],.5),'short_ttft_p95_s':quantile([q['ttft_s'] for q in short],.95),'short_e2e_p95_s':quantile([q['elapsed_s'] for q in short],.95),'short_stream_update_gap_p95_s':quantile(gaps,.95),'short_stream_update_gap_max_s':max(gaps),'short_decode_interval_mean_s':statistics.mean((q['elapsed_s']-q['ttft_s'])/(q['output_tokens']-1) for q in short),'long_ttft_s':long['ttft_s'],'long_e2e_s':long['elapsed_s'],'short_arrived_before_long_first':sum(q['actual_start_s']<long_first for q in short),'short_answered_before_long_first':sum(q['actual_start_s']+q['ttft_s']<long_first for q in short)})
summary=[]
for a in dict.fromkeys(x['attempt'] for x in rows):
 for case in ['mixed','constant']:
  group=[r for r in rows if r['attempt']==a and r['scenario']==case]
  summary.append({'attempt':a,'scenario':case,'repeats':len(group),**{k:statistics.median(r[k] for r in group) for k in ['short_ttft_p50_s','short_ttft_p95_s','short_e2e_p95_s','short_decode_interval_mean_s','short_stream_update_gap_p95_s','long_ttft_s','long_e2e_s']},'short_arrived_before_long_first':sum(r['short_arrived_before_long_first'] for r in group),'short_answered_before_long_first':sum(r['short_answered_before_long_first'] for r in group)})
out={'rows':rows,'summary':summary,'limits':'TTFT includes queue, prefill and first-token work. Queue cannot be separately reconstructed from this client log. Quantiles summarized as median of three per-run quantiles. Stream-update gaps are not individual token ITL. Seeds fixed across variants, no old-image baseline rerun.'};(R/'queue-fairness-results.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
