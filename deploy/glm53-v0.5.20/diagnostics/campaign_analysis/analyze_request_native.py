"""Matched final request metadata; separate scheduler queue and decode speculation."""
import os
import json,pathlib,statistics
R=pathlib.Path(os.environ["GLM53_CAMPAIGN_ROOT"]).resolve()

def q95(v):
 v=sorted(v);p=(len(v)-1)*.95;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
rows=[]
for a in ['p6buckets-r1','p8-expertlayout-r1','p9-padids-r1','p10-streamk-r1']:
 for f in sorted((R/'collected'/f'{a}-snapshot'/'bench').glob('*/requests.jsonl')):
  req=[json.loads(x) for x in f.read_text().splitlines()];case=json.loads((f.parent/'run.json').read_text())['scenario']
  for name,values in [('short',[x for x in req if x['input_tokens']<131072]),('long',[x for x in req if x['input_tokens']==131072])]:
   if not values:continue
   ms=[x['meta_info'] for x in values];assert all(x['ok'] for x in values)
   calls=sum(x['spec_verify_ct'] for x in ms);proposed=sum(x['spec_num_proposed_drafts'] for x in ms);accepted=sum(x['spec_num_correct_drafts'] for x in ms)
   rows.append({'attempt':a,'run':f.parent.name,'scenario':case,'class':name,'requests':len(values),'queue_p95_s':q95([x['queue_time'] for x in ms]),'receive_to_first_forward_p95_s':q95([x['forward_entry_time']-x['request_received_ts'] for x in ms]),'prefill_elapsed_p95_s':q95([x['prefill_finished_time']-x['forward_entry_time'] for x in ms]),'ttft_p95_s':q95([x['ttft_s'] for x in values]),'decode_interval_mean_s':statistics.mean((x['elapsed_s']-x['ttft_s'])/255 for x in values),'draft_accept_rate':accepted/proposed if proposed else None,'output_tokens_per_verify':sum(x['completion_tokens'] for x in ms)/calls if calls else None,'verify_calls':calls,'retractions':sum(x['num_retractions'] for x in ms)})
summary=[]
for a in ['p6buckets-r1','p8-expertlayout-r1','p9-padids-r1','p10-streamk-r1']:
 for case in ['fresh','mixed','constant','burst','repeated']:
  for cls in ['short','long']:
   rs=[x for x in rows if (x['attempt'],x['scenario'],x['class'])==(a,case,cls)]
   if rs:summary.append({'attempt':a,'scenario':case,'class':cls,**{k:statistics.median(x[k] for x in rs) for k in ['queue_p95_s','receive_to_first_forward_p95_s','prefill_elapsed_p95_s','ttft_p95_s','decode_interval_mean_s','draft_accept_rate','output_tokens_per_verify']}})
result={'rows':rows,'summary':summary,'limits':'Final per-request native metadata, no TP replication. Each summary is median of3per-run statistics. Different quantiles cannot be subtracted to decompose TTFT. Prefill elapsed includes scheduling gaps between chunks, not pure GPU time. Output/verify includes bonus tokens and is not accepted/proposed rate. No exact token-ITL claim.'};(R/'native-request-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps([x for x in summary if x['scenario']=='constant'],indent=2))
