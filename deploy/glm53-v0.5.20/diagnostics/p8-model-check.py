"""Internal-only campaign runner. No model changes or automatic retries."""
import hashlib,json,pathlib,subprocess,sys,time,urllib.request,uuid
OUT=pathlib.Path('/results/prefill16k-20260926/p8-expertlayout-r1/diagnostic');OUT.mkdir(parents=True,exist_ok=True)
BASE='http://glm53-pf16-26-direct:8080'
def request(path,body=None,timeout=30):
 r=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=timeout) as response:return response.read()
def status(stage,**kw):
 d={'stage':stage,'time':time.time(),**kw};(OUT/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
status('checking_ready')
request('/health');info=json.loads(request('/server_info'));(OUT/'server-info.json').write_text(json.dumps(info,indent=2))
expected={'hicache_size':64,'max_total_tokens':600000,'tp_size':8,'ep_size':8,'dp_size':1,'pp_size':1,'attn_cp_size':8,'dcp_size':4,'chunked_prefill_size':16384,'schedule_policy':'hrrn','prefill_interleaving_mode':'adaptive','speculative_algorithm':'DFLASH','moe_runner_backend':'humming','min_free_slots_delay':1,'glm53_dcp_prefill_mode':'kv-gather','glm53_dsa_indexer_mode':'legacy','glm53_hicache_event_sync':'sync','glm53_prefill_attention_graph':'off','init_expert_location':'/scripts/expert-layout.json','ep_dispatch_algorithm':'dynamic','enable_eplb':False,'ep_num_redundant_experts':0}
for key,value in expected.items():
 if info.get(key)!=value:raise RuntimeError(f'Effective runtime mismatch {key}: {info.get(key)!r} != {value!r}')
# A health response is not generation proof. Exact nonce with natural stop is required.
nonce='PF16_'+uuid.uuid4().hex[:12]
body={'model':'alpha-fm','messages':[{'role':'user','content':'Return exactly this string and nothing else: '+nonce}],'temperature':0.6,'max_tokens':4096,'chat_template_kwargs':{'reasoning_effort':'low'},'stream':False}
status('generation_smoke')
r=json.loads(request('/v1/chat/completions',body,timeout=600));(OUT/'smoke.json').write_text(json.dumps({'request':body,'response':r},indent=2))
c=r['choices'][0];assert c['finish_reason']=='stop' and c['message'].get('content','').strip()==nonce, 'Nonce smoke failed; do not benchmark'
status('smoke_passed')

import random,torch
status('model_reference_comparison')
comparisons=[]
assert request('/flush_cache?timeout=60',timeout=90).decode().startswith('Cache flushed.')

for i,seed in enumerate([546620,547620,548620]):
 if i:assert request('/flush_cache?timeout=60',timeout=90).decode().startswith('Cache flushed.')
 rng=random.Random(seed);ids=[rng.randrange(100,30000) for _ in range(131072)]
 body={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':1,'ignore_eos':True},'return_logprob':True,'logprob_start_len':-1,'top_logprobs_num':20,'stream':False}
 started=time.time();response=json.loads(request('/generate',body,timeout=600));assert response['meta_info']['completion_tokens']==1
 (OUT/f'fresh-{seed}.json').write_text(json.dumps({'seed':seed,'input_tokens':len(ids),'elapsed_s':time.time()-started,'response':response}))
 ref=json.loads(pathlib.Path(f'/results/prefill16k-20260926/p7-expertstats-r1/diagnostic/fresh-{seed}.json').read_text())['response']
 def first_top(resp):
  values=resp['meta_info']['output_top_logprobs'][0]
  if len(values)==1 and isinstance(values[0][0],list):values=values[0]
  return {x[1]:x[0] for x in values}
 a,b=first_top(ref),first_top(response);shared=a.keys()&b.keys()
 comparisons.append(dict(seed=seed,first_output_id_equal=ref['output_ids'][0]==response['output_ids'][0],top20_exact=a==b,shared_top_tokens=len(shared),max_abs_shared_logprob_difference=max(abs(a[x]-b[x]) for x in shared)))
 status('compared',completed_requests=i+1)

(OUT/'model-check-comparisons.json').write_text(json.dumps(comparisons,indent=2))
status('model_reference_comparison_complete',completed_requests=3,all_first_output_ids_equal=all(x['first_output_id_equal'] for x in comparisons),all_top20_exact=all(x['top20_exact'] for x in comparisons),full_model_numerical_gate='limited sampled top20 comparison; full-model parity is not established',not_performance_comparison=True)
