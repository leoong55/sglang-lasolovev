"""Prepare the unreleased P9 control / P10 candidate pair; never launch Jobs."""
import ast,copy,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parent
base=json.loads((R/'p9-padids-r1-serve.json').read_text())
bundle=(R.parents[1]/'reports/glm53-prefill16k-series-20260926/source-streamk-r1.tar.gz').read_bytes()
sha=hashlib.sha256(bundle).hexdigest();assert len(bundle)<1200000
model_check=(R/'p8-model-check.py').read_text()
for attempt,disabled,role in [('p9-padids-r1','0','known-nondeterministic-control'),('p10-streamk-r1','1','candidate')]:
 m=json.loads(json.dumps(base).replace('p9-padids-r1',attempt));cm=next(x for x in m['items'] if x.get('data'));job=next(x for x in m['items'] if x['kind']=='Job')
 profile=json.loads(cm['data']['profile.json']);profile['env']['SGLANG_GLM53_HUMMING_DISABLE_STREAM_K']=disabled
 profile.update(runtime_source='fe1dcba13',experiment_role=role,status='experimental comparison; not an accepted profile',comparison_parent='p8-expertlayout-r1' if disabled=='0' else 'p9-padids-r1')
 profile['argv'][profile['argv'].index('--weight-version')+1]=f'prefill16k-fe1dcba13-{attempt}-hc64'
 cm['data']['profile.json']=json.dumps(profile,indent=2);cm['data']['bundle.sha256']=sha
 serve=cm['data']['serve.py'].replace('source-padids-r1','source-streamk-r1')
 start=serve.index("arithmetic=json.loads(");end=serve.index("os.environ['EXPERT_REMAP_PROBE_OUTPUT']",start)
 serve=serve[:start]+'''arithmetic=json.loads(pathlib.Path('/results/prefill16k-20260926/expert-arithmetic-r3/arithmetic/complete.json').read_text())
assert arithmetic['status']=='component_reference_gates_passed' and arithmetic['unique_cases']==36 and arithmetic['ranks_count']==8
# This passes the corrected candidate's component gate, NOT the control's repeatability.
role=profile['experiment_role'];disabled=profile['env']['SGLANG_GLM53_HUMMING_DISABLE_STREAM_K']
if role=='known-nondeterministic-control':
 assert disabled=='0'
 diag=json.loads(pathlib.Path('/results/prefill16k-20260926/humming-drift-r1/drift/complete.json').read_text())
 assert diag['status']=='diagnostic_completed'
 native=[q for rank in diag['ranks'] for q in rank if q['mode']=='native_same_routes']
 assert len(native)==64 and all(q['graph_reference_l2']<=.08 and q['eager_reference_l2']<=.08 for q in native)
 print('KNOWN_STREAM_K_REPEATABILITY_FAILURE_CONTROL: experimental measurements only; not accepted',flush=True)
else:
 assert role=='candidate' and disabled=='1'
(out/'numerical-scope.json').write_text(json.dumps({'role':role,'candidate_component_gate':'expert-arithmetic-r3 passed36x8','control_repeatability':'known failure' if disabled=='0' else 'not selected','cross_layout_strict_parity':False,'full_model_parity':False},indent=2))
''' +serve[end:]
 cm['data']['serve.py']=serve
 # Check exact runtime revision in every client before nonce/generation.
 for key in ['bench.py']:
  text=cm['data'][key]
  marker="expected={"
  text=text.replace(marker,"expected={'weight_version':"+repr(profile['argv'][profile['argv'].index('--weight-version')+1])+",",1)
  cm['data'][key]=text
 check=model_check.replace('p8-expertlayout-r1',attempt)
 check=check.replace('expected={',"expected={'weight_version':"+repr(profile['argv'][profile['argv'].index('--weight-version')+1])+",",1)
 if disabled=='1':check=check.replace('/p7-expertstats-r1/diagnostic/','/p9-padids-r1/diagnostic/')
 cm['data']['model-check.py']=check
 for k,v in cm['data'].items():
  if k.endswith('.py'):ast.parse(v,filename=k)
 spec=job['spec']['template']['spec'];container=spec['containers'][0]
 container['env']=[{'name':k,'value':v} for k,v in profile['env'].items()]
 for v in spec['volumes']:
  if v['name'].startswith('bundle'):v['configMap']['name']='glm53-pf16-26-streamk-bundle-'+v['name'][6:]
 (R/(attempt+'-serve.json')).write_text(json.dumps(m,indent=2)+'\n');(R/(attempt+'-profile.json')).write_text(json.dumps(profile,indent=2)+'\n')
 bj=json.loads((R/'p9-padids-r1-bench.json').read_text());bj=json.loads(json.dumps(bj).replace('p9-padids-r1',attempt));(R/(attempt+'-bench.json')).write_text(json.dumps(bj,indent=2)+'\n')
 cj=copy.deepcopy(bj);cj['metadata']['name']=f'glm53-pf16-26-{attempt}-modelcheck';cj['spec']['template']['spec']['containers'][0]['command']=['python3','-u','/scripts/model-check.py'];(R/(attempt+'-modelcheck.json')).write_text(json.dumps(cj,indent=2)+'\n')
 print('Prepared',attempt,role,'Stream-K disabled=',disabled)
a=json.loads((R/'p9-padids-r1-profile.json').read_text());b=json.loads((R/'p10-streamk-r1-profile.json').read_text())
assert {k:v for k,v in a['env'].items() if k!='SGLANG_GLM53_HUMMING_DISABLE_STREAM_K'}=={k:v for k,v in b['env'].items() if k!='SGLANG_GLM53_HUMMING_DISABLE_STREAM_K'}
a['argv'][a['argv'].index('--weight-version')+1]='source';b['argv'][b['argv'].index('--weight-version')+1]='source';assert a['argv']==b['argv']
for attempt in ['p9-padids-r1','p10-streamk-r1']:
 m=json.loads((R/(attempt+'-serve.json')).read_text());cm=next(x for x in m['items'] if x.get('data'))
 assert hashlib.sha256(cm['data']['workload.py'].encode()).hexdigest()=='f52b0d77048b15cca149bd0235ac3e62f8c340e70df0edfac3dbeb9d5fc185df'
print('Pair differs by exactly one compute switch; workload unchanged; no launch.')
