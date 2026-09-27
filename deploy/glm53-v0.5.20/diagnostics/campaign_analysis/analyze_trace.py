"""Read Kineto traces; report kernel-time sums and GPU interval unions separately."""
import os
import argparse,collections,gzip,json,pathlib
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--all-ranks',action='store_true');a=p.parse_args();R=pathlib.Path(os.environ["GLM53_CAMPAIGN_ROOT"]).resolve()
root=R/'collected'/(a.attempt+'-snapshot')/'traces'
def union_ms(intervals):
 if not intervals:return 0
 intervals=sorted(intervals);lo,hi=intervals[0];total=0
 for l,h in intervals[1:]:
  if l>hi:total+=hi-lo;lo,hi=l,h
  else:hi=max(hi,h)
 return (total+hi-lo)/1000

def group(name):
 if 'nccl' in name.lower():return 'nccl'
 if 'sparse_mla' in name.lower():return 'sparse_attention'
 if 'mqa_logits' in name:return 'indexer_logits'
 if 'humming' in name:return 'humming_gemm'
 if 'deep_gemm' in name:return 'other_deepgemm'
 if 'topk' in name.lower():return 'topk'
 if any(x in name for x in ['at::native','elementwise','where_kernel','reduce_kernel']):return 'torch_elementwise_reduce_copy'
 return 'other'
out=[]
for f in sorted(root.glob('*.json.gz')):
 if not a.all_ranks and 'TP-0-' not in f.name:continue
 trace=json.loads(gzip.decompress(f.read_bytes()));events=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0];agg=collections.defaultdict(lambda:[0,0]);intervals=collections.defaultdict(list);cpu=collections.defaultdict(lambda:[0,0]);names=collections.defaultdict(lambda:[0,0])
 for e in events:
  cat=e.get('cat');name=e.get('name','');dur=e['dur']
  if cat=='kernel':
   g=group(name);agg[g][0]+=1;agg[g][1]+=dur;interval=(e['ts'],e['ts']+dur);intervals[g].append(interval);intervals['all_kernels'].append(interval);names[name][0]+=1;names[name][1]+=dur
  if cat in ['cuda_runtime','cuda_driver']:
   cpu[name][0]+=1;cpu[name][1]+=dur
 all_intervals=intervals['all_kernels'];span=(max(h for _,h in all_intervals)-min(l for l,_ in all_intervals))/1000 if all_intervals else 0
 out.append({'trace':f.name,'kernel_span_ms':span,'kernel_union_ms':union_ms(all_intervals),'groups':{g:{'count':n,'sum_kernel_ms':t/1000,'union_ms':union_ms(intervals[g])} for g,(n,t) in agg.items()},'top_kernels':[{'name':n,'count':v[0],'sum_ms':v[1]/1000} for n,v in sorted(names.items(),key=lambda x:-x[1][1])[:20]],'top_cuda_api':[{'name':n,'count':v[0],'inclusive_cpu_ms':v[1]/1000} for n,v in sorted(cpu.items(),key=lambda x:-x[1][1])[:15]]})
result={'attempt':a.attempt,'ranks':out,'limits':'Measured under profiler for two steps. Kernel time sums and category unions can overlap; do not add category unions or ranks. Grouping is heuristic by kernel name, not causal layer attribution. All-kernel union is time with at least one recorded kernel, not SM occupancy.'};(R/(a.attempt+'-trace-analysis.json')).write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in r.items() if k not in ['top_kernels','top_cuda_api']} for r in out],indent=2))
