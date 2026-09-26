"""Match ordered MoE ReduceScatter kernels; compare rank entry skew and Humming work."""
import argparse,collections,gzip,json,pathlib,re,statistics
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();R=pathlib.Path(__file__).resolve().parent
ranks={};bases=[]
for f in sorted((R/'collected'/(a.attempt+'-snapshot')/'traces').glob('*.json.gz')):
 rank=int(re.search(r'TP-(\d+)-',f.name).group(1));d=json.loads(gzip.decompress(f.read_bytes()));bases.append(d.get('baseTimeNanoseconds'));events=sorted((x for x in d['traceEvents'] if x.get('cat')=='kernel' and x.get('ph')=='X'),key=lambda x:x['ts']);rs=[x for x in events if 'nccl' in x['name'].lower() and 'ReduceScatter' in x['name']];hm=[x for x in events if 'humming' in x['name'].lower()];ranks[rank]=(rs,hm)
assert len(ranks)==8 and len(set(bases))==1,(len(ranks),bases)
assert {len(v[0]) for v in ranks.values()}=={150}
rows=[]
for i in range(150):
 kernels={r:v[0][i] for r,v in ranks.items()};late=max(kernels,key=lambda r:kernels[r]['ts']);early=min(kernels,key=lambda r:kernels[r]['ts']);h={}
 for rank,(rs,hm) in ranks.items():
  current=rs[i];prev_end=rs[i-1]['ts']+rs[i-1]['dur'] if i else float('-inf');hs=[e for e in hm if prev_end<=e['ts']<current['ts']];assert len(hs)==2,(rank,i,len(hs));h[rank]=sum(x['dur'] for x in hs)/1000
 row={'i':i,'late_rank':late,'early_rank':early,'start_skew_ms':(kernels[late]['ts']-kernels[early]['ts'])/1000,'end_skew_ms':(max(x['ts']+x['dur'] for x in kernels.values())-min(x['ts']+x['dur'] for x in kernels.values()))/1000,'min_rs_ms':min(x['dur'] for x in kernels.values())/1000,'max_rs_ms':max(x['dur'] for x in kernels.values())/1000,'late_humming_ms':h[late],'early_humming_ms':h[early],'max_humming_ms':max(h.values()),'min_humming_ms':min(h.values())};rows.append(row)
summary={k:statistics.median(x[k] for x in rows) for k in rows[0] if k not in ['i','late_rank','early_rank']};summary['sum_max_humming_ms']=sum(x['max_humming_ms'] for x in rows);summary['late_ranks']=dict(collections.Counter(x['late_rank'] for x in rows));out={'attempt':a.attempt,'matched_collectives':150,'summary':summary,'rows':rows,'limits':'Two steps under profiler, matched by ordered150MoEreduce-scatters, each preceded by2Hummingkernels. Same trace clock across8ranks. Timing correlation; not fullTTFT decomposition or SMutilization.'};(R/(a.attempt+'-moe-skew.json')).write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
