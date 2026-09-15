"""Reproducible fixed-concurrency native streaming decode benchmark (aiohttp)."""
import argparse
import asyncio
import bisect
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
import aiohttp
import yaml


def sha(data): return hashlib.sha256(data).hexdigest()

def percentile(values, p):
    xs=sorted(values)
    return xs[min(len(xs)-1, math.ceil(p*len(xs))-1)] if xs else None


def prepare(a):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True, local_files_only=True)
    vocab=sorted(set(tok.get_vocab().values())-set(tok.all_special_ids))
    rng=random.Random(a.seed)
    data={'seed':a.seed, 'input_tokens':a.input_tokens, 'output_tokens':a.output_tokens,
          'tokenizer_vocab_sha256':sha(json.dumps(sorted(tok.get_vocab().items())).encode()),
          'requests':[rng.choices(vocab,k=a.input_tokens) for _ in range(a.concurrency)]}
    Path(a.output).write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(sha(Path(a.output).read_bytes()))


def summarize(rows, warmup=64, min_window=10):
    reasons=[]
    complete=[r for r in rows if not r.get('error') and r.get('events')]
    if len(complete)!=len(rows): reasons.append('request_failure_or_missing_token_counts')
    if any(len(r['events']) < 2 or r['events'][-1][1] != r['wanted'] for r in complete):
        reasons.append('incomplete_generation')
    starts=[]; ends=[]
    for r in complete:
        ev=r['events']; wanted=r['wanted']
        left=next((t for t,n in ev if n>=warmup),None)
        right=next((t for t,n in ev if n>=wanted-warmup),None)
        if left is None or right is None: reasons.append('insufficient_warmup_or_tail'); continue
        starts.append(left); ends.append(right)
    t0=max(starts,default=0); t1=min(ends,default=0)
    duration=t1-t0
    if duration < min_window: reasons.append('no_sufficient_common_resident_decode_window')
    tokens=0; gaps=[]; per_request=[]; coalesced=False
    if duration>0:
        for r in complete:
            ev=r['events']; times=[x[0] for x in ev]
            i=bisect.bisect_right(times,t0)-1; j=bisect.bisect_right(times,t1)-1
            n=(ev[j][1] if j>=0 else 0)-(ev[i][1] if i>=0 else 0)
            tokens+=n
            if n: per_request.append(duration*1000/n)
            for (ta,na),(tb,nb) in zip(ev,ev[1:]):
                if ta>=t0 and tb<=t1 and nb>na:
                    gaps.append((tb-ta)*1000)
                    coalesced |= nb-na!=1
    return {'valid':not reasons,'invalid_reasons':sorted(set(reasons)),
            'concurrency':len(rows),'successful':len(complete),
            'window_start':t0,'window_end':t1,'window_seconds':max(0,duration),
            'decode_output_tokens':tokens,'decode_output_tokens_per_second':tokens/duration if duration>0 else None,
            'per_request_decode_ms_per_token_median':statistics.median(per_request) if per_request else None,
            'ttft_p50_ms':percentile([(r['events'][0][0]-r['sent'])*1000 for r in complete],.5),
            'ttft_p95_ms':percentile([(r['events'][0][0]-r['sent'])*1000 for r in complete],.95),
            'stream_gap_p95_ms':percentile(gaps,.95),'stream_gap_p99_ms':percentile(gaps,.99),
            'stream_events_coalesced':coalesced,
            'stream_gap_note':'Client SSE arrival gaps; only single-token events approximate ITL.'}


def retract_counter(text):
    rows=[line for line in text.splitlines() if line.startswith('sglang:num_retracted_requests_total')]
    return sum(float(x.split()[-1]) for x in rows) if rows else None


async def run(a):
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    raw=Path(a.dataset).read_bytes(); data=json.loads(raw)
    manifest=list(yaml.safe_load_all(Path(a.manifest).read_text()))[0]
    c=manifest['spec']['template']['spec']['containers'][0]
    if '@sha256:' not in c['image']: raise ValueError('Pin the manifest image by digest before measuring')
    pods=json.loads(Path(a.pod_json).read_text())['items']
    if len(pods)!=1: raise ValueError('Expected exactly one lab pod')
    pod=pods[0]
    status=next(x for x in pod['status']['containerStatuses'] if x['name']==c['name'])
    actual=next(x for x in pod['spec']['containers'] if x['name']==c['name'])
    if actual['image']!=c['image'] or not status.get('ready') or status.get('restartCount',0)!=0 or not status.get('imageID'):
        raise ValueError('Pod image/readiness/restart status does not match a clean arm')
    if actual.get('args')!=c['args'] or actual.get('env')!=c['env'] or actual.get('command')!=c['command']:
        raise ValueError('Pod effective specification differs from the manifest')
    (out/'pod-before.json').write_text(Path(a.pod_json).read_text())
    (out/'manifest.yaml').write_text(Path(a.manifest).read_text())
    headers={'Authorization':'Bearer '+a.api_key} if a.api_key else {}
    timeout=aiohttp.ClientTimeout(total=a.timeout,sock_read=a.timeout)
    async with aiohttp.ClientSession(timeout=timeout,headers=headers,connector=aiohttp.TCPConnector(limit=0)) as session:
        async def get(path):
            async with session.get(a.url.rstrip('/')+path) as response:
                response.raise_for_status(); return await response.text()
        info=json.loads(await get('/server_info'))
        (out/'server-info-before.json').write_text(json.dumps(info,indent=2))
        server=info.get('server_args',info)
        if not isinstance(server,dict): raise ValueError('server_args is not a JSON object; inspect saved server-info')
        expected={'max_running_requests':len(data['requests']), 'speculative_algorithm':None,
                  'enable_hierarchical_cache':False,'enable_mixed_chunk':False,'stream_interval':1}
        for key,want in expected.items():
            if key not in server or server[key]!=want:
                raise ValueError(f'Not a decode-only profile: effective {key}={server.get(key)!r}; expected {want!r}')
        for flag in ('tp-size','pp-size','ep-size','dcp-size','model-path','kv-cache-dtype'):
            i=c['args'].index('--'+flag); want=c['args'][i+1]
            key=flag.replace('-','_')
            if str(server.get(key))!=str(want): raise ValueError(f'Effective {key} does not match manifest')
        if data['input_tokens']+data['output_tokens']>server.get('context_length',0):
            raise ValueError('Dataset exceeds the effective context length')
        # This endpoint only flushes an idle engine. Refusal is an error, not a retry loop.
        async with session.post(a.url.rstrip('/')+'/flush_cache') as response:
            response.raise_for_status(); await response.read()
        before=await get('/metrics'); (out/'metrics-before.txt').write_text(before)
        start=time.perf_counter()
        def now(): return time.perf_counter()-start
        async def request(index, ids):
            row={'index':index,'sent':now(),'wanted':data['output_tokens'],'events':[]}
            body={'input_ids':ids,'stream':True,'sampling_params':{
                'temperature':0,'max_new_tokens':data['output_tokens'],'ignore_eos':True}}
            try:
                async with session.post(a.url.rstrip('/')+'/generate',json=body) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if not line.startswith(b'data:'): continue
                        payload=line[5:].strip()
                        if payload==b'[DONE]': break
                        value=json.loads(payload)
                        if value.get('error') or value.get('object')=='error': raise RuntimeError(str(value))
                        meta=value.get('meta_info',{})
                        finish=meta.get('finish_reason')
                        if isinstance(finish,dict) and finish.get('type')=='abort': raise RuntimeError(str(finish))
                        n=meta.get('completion_tokens')
                        if isinstance(n,int) and n>0 and (not row['events'] or n>row['events'][-1][1]):
                            row['events'].append([now(),n])
                    row['finished']=now()
            except Exception as e: row['error']=repr(e)
            print(f"completed {index+1}/{len(data['requests'])}: tokens={row['events'][-1][1] if row['events'] else 0}",flush=True)
            return row
        rows=await asyncio.gather(*(request(i,ids) for i,ids in enumerate(data['requests'])))
        (out/'requests.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
        after=await get('/metrics'); (out/'metrics-after.txt').write_text(after)
        info_after=json.loads(await get('/server_info'))
        (out/'server-info-after.json').write_text(json.dumps(info_after,indent=2))
    result=summarize(rows,min_window=a.min_window)
    if info.get('startup_time') != info_after.get('startup_time'):
        result['valid']=False; result['invalid_reasons'].append('server_restarted')
    old,new=retract_counter(before),retract_counter(after)
    result['retractions_before']=old; result['retractions_after']=new
    if old is None or new is None or old!=new:
        result['valid']=False; result['invalid_reasons'].append('retractions_or_missing_counter')
    result.update(dataset_sha256=sha(raw),image=c['image'],label=a.label,
                  input_tokens=data['input_tokens'],output_tokens=data['output_tokens'],
                  manifest_sha256=sha(Path(a.manifest).read_bytes()),
                  pod_uid=pod['metadata']['uid'],node=pod['spec']['nodeName'],image_id=status['imageID'])
    (out/'requests.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['valid']: raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--tokenizer',required=True)
    q.add_argument('--concurrency',type=int,default=80); q.add_argument('--input-tokens',type=int,default=8192)
    q.add_argument('--output-tokens',type=int,default=2048); q.add_argument('--seed',type=int,default=12345)
    q.add_argument('--output',required=True)
    q=sub.add_parser('run'); q.add_argument('--url',required=True); q.add_argument('--dataset',required=True)
    q.add_argument('--manifest',required=True); q.add_argument('--pod-json',required=True); q.add_argument('--label',required=True); q.add_argument('--output',required=True)
    q.add_argument('--timeout',type=float,default=3600); q.add_argument('--min-window',type=float,default=10)
    q.add_argument('--api-key',default=None)
    a=p.parse_args()
    if a.command=='prepare':
        if min(a.concurrency,a.input_tokens)<=0 or a.output_tokens<=128: p.error('Positive inputs and output >128 required')
        prepare(a)
    else: asyncio.run(run(a))


if __name__=='__main__': main()
