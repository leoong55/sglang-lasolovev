"""Explicit, bounded validation workload for an isolated SGLang /generate server.

No default endpoint, no automatic cache flushing, no stored API credentials.
Run the same seed/arrival schedule on every variant with the same KV token cap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.request import Request, urlopen


def workload(scenario):
    if scenario=='fresh':return [(0,131072)]
    if scenario=='mixed':return [(0,131072),*((1,1024*n) for n in (4,5,6,7,10,15))]
    if scenario=='burst':return [(0,1024*(4+i%4)) for i in range(40)]
    if scenario=='constant':return [(0,131072),*((1+i*.5,4096) for i in range(80))]
    if scenario=='repeated':return [(0,33792) for _ in range(40)]
    raise ValueError(scenario)


def parse_sse(lines):
    for raw in lines:
        if not raw.startswith(b'data:'):continue
        data=raw[5:].strip()
        if data==b'[DONE]':return
        item=json.loads(data)
        if 'error' in item:raise RuntimeError(str(item['error']))
        yield item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True,help='Isolated server, e.g. http://127.0.0.1:8080')
    p.add_argument('--scenario',required=True,choices=['fresh','mixed','burst','constant','repeated'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=531620)
    p.add_argument('--output-tokens',type=int,default=256)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--token-env',default='GLM53_TEST_API_KEY')
    p.add_argument('--repeat-rounds',type=int,default=2,help='Repeated-prefix waves; first is cold if server cache is clean')
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory')
    if a.output_tokens<=0 or not 1<=a.repeat_rounds<=10:p.error('Invalid output tokens or round count')
    a.output.mkdir(parents=True)
    headers={'Content-Type':'application/json'}
    if os.getenv(a.token_env):headers['Authorization']='Bearer '+os.environ[a.token_env]
    def request(path,body=None):
        return urlopen(Request(a.base_url.rstrip('/')+path,
                              data=None if body is None else json.dumps(body).encode(),
                              headers=headers),timeout=a.timeout)
    stop=threading.Event()
    def metrics():
        while not stop.is_set():
            try:
                with request('/metrics') as response:
                    text=response.read().decode()
                with (a.output/'metrics.prom').open('a') as file:
                    file.write(f'# sample_unix_seconds {time.time()}\n'+text+'\n')
            except Exception as exc:
                # Exception types only: HTTP exception bodies may include request data.
                with (a.output/'metrics-errors.txt').open('a') as file:file.write(type(exc).__name__+'\n')
            stop.wait(1)
    sampler=threading.Thread(target=metrics,daemon=True);sampler.start()
    def run_one(case,base,round_index):
        i,(delay,length)=case
        rng=random.Random(a.seed+i)
        if a.scenario=='repeated':
            shared=random.Random(a.seed)
            ids=[shared.randrange(100,30000) for _ in range(32768)]+[rng.randrange(100,30000) for _ in range(length-32768)]
        else:ids=[rng.randrange(100,30000) for _ in range(length)]
        rid=f'prefill-{a.scenario}-{round_index}-{i}'
        body={'rid':rid,'input_ids':ids,'sampling_params':{'max_new_tokens':a.output_tokens,'temperature':0},'stream':True}
        time.sleep(max(0,base+delay-time.perf_counter()))
        started=time.perf_counter();first=None;last_tokens=0;token_times=[];meta={}
        result=dict(rid=rid,input_tokens=length,scheduled_s=delay,actual_start_s=started-base,round=round_index)
        try:
            with request('/generate',body) as response:
                for item in parse_sse(response):
                    now=time.perf_counter();meta=item.get('meta_info',{})
                    n=meta.get('completion_tokens',0)
                    if n>last_tokens:
                        if first is None:first=now
                        token_times.append((now-started,n));last_tokens=n
            if first is None or not meta.get('finish_reason') or meta['finish_reason'].get('type')=='abort':
                raise RuntimeError('Generation did not finish successfully')
            result.update(ok=True,ttft_s=first-started,elapsed_s=time.perf_counter()-started,
                          output_tokens=last_tokens,finish_reason=meta['finish_reason'],
                          token_updates=token_times,cached_tokens=meta.get('cached_tokens',0))
        except Exception as exc:result.update(ok=False,error_type=type(exc).__name__,elapsed_s=time.perf_counter()-started)
        return result
    all_results=[]
    try:
        rounds=a.repeat_rounds if a.scenario=='repeated' else 1
        for round_index in range(rounds):
            base=time.perf_counter()+2
            with ThreadPoolExecutor(max_workers=41) as pool:
                futures=[pool.submit(run_one,case,base,round_index) for case in enumerate(workload(a.scenario))]
                for future in as_completed(futures):
                    result=future.result();all_results.append(result)
                    with (a.output/'requests.jsonl').open('a') as file:file.write(json.dumps(result)+'\n')
                    print(result['rid'], 'ok' if result['ok'] else result['error_type'],flush=True)
    finally:stop.set();sampler.join(timeout=2)
    (a.output/'run.json').write_text(json.dumps(dict(scenario=a.scenario,seed=a.seed,output_tokens=a.output_tokens,
        completed=len(all_results),errors=sum(not r['ok'] for r in all_results),
        status='client measurements only; inspect scheduler metrics, GPU traces and memory separately'),indent=2)+'\n')
    if any(not r['ok'] for r in all_results):raise SystemExit(1)


if __name__=='__main__':main()
