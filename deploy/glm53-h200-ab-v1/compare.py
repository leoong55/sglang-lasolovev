"""Reject mismatched A/B inputs, then compare resident decode measurements."""
import argparse
import json
from pathlib import Path
import yaml


def argv_map(args):
    result={}; key=None
    for value in args:
        if value.startswith('--'):
            key=value
            if key in result: raise ValueError('Duplicate argument: '+key)
            result[key]=[]
        elif key is None: raise ValueError('Value before argument')
        else: result[key].append(value)
    return result


def check(a, b, mode):
    for key in ('dataset_sha256','concurrency','input_tokens','output_tokens','node'):
        if a['summary'][key]!=b['summary'][key]: raise ValueError('Different '+key)
    if not a['summary']['valid'] or not b['summary']['valid']: raise ValueError('Invalid benchmark arm')
    ca=a['container']; cb=b['container']
    allowed_args=set(); allowed_env=set()
    if mode=='topology':
        allowed_args={'--tp-size','--pp-size','--ep-size','--dcp-size','--pp-max-micro-batch-size',
            '--enable-prefill-cp','--cp-strategy','--enable-cp-decode-attn-tp','--dcp-comm-backend',
            '--cuda-graph-max-bs-decode','--cuda-graph-bs-decode'}
        allowed_env={'SGLANG_ENABLE_CP_V2','SGLANG_PP_LAYER_PARTITION'}
    elif mode=='park': allowed_env={'SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL'}
    elif mode=='skip': allowed_env={'SGLANG_ENABLE_H200_SKIP_NOT_FITTING'}
    for field, allowed in (('args',allowed_args),('env',allowed_env)):
        maps=[argv_map(c[field]) if field=='args' else {e['name']:e['value'] for e in c[field]} for c in (ca,cb)]
        diff={k for k in maps[0].keys()|maps[1].keys() if maps[0].get(k)!=maps[1].get(k)}
        if diff-allowed: raise ValueError(f'Uncontrolled {field} differences: {sorted(diff-allowed)}')
    for key in ('image','resources','volumeMounts','command'):
        if key=='image' and mode=='image': continue
        if ca.get(key)!=cb.get(key): raise ValueError('Different container '+key)
    for key in ('affinity','nodeSelector','volumes'):
        if a['pod'].get(key)!=b['pod'].get(key): raise ValueError('Different pod '+key)
    # Compare actual resolved options too, including defaults changed by PP.
    if a['summary']['image']!=ca['image'] or b['summary']['image']!=cb['image']:
        raise ValueError('Summary/manifest image mismatch')
    if mode=='image' and ca['image']==cb['image']: raise ValueError('Image comparison has identical images')


def load(path):
    path=Path(path); manifest=list(yaml.safe_load_all((path/'manifest.yaml').read_text()))[0]
    pod=manifest['spec']['template']['spec']
    return {'summary':json.loads((path/'summary.json').read_text()),'pod':pod,'container':pod['containers'][0]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('a'); p.add_argument('b')
    p.add_argument('--mode',choices=('topology','park','skip','image'),required=True)
    x=p.parse_args(); a=load(x.a); b=load(x.b)
    try: check(a,b,x.mode)
    except ValueError as e: p.error(str(e))
    for key in ('decode_output_tokens_per_second','per_request_decode_ms_per_token_median','stream_gap_p95_ms','stream_gap_p99_ms'):
        va=a['summary'][key]; vb=b['summary'][key]
        print(f'{key}: A={va:.3f}, B={vb:.3f}, B/A={vb/va:.4f}')
    print('TTFT is reported separately and is not included in the resident decode throughput.')


if __name__=='__main__': main()
