"""Render isolated H200 lab profiles; print YAML without deploying anything."""
import argparse
import copy
import math
from pathlib import Path
import yaml

PROFILES = ('tp8-dcp4-decode', 'pp4-decode', 'pp4-archive')


def render(*, image, profile, concurrency=80, weights='w4afp8', model_pvc=None,
           context=131072, park=False, skip=False, name='sglang-glm53-h200-ab'):
    if profile not in PROFILES or concurrency < 1 or context < 4096:
        raise ValueError('Invalid profile/concurrency/context')
    if weights == 'fp8' and not model_pvc:
        raise ValueError('FP8 requires the actual --model-pvc name')
    obj = yaml.safe_load((Path(__file__).parent/'deployment.template.yaml').read_text())
    obj['metadata']['name'] = name
    obj['spec']['selector']['matchLabels']['app'] = name
    pod = obj['spec']['template']
    pod['metadata']['labels']['app'] = name
    pod['metadata']['annotations']['glm53-test-profile'] = profile
    spec = pod['spec']
    c = spec['containers'][0]
    c['image'] = image
    c['command'] = ['python3', '/opt/glm53-cp8-dcp4-v1/h200/launch.py']
    pp = 1 if profile == 'tp8-dcp4-decode' else 4
    tp = 8//pp
    micro = math.ceil(concurrency/pp)
    buckets = sorted({n for n in (1,2,4,8,16,20,24,32,40,48,64,80,96,120,128,160) if n <= micro} | {micro})
    model_path = '/mnt/model-pvc-fp8' if weights == 'fp8' else '/mnt/model-pvc-w4fp8'
    args = ['--host','0.0.0.0','--port','8080','--model-path',model_path,
            '--served-model-name','alpha-fm','--trust-remote-code',
            '--tp-size',str(tp),'--pp-size',str(pp),'--ep-size',str(tp),'--dp-size','1',
            '--dcp-size','4' if pp == 1 else '1', '--moe-a2a-backend','none',
            '--kv-cache-dtype','fp8_e4m3','--dsa-prefill-backend','flashmla_sparse_q8',
            '--dsa-decode-backend','flashmla_kv','--disable-shared-experts-fusion',
            '--page-size','64','--mem-fraction-static','0.80',
            '--context-length',str(context),'--max-running-requests',str(concurrency),
            '--chunked-prefill-size','4096','--prefill-decode-interval','0',
            '--cuda-graph-backend-prefill','disabled','--cuda-graph-backend-decode','full',
            '--cuda-graph-max-bs-decode',str(micro),'--cuda-graph-bs-decode',*map(str,buckets),
            '--stream-interval','1','--random-seed','12345',
            '--reasoning-parser','glm45','--tool-call-parser','glm47',
            '--enable-metrics','--enable-cache-report','--enforce-disable-flashinfer-allreduce-fusion']
    if weights == 'w4afp8':
        args += ['--quantization','w4afp8','--moe-runner-backend','humming']
    # Official FP8 checkpoint carries its own quantization config; native backend selection.
    if pp == 1:
        args += ['--enable-prefill-cp','--cp-strategy','interleave',
                 '--enable-cp-decode-attn-tp','--dcp-comm-backend','ag_rs']
    else:
        args += ['--pp-max-micro-batch-size',str(micro)]
    if profile == 'pp4-archive':
        args += ['--enable-hierarchical-cache','--hicache-size','96',
                 '--hicache-write-policy','write_back','--hicache-io-backend','direct',
                 '--hicache-mem-layout','layer_first','--enable-mixed-chunk']
    c['args'] = args
    env = {x['name']: x['value'] for x in c['env']}
    for key in list(env):
        if key.startswith('SGLANG_GLM53_') or key == 'SGLANG_ENABLE_CP_V2':
            env[key] = '0'
    env.update(SGLANG_ENABLE_CP_V2='1' if pp == 1 else '0',
               SGLANG_GLM53_HUMMING_EP_AWARE='1',
               SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL='1' if park else '0',
               SGLANG_ENABLE_H200_SKIP_NOT_FITTING='1' if skip else '0')
    if pp > 1:
        env['SGLANG_PP_LAYER_PARTITION'] = '21,20,20,17'
    c['env'] = [{'name':k,'value':v} for k,v in env.items()]
    for mount in c['volumeMounts']:
        if mount['name'] == 'model': mount['mountPath'] = model_path
    for vol in spec['volumes']:
        if vol['name'] == 'model': vol['persistentVolumeClaim']['claimName'] = model_pvc or 'sglang-w4fp8-pvc'
    service = {'apiVersion':'v1','kind':'Service','metadata':{'name':name,'namespace':'inf-glm53'},
               'spec':{'selector':{'app':name},'ports':[{'name':'http','port':8080,'targetPort':'http'}]}}
    return [obj, service]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--profile', choices=PROFILES, required=True)
    p.add_argument('--concurrency', type=int, default=80)
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--weights', choices=('w4afp8','fp8'), default='w4afp8')
    p.add_argument('--model-pvc')
    p.add_argument('--park', action='store_true')
    p.add_argument('--skip', action='store_true')
    p.add_argument('--name', default='sglang-glm53-h200-ab')
    a=p.parse_args()
    try: docs=render(**vars(a))
    except ValueError as e: p.error(str(e))
    print(yaml.safe_dump_all(docs, sort_keys=False), end='')


if __name__ == '__main__': main()
