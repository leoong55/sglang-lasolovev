"""Compare resident and reloaded prefixes with identical 256-token suffix work.

Cold full-prefill output is deliberately NOT the numerical reference: its
batch/quantization shape differs. No retries and no relaxed numerical tolerance.
"""
import argparse
import hashlib
import json
import pathlib
import random
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--attempt', choices=['p9-padids-r1', 'p10-streamk-r1'], required=True)
args = parser.parse_args()
ROOT = pathlib.Path('/results/prefill16k-20260926') / args.attempt
OUT = ROOT / 'cache-matched'
OUT.mkdir(exist_ok=False)
BASE = 'http://glm53-pf16-26-direct:8080'


def call(path, body=None):
    request = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def status(stage, **kwargs):
    record = dict(stage=stage, time=time.time(), **kwargs)
    (OUT/'status.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


# The measurement runner must have finished, including its own exact nonce gate.
# Never flush a running performance workload or test the wrong serving variant.
bench_status = json.loads((ROOT/'bench/status.json').read_text())
assert bench_status['stage'] == 'focused_workloads_complete'
assert bench_status['completed_scenarios'] == 15
smoke = json.loads((ROOT/'bench/smoke.json').read_text())
choice = smoke['response']['choices'][0]
nonce = smoke['request']['messages'][0]['content'].split('Return exactly this string and nothing else: ', 1)[1]
assert choice['finish_reason'] == 'stop'
assert choice['message']['content'].strip() == nonce
info = json.loads(call('/server_info'))
expected = dict(weight_version='prefill16k-fe1dcba13-' + args.attempt + '-hc64',
                hicache_size=64, max_total_tokens=600000, tp_size=8, ep_size=8,
                dp_size=1, pp_size=1, attn_cp_size=8, dcp_size=4,
                chunked_prefill_size=16384, speculative_algorithm='DFLASH',
                moe_runner_backend='humming', init_expert_location='/scripts/expert-layout.json',
                glm53_dcp_prefill_mode='kv-gather', glm53_dsa_indexer_mode='legacy',
                glm53_hicache_event_sync='sync', enable_eplb=False)
for key, value in expected.items():
    assert info.get(key) == value, (key, info.get(key), value)
(OUT/'server-info.json').write_text(json.dumps(info, indent=2))
status('preconditions_passed', attempt=args.attempt, performance_completed=True)
assert call('/flush_cache?timeout=60').decode().startswith('Cache flushed.')
SEED = 549620
responses = {}
completed = 0


def generate(name, seed, output_tokens):
    global completed
    rng = random.Random(seed)
    ids = [rng.randrange(100, 30000) for _ in range(131072)]
    body = dict(rid='cache-matched-' + name, input_ids=ids, stream=False,
                sampling_params=dict(temperature=0, max_new_tokens=output_tokens, ignore_eos=True),
                return_logprob=True, logprob_start_len=-1, top_logprobs_num=20)
    start = time.time()
    result = json.loads(call('/generate', body))
    meta = result['meta_info']
    assert meta['completion_tokens'] == output_tokens
    assert meta['finish_reason']['type'] != 'abort'
    record = dict(seed=seed, input_tokens=len(ids), elapsed_s=time.time()-start,
                  input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), response=result)
    (OUT/(name + '.json')).write_text(json.dumps(record, indent=2))
    completed += 1
    status(name, completed_requests=completed, cache=meta.get('cached_tokens_details'))
    return result


generate('cold-prime', SEED, 1)
for i in range(3):
    name = f'resident-before-{i}'
    responses[name] = generate(name, SEED, 32)
    assert responses[name]['meta_info']['cached_tokens_details'] == dict(device=130816, host=0)
# 21 independent 128k prefixes exceed the unchanged 2.4M logical GPU capacity.
for i in range(21):
    generate(f'evict-{i:02d}', SEED + 1 + i, 1)
responses['host-reload'] = generate('host-reload', SEED, 32)
assert responses['host-reload']['meta_info']['cached_tokens_details'] == dict(device=0, host=130816)
for i in range(3):
    name = f'resident-after-{i}'
    responses[name] = generate(name, SEED, 32)
    assert responses[name]['meta_info']['cached_tokens_details'] == dict(device=130816, host=0)


def first_top(result):
    values = result['meta_info']['output_top_logprobs'][0]
    # Some runtime versions wrap the first token's list in another dimension.
    if len(values) == 1 and isinstance(values[0][0], list):
        values = values[0]
    return {entry[1]: entry[0] for entry in values}


comparisons = []
reference = responses['resident-before-0']
for name, result in responses.items():
    a, b = first_top(reference), first_top(result)
    shared = sorted(a.keys() & b.keys())
    comparisons.append(dict(
        reference='resident-before-0', candidate=name,
        output_ids_equal=reference['output_ids'] == result['output_ids'],
        first_output_top20_exact=a == b, shared_top_tokens=len(shared),
        max_abs_logprob_difference_on_shared=max(abs(a[t]-b[t]) for t in shared),
        first_output_id_equal=reference['output_ids'][0] == result['output_ids'][0],
    ))
(OUT/'comparisons.json').write_text(json.dumps(comparisons, indent=2))
status('cache_matched_complete', completed_requests=completed,
       all_first_top20_exact=all(row['first_output_top20_exact'] for row in comparisons),
       all_32_output_ids_equal=all(row['output_ids_equal'] for row in comparisons),
       scope='Same resident/reloaded 130816-token prefix and 256-token fresh suffix; not full-logit parity')
