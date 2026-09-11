"""Independent full GLM-5.3 W4AFP8 / H200 / TP2 PP4 experiment launcher."""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from install import install, package_root


def command(args):
    return [
        sys.executable, '-m', 'sglang.launch_server',
        '--model-path', args.model_path,
        '--served-model-name', 'GLM-5.3',
        '--host', '0.0.0.0', '--port', str(args.port),
        '--trust-remote-code', '--reasoning-parser', 'glm45', '--tool-call-parser', 'glm47',
        '--tp-size', '2', '--pp-size', '4', '--ep-size', '1', '--dp-size', '1', '--dcp-size', '1',
        '--pp-max-micro-batch-size', '10', '--pp-async-batch-depth', '0',
        '--max-running-requests', '40', '--context-length', '98304',
        '--quantization', 'w4afp8', '--kv-cache-dtype', 'fp8_e4m3', '--page-size', '64',
        '--attention-backend', 'dsa', '--dsa-prefill-backend', 'flashmla_kv',
        '--dsa-decode-backend', 'flashmla_kv', '--moe-a2a-backend', 'none',
        '--disable-shared-experts-fusion',
        '--mem-fraction-static', str(args.mem_fraction_static),
        '--chunked-prefill-size', '4096', '--max-prefill-tokens', '8192', '--enable-mixed-chunk',
        '--cuda-graph-backend-prefill', 'disabled', '--cuda-graph-backend-decode', 'full',
        '--cuda-graph-max-bs-decode', '16',
        '--enable-metrics', '--enable-metrics-for-all-schedulers',
        *([] if args.hicache_size == 0 else [
            '--enable-hierarchical-cache', '--hicache-size', str(args.hicache_size),
            '--hicache-write-policy', 'write_back', '--hicache-io-backend', 'kernel',
            '--hicache-mem-layout', 'page_first',
        ]),
        *([] if not args.l3_path else [
            '--hicache-storage-backend', 'file',
            '--hicache-storage-backend-extra-config', json.dumps({
                'max_size': args.l3_max_size, 'min_free_space': '50G',
            }),
        ]),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--model-path', default='/model')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--mem-fraction-static', type=float, default=0.90)
    p.add_argument('--hicache-size', type=int, default=128, help='Decimal GB per rank; 0 disables L2')
    p.add_argument('--l3-path', help='Optional mounted dedicated file-cache directory')
    p.add_argument('--l3-max-size', default='250G', help='Cap per PP stage; four stages total')
    p.add_argument('--print-command', action='store_true', help='Print profile without starting or loading the model')
    args = p.parse_args()
    if not 0.5 <= args.mem_fraction_static <= 0.95 or args.hicache_size < 0:
        p.error('Invalid memory fraction or HiCache size')
    if args.l3_path and args.hicache_size == 0:
        p.error('L3 requires HiCache')
    if int(os.environ.get('SGLANG_PP_SHORT_BYPASS_TOKENS', '512')) not in range(0, 513, 64):
        p.error('SGLANG_PP_SHORT_BYPASS_TOKENS must be 0..512, aligned to 64')
    if os.environ.get('SGLANG_PP_FULL_NEED', '1') not in ('0', '1'):
        p.error('SGLANG_PP_FULL_NEED must be 0 or 1')
    if os.environ.get('SGLANG_PP_LAYER_PARTITION', '21,20,20,17') != '21,20,20,17':
        p.error('This experiment pins the colleague partition: 21,20,20,17')
    # An independent launch must not inherit environment overrides from v3/v5.
    for key in ('SGLANG_ENABLE_CP_V2', 'SGLANG_GLM53_DEEPEP_PREFILL'):
        os.environ.pop(key, None)
    os.environ['SGLANG_PP_LAYER_PARTITION'] = '21,20,20,17'
    os.environ.setdefault('SGLANG_PP_FULL_NEED', '1')
    os.environ.setdefault('SGLANG_PP_SHORT_BYPASS_TOKENS', '512')
    os.environ['SGLANG_DSA_FUSE_TOPK'] = '0'
    if args.l3_path:
        os.environ['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR'] = args.l3_path
    argv = command(args)
    print('PP4 profile: ' + shlex.join(argv), flush=True)
    if args.print_command:
        return
    install(package_root(), Path(__file__).resolve().parent, verify_only=True)
    config_path = Path(args.model_path) / 'config.json'
    config = json.loads(config_path.read_text())
    expected = dict(num_hidden_layers=78, kv_lora_rank=512, qk_rope_head_dim=64)
    if any(config.get(k) != v for k, v in expected.items()):
        p.error(f'Expected full 78-layer GLM-5.3 W4AFP8 model: {config_path}')
    if 'GlmMoeDsaForCausalLM' not in config.get('architectures', []):
        p.error('Expected GlmMoeDsaForCausalLM, not GLM Flash / mHC')
    import torch
    if torch.cuda.device_count() != 8 or any(
        torch.cuda.get_device_capability(i) != (9, 0) for i in range(8)
    ):
        p.error('Expected eight visible Hopper GPUs (H200 target)')
    print('PP full-need policy=' + os.environ['SGLANG_PP_FULL_NEED'] + '; capacity and speed require the H200 workload test', flush=True)
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
