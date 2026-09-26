"""Load a versioned JSON profile and launch the installed source overlay."""
import argparse
import json
import os
from pathlib import Path
import sys


def argv_for(profile, *, model_path=None, kv_tokens=None, mem_fraction=None):
    argv=list(profile['argv'])
    for flag,value in (('--model-path',model_path),('--max-total-tokens',kv_tokens),
                       ('--mem-fraction-static',mem_fraction)):
        if value is not None:
            if flag in argv:argv[argv.index(flag)+1]=str(value)
            else:argv.extend([flag,str(value)])
    return argv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path)
    parser.add_argument('--model-path')
    parser.add_argument('--kv-tokens',type=int,help='Same effective KV token cap for every A/B variant')
    parser.add_argument('--mem-fraction',type=float,help='Start .78; lower by .02 after an OOM/headroom failure')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.kv_tokens is not None and args.kv_tokens<=0:parser.error('KV token cap must be positive')
    if args.mem_fraction is not None and not 0<args.mem_fraction<1:parser.error('Memory fraction must be in (0,1)')
    profile=json.loads(args.profile.read_text())
    argv=argv_for(profile,model_path=args.model_path,kv_tokens=args.kv_tokens,mem_fraction=args.mem_fraction)
    if args.dry_run:
        print(json.dumps({'profile':str(args.profile),'status':profile['status'],'argv':argv},indent=2))
        return
    os.environ.update(profile.get('env',{}))
    os.execv(sys.executable,[sys.executable,str(Path(__file__).with_name('launch.py')),*argv])


if __name__=='__main__':main()
