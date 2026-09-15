"""Parse generated native arguments using the installed SGLang parser, without loading weights."""

import argparse

from render import PROFILES, render

from sglang.srt.server_args import ServerArgs

for profile in PROFILES:
    for weights in ("w4afp8", "fp8"):
        args = render(
            image="validation",
            profile=profile,
            weights=weights,
            model_pvc="validation-pvc",
        )[0]["spec"]["template"]["spec"]["containers"][0]["args"]
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parser.parse_args(args)
        print("Native parser accepted:", profile, weights, flush=True)
