"""Reproduce vLLM 0.23.0's long dataset offline and emit metadata only.

Run in the same pinned benchmark image with the same read-only model directory:
  python /scripts/dataset_catalog.py --model-path /model --output /results/dataset-catalog.json

No HTTP inference, serving configuration, or runtime source is changed. Prefix
fingerprints come from the actual adjusted prefix token array before suffix
concatenation, so tokenizer boundary merging cannot misclassify prefix groups.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

CONFIG = {
    "seed": 0,
    "prefix_len": 60000,
    "suffix_len": 15000,
    "output_len": 1000,
    "num_prefixes": 20,
    "num_requests": 300,
}
SOURCE_HASHES = {
    "PrefixRepetitionRandomDataset": "0ca4e3ae532f5b16298e0b645b1d15d51056da3d01d94d366eb07a1d1f33fb59",
    "gen_prompt_decode_to_target_len": "de994cd54cbc1b9fa203f59ba137d51ab94b9059237c1027e42bafc437d17b59",
}
STAGE = "argument_validation"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
)
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def token_fingerprint(tokens):
    values = array.array("I", tokens)
    if values.itemsize != 4:
        raise RuntimeError("Unexpected unsigned-int size")
    if sys.byteorder != "little":
        values.byteswap()
    return sha256(values.tobytes())


def file_fingerprint(directory, names):
    files = {}
    for name in names:
        path = directory / name
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            files[name] = hasher.hexdigest()
    if not files:
        raise RuntimeError("Required model/tokenizer metadata files were not found")
    combined = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": sha256(combined), "files": files}


def verify_dataset_source(module):
    observed = {
        name: sha256(inspect.getsource(getattr(module, name)).encode())
        for name in SOURCE_HASHES
    }
    if observed != SOURCE_HASHES:
        raise RuntimeError(
            "Installed dataset source differs from the verified vLLM v0.23.0 source"
        )
    return observed


def generate_catalog(module, tokenizer, configuration=None):
    """Call the actual generator; observe token-prefix creation and object IDs.

    The wrappers do not consume RNG, transform prompts, or alter sample order.
    Original module functions are always restored. The caller runs this in a
    dedicated, single-threaded offline process.
    """
    config = dict(CONFIG if configuration is None else configuration)
    if config["prefix_len"] == config["suffix_len"]:
        raise ValueError(
            "Prefix and suffix lengths must differ for unambiguous observation"
        )
    original_adjust = module.gen_prompt_decode_to_target_len
    original_sample = module.SampleRequest
    current_prefix = None
    prefixes = []
    sample_metadata = {}

    def adjust(*args, **kwargs):
        nonlocal current_prefix
        value = original_adjust(*args, **kwargs)
        bound = inspect.signature(original_adjust).bind(*args, **kwargs)
        target = bound.arguments["target_token_len"]
        if target == config["prefix_len"]:
            current_prefix = token_fingerprint(value[1])
            prefixes.append(
                {
                    "prefix_fingerprint": current_prefix,
                    "adjusted_prefix_tokens": len(value[1]),
                    "token_mismatch": value[2],
                }
            )
        elif target != config["suffix_len"]:
            raise RuntimeError(
                "Unexpected generation helper call; no trustworthy prefix catalog"
            )
        return value

    def sample(*args, **kwargs):
        value = original_sample(*args, **kwargs)
        if current_prefix is None or not isinstance(value.prompt, str):
            raise RuntimeError("Unexpected dataset sample shape")
        sample_metadata[id(value)] = {
            "prefix_fingerprint": current_prefix,
            "full_prompt_sha256": sha256(value.prompt.encode("utf-8")),
            "prompt_tokens_before_chat_template": value.prompt_len,
            "output_tokens": value.expected_output_len,
        }
        return value

    module.gen_prompt_decode_to_target_len = adjust
    module.SampleRequest = sample
    try:
        random.seed(config["seed"])
        module.np.random.seed(config["seed"])
        dataset = module.PrefixRepetitionRandomDataset(
            random_seed=config["seed"], dataset_path=None, disable_shuffle=False
        )
        samples = dataset.sample(
            tokenizer=tokenizer,
            num_requests=config["num_requests"],
            prefix_len=config["prefix_len"],
            suffix_len=config["suffix_len"],
            num_prefixes=config["num_prefixes"],
            output_len=config["output_len"],
            request_id_prefix="",
            no_oversample=False,
        )
    finally:
        module.gen_prompt_decode_to_target_len = original_adjust
        module.SampleRequest = original_sample
    entries = [
        {"sample_index": index, **sample_metadata[id(value)]}
        for index, value in enumerate(samples)
    ]
    counts = Counter(entry["prefix_fingerprint"] for entry in entries)
    expected_per_prefix = config["num_requests"] // config["num_prefixes"]
    if (
        len(entries) != config["num_requests"]
        or len(prefixes) != config["num_prefixes"]
        or len(counts) != config["num_prefixes"]
        or any(value != expected_per_prefix for value in counts.values())
    ):
        raise RuntimeError(
            "Generated prefix groups do not match the accepted dataset contract"
        )
    return entries, prefixes, samples


class _PayloadCaptured(BaseException):
    pass


def request_body_fingerprints(samples):
    """Obtain exact body hashes from vLLM's real HTTP function, with no network.

    A fake session captures the Python JSON payload before any transport exists.
    json.dumps uses aiohttp ClientSession's default serialization. This lets the
    local analyzer verify sample-index joins against requests.jsonl body hashes.
    """
    import asyncio

    from vllm.benchmarks.lib.endpoint_request_func import (
        RequestFuncInput,
        async_request_openai_chat_completions,
    )

    hashes = []

    class CaptureSession:
        def post(self, *, url, json, headers):
            # Never retain headers: the vLLM helper can read OPENAI_API_KEY.
            hashes.append(sha256(__import__("json").dumps(json).encode("utf-8")))
            raise _PayloadCaptured()

    async def capture():
        session = CaptureSession()
        for sample in samples:
            request = RequestFuncInput(
                prompt=sample.prompt,
                api_url="http://offline.invalid/v1/chat/completions",
                prompt_len=sample.prompt_len,
                output_len=sample.expected_output_len,
                model="GLM-5.3",
                model_name=None,
                ignore_eos=True,
                extra_body={
                    "temperature": 0.3,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            try:
                await async_request_openai_chat_completions(request, session)
            except _PayloadCaptured:
                pass
            else:
                raise RuntimeError(
                    "vLLM request construction bypassed the offline capture"
                )

    asyncio.run(capture())
    if len(hashes) != len(samples):
        raise RuntimeError("Incomplete request body catalog")
    return hashes


def build(model_path):
    global STAGE
    STAGE = "package_version"
    version = importlib.metadata.version("vllm")
    if version.split("+")[0] != "0.23.0":
        raise RuntimeError("The catalog requires vLLM 0.23.0")
    from vllm.benchmarks.datasets import datasets as dataset_module
    from vllm.tokenizers import get_tokenizer

    STAGE = "dataset_source_verification"
    source_hashes = verify_dataset_source(dataset_module)
    STAGE = "metadata_file_hashing"
    model_files = file_fingerprint(model_path, MODEL_METADATA_FILES)
    tokenizer_files = file_fingerprint(model_path, TOKENIZER_FILES)
    STAGE = "local_tokenizer_loading"
    tokenizer = get_tokenizer(
        str(model_path), tokenizer_mode="auto", trust_remote_code=False
    )
    STAGE = "dataset_generation"
    entries, prefixes, samples = generate_catalog(dataset_module, tokenizer)
    STAGE = "offline_request_payload_capture"
    body_hashes = request_body_fingerprints(samples)
    for entry, body_hash in zip(entries, body_hashes):
        entry["request_body_sha256"] = body_hash
    return {
        "schema_version": 1,
        "dataset": "prefix_repetition",
        "configuration": CONFIG,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("vllm", "numpy", "transformers", "aiohttp")
        },
        "tokenizer_mode": "auto",
        "trust_remote_code": False,
        "dataset_source_sha256": source_hashes,
        "model_metadata": model_files,
        "tokenizer_metadata": tokenizer_files,
        "weights_content_hashed": False,
        "model_hash_scope": "config, generation config and weight index metadata; not weight tensor contents",
        "prefix_fingerprint_encoding": "SHA256 of actual adjusted prefix token IDs as unsigned 32-bit little-endian bytes",
        "full_prompt_sha256_encoding": "SHA256 of sampled prompt UTF-8 before chat-template application",
        "request_body_sha256_encoding": "SHA256 of json.dumps(payload) UTF-8 from the pinned vLLM HTTP function, captured before transport",
        "sample_index_contract": "Final shuffled order; equals numeric suffix of measured vLLM request ID. Verify request_body_sha256 before joining.",
        "prefixes": prefixes,
        "samples": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Also emit metadata catalog to stdout for Kubernetes log collection",
    )
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    try:
        if (
            not args.model_path.is_dir()
            or not (args.model_path / "config.json").is_file()
        ):
            raise RuntimeError("Expected the model directory containing config.json")
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite an existing catalog")
        # Third-party debug logs may contain local paths or tokenizer details.
        # Keep them out of the metadata-only Job output even on exceptions.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            catalog = build(args.model_path)
        serialized = json.dumps(
            catalog, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            os.chmod(args.output, 0o600)
            stream.write(serialized + "\n")
        if args.emit_json:
            print("GLM53_DATASET_CATALOG " + serialized, flush=True)
        print(
            "GLM53_DATASET_CATALOG_SUMMARY "
            + json.dumps(
                {
                    "samples": len(catalog["samples"]),
                    "prefixes": len(catalog["prefixes"]),
                    "catalog_sha256": sha256(serialized.encode()),
                    "http_requests": 0,
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        # Do not echo arbitrary third-party exception messages: they may include
        # tokenized prompts. The failure type and stage suffice to reject a catalog.
        print(
            "GLM53_DATASET_CATALOG_FAILED "
            + json.dumps({"error_type": type(exc).__name__, "stage": STAGE}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
