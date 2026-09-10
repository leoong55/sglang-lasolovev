"""Small manual HTTP smoke; no external dependencies."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, default=Path("smoke-results.json"))
    args = parser.parse_args()
    url = args.url.rstrip("/")
    results = []
    def save():
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    # Numerous short words ensure multiple prefill chunks; verify actual usage.
    long_prompt = "Facts: the marker is TURBO42.\n" + "alpha beta gamma delta\n" * 2500 + "\nRepeat the marker:"
    for name, prompt in (("short", "The capital of France is"),
                         ("chunked_prefill", long_prompt),
                         ("repeat_prefix", long_prompt)):
        body = dict(model="GLM-5.3", prompt=prompt, max_tokens=32, temperature=0)
        request = urllib.request.Request(url + "/v1/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        row = dict(test=name)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                row["response"] = json.load(response)
            choices = row["response"].get("choices", [])
            if not choices or not choices[0].get("text", "").strip():
                raise RuntimeError("Empty completion")
            if name != "short" and row["response"].get("usage", {}).get("prompt_tokens", 0) <= 8192:
                raise RuntimeError("Prompt did not exceed the prefill chunk; increase repetitions")
            row["ok"] = True
        except Exception as error:
            row["ok"] = False
            row["error"] = str(error)
            if isinstance(error, urllib.error.HTTPError):
                row["http_body"] = error.read().decode(errors="replace")
            raise
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            results.append(row)
            save()
            print(f"{name}: {'OK' if row['ok'] else 'FAILED'}; {row['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
