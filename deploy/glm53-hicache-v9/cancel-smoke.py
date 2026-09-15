#!/usr/bin/env python3
"""One short streamed request, then close its HTTP connection. Prints the RID.
Exit 0 means client disconnect was exercised; server cancellation needs log verification.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://127.0.0.1:8080")
p.add_argument("--model", default="alpha-fm")
p.add_argument("--seconds", type=float, default=5)
a = p.parse_args()
if not 0 < a.seconds <= 15:
    p.error("--seconds must be >0 and <=15")
headers = {"Content-Type": "application/json"}
if os.environ.get("GLM53_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["GLM53_KEY"]
payload = {
    "model": a.model,
    "messages": [
        {
            "role": "user",
            "content": "Напиши длинный рассказ о путешествии по России, подробно описывая каждую остановку.",
        }
    ],
    "max_tokens": 4096,
    "ignore_eos": True,
    "stream": True,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    a.base_url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers=headers,
)
first_content = None
rid = None
start = time.monotonic()
try:
    # Connection/read timeout; this is a client-side smoke test, not a load test.
    with urllib.request.urlopen(req, timeout=15) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                raise SystemExit(
                    "Request finished naturally: cancellation was not exercised."
                )
            item = json.loads(raw)
            if item.get("error"):
                raise SystemExit("API returned an error; cancellation not verified.")
            if rid is None and item.get("id"):
                rid = item["id"]
                print("RID=" + rid, flush=True)
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if first_content is None:
                        first_content = time.monotonic()
            if (
                first_content is not None
                and time.monotonic() - first_content >= a.seconds
            ):
                break
            if time.monotonic() - start > 30:
                raise SystemExit(
                    "No usable stream within 30s; client closed, test inconclusive."
                )
        else:
            raise SystemExit("Unexpected EOF; cancellation test inconclusive.")
except urllib.error.HTTPError as e:
    raise SystemExit("HTTP " + str(e.code) + "; request was not tested.")
except (TimeoutError, OSError) as e:
    raise SystemExit(
        "Client I/O error: " + str(e) + "; inspect server logs, test inconclusive."
    )
print(
    "Connection closed intentionally after streaming. Check scheduler abort and terminal logs for RID above."
)
