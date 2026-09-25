#!/usr/bin/env bash
set -euo pipefail
# Run from the vLLM benchmark pod; service DNS resolves inside the cluster.
# Same 60k+15k -> 1k workload and 20 prefixes, with client concurrency 32.
vllm bench serve \
  --backend openai-chat \
  --base-url http://sglang-glm53-dcp4.inf-glm53.svc:8080 \
  --endpoint /v1/chat/completions \
  --model GLM-5.3 \
  --tokenizer PhalaCloud/GLM-5.3-W4AFP8 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 60000 \
  --prefix-repetition-suffix-len 15000 \
  --prefix-repetition-output-len 1000 \
  --prefix-repetition-num-prefixes 20 \
  --request-rate inf \
  --max-concurrency 32 \
  --num-prompts 300 \
  --num-warmups 1 \
  --ignore-eos \
  --temperature 0.3 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir /results \
  --result-filename glm53-q8-v2-c32-prefix20.json
