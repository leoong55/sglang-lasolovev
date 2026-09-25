#!/usr/bin/env bash
set -euo pipefail

# Run in the same vLLM benchmark image used for the operator baseline.
# The service name and model intentionally target the full GLM-5.3 model.
BENCH_BASE_URL=${BENCH_BASE_URL:-http://sglang-glm53-dcp4.inf-glm53.svc:8080}
BENCH_RESULT_DIR=${BENCH_RESULT_DIR:-/results}
BENCH_RESULT_NAME=${BENCH_RESULT_NAME:-glm53-v9.7-c40-prefix20.json}
mkdir -p "$BENCH_RESULT_DIR"

vllm bench serve \
  --backend openai-chat \
  --base-url "$BENCH_BASE_URL" \
  --endpoint /v1/chat/completions \
  --model GLM-5.3 \
  --tokenizer PhalaCloud/GLM-5.3-W4AFP8 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 60000 \
  --prefix-repetition-suffix-len 15000 \
  --prefix-repetition-output-len 1000 \
  --prefix-repetition-num-prefixes 20 \
  --request-rate inf \
  --max-concurrency 40 \
  --num-prompts 300 \
  --num-warmups 1 \
  --seed 0 \
  --ignore-eos \
  --temperature 0.3 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir "$BENCH_RESULT_DIR" \
  --result-filename "$BENCH_RESULT_NAME"
