#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?Specify cutlass or humming}
WORKLOAD=${2:?Specify short or long}
BASE_URL=${3:-http://sglang-glm53-dcp4.inf-glm53.svc:8080}
case "$ARM" in cutlass|humming) ;; *) exit 2 ;; esac
case "$WORKLOAD" in
  short)
    DATA=(--dataset-name random --random-input-len 1000 --random-output-len 1000 --num-prompts 200 --temperature 0)
    ;;
  long)
    DATA=(--dataset-name prefix_repetition --prefix-repetition-prefix-len 60000
      --prefix-repetition-suffix-len 15000 --prefix-repetition-output-len 1000
      --prefix-repetition-num-prefixes 20 --num-prompts 300 --temperature 0.3)
    ;;
  *) exit 2 ;;
esac
mkdir -p /results
vllm bench serve --backend openai-chat --base-url "$BASE_URL" \
  --endpoint /v1/chat/completions --model GLM-5.3 \
  --tokenizer PhalaCloud/GLM-5.3-W4AFP8 "${DATA[@]}" \
  --num-warmups 1 --max-concurrency 40 --request-rate inf --seed 0 --ignore-eos \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --save-detailed --result-dir /results \
  --result-filename "glm53-v99-${ARM}-${WORKLOAD}.json" \
  2>&1 | tee "/results/glm53-v99-${ARM}-${WORKLOAD}.log"
