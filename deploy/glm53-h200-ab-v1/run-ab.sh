#!/usr/bin/env bash
# Run on an operator host with kubectl, Python/aiohttp/PyYAML and access to the lab Service.
set -euo pipefail
KIT=$(cd -- "$(dirname -- "$0")" && pwd)
: "${IMAGE:?Set an immutable image@sha256:digest}"
: "${DATASET:?Set the prepared token-ID dataset path}"
: "${RESULTS:?Set a NEW result directory}"
[[ $IMAGE == *@sha256:* ]] || { echo 'IMAGE must include digest' >&2; exit 2; }
[[ ! -e $RESULTS ]] || { echo 'RESULTS already exists' >&2; exit 2; }
C=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["requests"]))' "$DATASET")
NS=inf-glm53
NAME=sglang-glm53-h200-ab
URL=${URL:-http://sglang-glm53-h200-ab.inf-glm53.svc:8080}
render_opts=(--patches --weights "${WEIGHTS:-w4afp8}" --context "${CONTEXT:-500000}" --mem-fraction "${MEM_FRACTION:-0.90}")
[[ -z ${MODEL_PVC:-} ]] || render_opts+=(--model-pvc "$MODEL_PVC")
mkdir -p "$RESULTS"
for arm in A1 B1 B2 A2; do
  profile=tp8-decode
  [[ $arm == B* ]] && profile=${PP_PROFILE:-pp4-decode}
  yaml_path="$RESULTS/$arm.yaml"
  python3 "$KIT/render.py" --image "$IMAGE" --profile "$profile" --concurrency "$C" "${render_opts[@]}" > "$yaml_path"
  # Explicit restart also between B1/B2: radix/runtime state must not leak between runs.
  if kubectl -n "$NS" get deployment "$NAME" >/dev/null 2>&1; then
    kubectl -n "$NS" scale deployment "$NAME" --replicas=0
    kubectl -n "$NS" wait --for=delete pod -l "app=$NAME" --timeout=600s
  fi
  kubectl apply -f "$yaml_path"
  kubectl -n "$NS" rollout status deployment "$NAME" --timeout=7200s
  kubectl -n "$NS" get pods -l "app=$NAME" -o json > "$RESULTS/$arm.pod-before.json"
  # Warm compiler/kernels using the same wave; discard its timings. An idle cache flush
  # precedes each wave. Measurement uses a fresh directory and the unchanged dataset.
  python3 "$KIT/bench_decode.py" run --url "$URL" --dataset "$DATASET" --manifest "$yaml_path" \
    --pod-json "$RESULTS/$arm.pod-before.json" --label "$arm-warmup" --output "$RESULTS/$arm-warmup"
  python3 "$KIT/bench_decode.py" run --url "$URL" --dataset "$DATASET" --manifest "$yaml_path" \
    --pod-json "$RESULTS/$arm.pod-before.json" --label "$arm" --output "$RESULTS/$arm"
  kubectl -n "$NS" get pods -l "app=$NAME" -o json > "$RESULTS/$arm.pod-after.json"
  kubectl -n "$NS" logs -l "app=$NAME" --all-containers=true --prefix=true --tail=-1 > "$RESULTS/$arm.server.log"
  python3 - "$RESULTS/$arm.pod-before.json" "$RESULTS/$arm.pod-after.json" <<'PY'
import json,sys
before,after=[json.load(open(p))['items'] for p in sys.argv[1:]]
assert len(before)==len(after)==1, 'Expected exactly one lab pod'
a,b=before[0],after[0]
assert a['metadata']['uid']==b['metadata']['uid'], 'Pod replaced during benchmark'
assert a['status']['containerStatuses']==b['status']['containerStatuses'], 'Container status changed; inspect restarts'
PY
done
python3 "$KIT/compare.py" "$RESULTS/A1" "$RESULTS/B1" --mode topology
python3 "$KIT/compare.py" "$RESULTS/A2" "$RESULTS/B2" --mode topology
