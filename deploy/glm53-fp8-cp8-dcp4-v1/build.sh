#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
[[ $# -eq 0 || ( $# -eq 1 && $1 == --push ) ]] || { echo 'Usage: build.sh [--push]' >&2; exit 2; }
commit=$(python3 -c 'import json; print(json.load(open("REVISION.json"))["source_commit"])')
IMAGE=${IMAGE:-i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-fp8-cp8-dcp4-v1-${commit:0:12}}
sha256sum -c SHA256SUMS
args=(--build-arg "SOURCE_COMMIT=$commit")
if [[ -n ${GLM53_BASE_IMAGE:-} ]]; then args+=(--build-arg "BASE_IMAGE=$GLM53_BASE_IMAGE"); fi
docker build --progress=plain "${args[@]}" -t "$IMAGE" .
docker run --rm --entrypoint python3 "$IMAGE" /opt/glm53-fp8-cp8-dcp4/install.py --verify-only
if [[ ${1:-} == --push ]]; then docker push "$IMAGE"; fi
