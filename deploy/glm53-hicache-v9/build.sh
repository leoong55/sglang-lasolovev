#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-hicache-v9.14-0bcd822377da
if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--push" ) ]]; then
  echo "Usage: $0 [--push]" >&2
  exit 2
fi
sha256sum -c SHA256SUMS
BUILD_ARGS=()
if [[ -n "${GLM53_BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=$GLM53_BASE_IMAGE")
fi
docker build --progress=plain "${BUILD_ARGS[@]}" -t "$IMAGE" .
docker run --rm --entrypoint python3 "$IMAGE" /opt/glm53-cp8-dcp4-v1/install.py --verify-only
docker run --rm --entrypoint python3 -e GLM53_CANCEL_TEST_INSTALLED=1 "$IMAGE" \
  -m unittest discover -s /opt/glm53-cp8-dcp4-v1/tests -p 'test_host_cancel*.py' -v
if [[ "${1:-}" == "--push" ]]; then
  docker push "$IMAGE"
fi
