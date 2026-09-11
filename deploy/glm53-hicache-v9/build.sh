#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-hicache-v9.2-0bcd822377da
sha256sum -c SHA256SUMS
docker build --progress=plain -t "$IMAGE" .
docker run --rm --entrypoint python3 "$IMAGE" /opt/glm53-cp8-dcp4-v1/install.py --verify-only
if [[ "${1:-}" == "--push" ]]; then
  docker push "$IMAGE"
fi
