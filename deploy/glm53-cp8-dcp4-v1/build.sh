#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${TARGET_IMAGE:=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-v1-0bcd822377da}"
test -f base-files.json || { echo 'Run make_bundle.py first, or use the release archive.' >&2; exit 1; }
build_args=()
if [[ -n "${BASE_IMAGE:-}" ]]; then build_args+=(--build-arg "BASE_IMAGE=$BASE_IMAGE"); fi
docker build "${build_args[@]}" -t "$TARGET_IMAGE" .
docker run --rm --entrypoint python3 "$TARGET_IMAGE" /opt/glm53-cp8-dcp4-v1/install.py --verify-only
docker image inspect "$TARGET_IMAGE" > image-inspect.json
docker save "$TARGET_IMAGE" | gzip -1 > sglang-glm53-cp8-dcp4-v1-image.tar.gz
sha256sum sglang-glm53-cp8-dcp4-v1-image.tar.gz > sglang-glm53-cp8-dcp4-v1-image.tar.gz.sha256
echo "Built and exported $TARGET_IMAGE. Publish separately with: docker push $TARGET_IMAGE"
