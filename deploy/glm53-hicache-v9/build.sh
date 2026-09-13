#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
build_cutlass=1
push_image=0
image_suffix=
for option in "$@"; do
  case "$option" in
    --push) push_image=1 ;;
    --humming-only) build_cutlass=0; image_suffix=-humming-only ;;
    *) echo "Usage: bash build.sh [--humming-only] [--push]" >&2; exit 2 ;;
  esac
done
IMAGE="i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-hicache-v9.11.1${image_suffix}-0bcd822377da"
sha256sum -c SHA256SUMS
echo "Building $IMAGE; optional CUTLASS extension=$build_cutlass"
docker build --progress=plain --build-arg "GLM53_BUILD_CUTLASS=$build_cutlass" -t "$IMAGE" .
docker run --rm --entrypoint python3 "$IMAGE" /opt/glm53-cp8-dcp4-v1/install.py --verify-only
docker run --rm --entrypoint python3 "$IMAGE" /opt/glm53-cp8-dcp4-v1/cutlass/check_build.py --expected "$build_cutlass"
if [[ "$push_image" == "1" ]]; then
  docker push "$IMAGE"
fi
