#!/usr/bin/env bash
# Manual entrypoint for the later image-building phase. Nothing auto-pushes.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 IMAGE_TAG (run from a generated source bundle)" >&2
  exit 2
fi
sha256sum -c SHA256SUMS
BUILD_ARGS=()
if [[ -n "${GLM53_BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=$GLM53_BASE_IMAGE")
fi
docker build --platform linux/amd64 --progress=plain "${BUILD_ARGS[@]}" -t "$1" .
