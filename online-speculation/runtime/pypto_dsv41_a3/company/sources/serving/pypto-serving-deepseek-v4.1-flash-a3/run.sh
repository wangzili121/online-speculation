#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$bundle_dir"
exec "${PYPTO_SERVING_PYTHON:-python}" -m serving.api "$@"
