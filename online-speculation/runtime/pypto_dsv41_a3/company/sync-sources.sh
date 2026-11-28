#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo 'Usage: sync-sources.sh /path/to/pypto-lib /path/to/DeepSeek-V4.1-Flash-A3' >&2
  exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib="$1"
serving="$2"
lib_commit="$(git -C "$lib" rev-parse HEAD)"
serving_commit="$(git -C "$serving" rev-parse HEAD)"
bundle=pypto-serving-deepseek-v4.1-flash-a3

stage="$(mktemp -d "$here/.sources.XXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
mkdir "$stage/pypto-lib" "$stage/serving"
git -C "$lib" archive "$lib_commit" golden models/deepseek_v4_1_flash_a3 \
  | tar -xf - -C "$stage/pypto-lib"
git -C "$serving" archive "$serving_commit" LICENSE "$bundle/models/__init__.py" \
  "$bundle/serving" "$bundle/requirements.txt" "$bundle/pypto.pin" "$bundle/run.sh" \
  | tar -xf - -C "$stage/serving"

printf 'LIB_COMMIT=%s\nSERVING_COMMIT=%s\n' "$lib_commit" "$serving_commit" > "$stage/commits.env"
rm -rf -- "$here/sources"
mv "$stage" "$here/sources"
printf 'Synchronized Lib %s and Serving %s\n' "$lib_commit" "$serving_commit"
