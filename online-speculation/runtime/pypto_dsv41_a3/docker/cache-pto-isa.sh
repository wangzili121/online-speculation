#!/usr/bin/env bash
set -euo pipefail

python=/opt/pypto/venv/bin/python
source_checkout="$PYPTO_SRC/runtime/build/pto-isa"
pin="$("$python" -c 'from simpler_setup.pto_isa import read_pto_isa_pin; print(read_pto_isa_pin())')"
runtime_checkout="$("$python" -c 'from simpler_setup.pto_isa import get_pto_isa_clone_path; print(get_pto_isa_clone_path())')"

check_checkout() {
    local head changes
    [[ -d "$1/.git" && -d "$1/include" ]] || return 1
    head="$(git -C "$1" rev-parse HEAD)" || return 1
    changes="$(git -C "$1" status --porcelain)" || return 1
    [[ "$head" == "$pin" && -z "$changes" ]]
}

if ! check_checkout "$runtime_checkout"; then
    check_checkout "$source_checkout" || {
        echo "Missing clean build checkout of PTO-ISA at $pin: $source_checkout" >&2
        exit 1
    }
    mkdir -p "$(dirname "$runtime_checkout")"
    staging="$(mktemp -d "$(dirname "$runtime_checkout")/.pto-isa.XXXXXX")"
    trap 'rm -rf -- "$staging"' EXIT
    cp -a "$source_checkout" "$staging/pto-isa"
    check_checkout "$staging/pto-isa"
    # The base image can carry a cache pinned by an older Simpler version.
    rm -rf -- "$runtime_checkout"
    mv "$staging/pto-isa" "$runtime_checkout"
fi
"$python" -c 'from simpler_setup.pto_isa import ensure_pto_isa_root; print(ensure_pto_isa_root())'
