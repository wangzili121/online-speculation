# Sourced only by build steps that download dependencies.
if [[ -s /run/secrets/pypto_proxy ]]; then
    https_proxy="$(cat /run/secrets/pypto_proxy)"
    export https_proxy http_proxy="$https_proxy" HTTPS_PROXY="$https_proxy" HTTP_PROXY="$https_proxy"
fi

if [[ -s /run/secrets/pypto_ca ]]; then
    build_ca_bundle="$(mktemp)"
    trap 'rm -f -- "$build_ca_bundle"' EXIT
    cat /etc/ssl/certs/ca-certificates.crt > "$build_ca_bundle"
    printf '\n' >> "$build_ca_bundle"
    cat /run/secrets/pypto_ca >> "$build_ca_bundle"
    export SSL_CERT_FILE="$build_ca_bundle" GIT_SSL_CAINFO="$build_ca_bundle"
    export CURL_CA_BUNDLE="$build_ca_bundle" REQUESTS_CA_BUNDLE="$build_ca_bundle"
    export PIP_CERT="$build_ca_bundle" CMAKE_TLS_CAINFO="$build_ca_bundle"
fi
