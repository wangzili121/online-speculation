# Build on a networked ARM64 Linux Docker host, then docker save/load for offline use.
ARG BASE_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev@sha256:7712c0bae0f8a3c46cae127be4802324682e33ad1c49436e17db7bd60975dbee
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

ENV PYPTO_HOME=/opt/pypto \
    PYPTO_SRC=/opt/orthrus/pypto \
    PYPTO_LIB_ROOT=/opt/orthrus/pypto-lib \
    PYPTO_LIB_SRC=/opt/orthrus/pypto-lib \
    PYPTO_SERVING_ROOT=/opt/orthrus/pypto-serving \
    PYPTO_PLATFORM=a2a3 \
    CMAKE_BUILD_TYPE=Release \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy=
ARG BUILD_JOBS=4
WORKDIR /opt/orthrus
COPY online-speculation/runtime/pypto_dsv41_a3/lib-serving.patch /opt/orthrus/integration/lib-serving.patch
COPY online-speculation/runtime/pypto_dsv41_a3/lib-online-expert-capture.patch /opt/orthrus/integration/lib-online-expert-capture.patch
COPY online-speculation/runtime/pypto_dsv41_a3/serving-pair.patch /opt/orthrus/integration/serving-pair.patch
COPY online-speculation/runtime/pypto_dsv41_a3/runtime-release.patch /opt/orthrus/integration/runtime-release.patch

COPY online-speculation/runtime/pypto_dsv41_a3/docker/build-network.sh /opt/orthrus/integration/build-network.sh

# The runtime submodule is the revision checked by PyPTO at import time.
RUN --mount=type=secret,id=pypto_proxy --mount=type=secret,id=pypto_ca set -e; \
    source /opt/orthrus/integration/build-network.sh; \
    git init pypto; \
    git -C pypto fetch --depth 1 https://github.com/hw-native-sys/pypto.git 72a62d13efabbb25d240e35944f68ba5bcc22b7f; \
    git -C pypto checkout --detach FETCH_HEAD; \
    git -C pypto submodule update --init --recursive --depth 1; \
    test "$(git -C pypto/runtime rev-parse HEAD)" = 6b6b88f2353cca12cfafe809750b50b59931b0d1; \
    git -C pypto/runtime apply --check /opt/orthrus/integration/runtime-release.patch; \
    git -C pypto/runtime apply /opt/orthrus/integration/runtime-release.patch

RUN --mount=type=secret,id=pypto_proxy --mount=type=secret,id=pypto_ca set -e; \
    source /opt/orthrus/integration/build-network.sh; \
    git init pypto-lib; \
    git -C pypto-lib fetch --depth 1 https://github.com/Adelaide999/pypto-lib.git e32726b7edb31ee17e398a0431c0272ebd545974; \
    git -C pypto-lib checkout --detach FETCH_HEAD; \
    git -C pypto-lib -c user.name=Integration -c user.email=integration@localhost am \
        /opt/orthrus/integration/lib-serving.patch; \
    git -C pypto-lib apply --check /opt/orthrus/integration/lib-online-expert-capture.patch; \
    git -C pypto-lib apply /opt/orthrus/integration/lib-online-expert-capture.patch; \
    git init pypto-serving; \
    git -C pypto-serving fetch --depth 1 https://github.com/Adelaide999/pypto-serving.git cd818945c3c4e5454b8c259306d4e6f9e865e05c; \
    git -C pypto-serving checkout --detach FETCH_HEAD; \
    git -C pypto-serving -c user.name=Integration -c user.email=integration@localhost am \
        /opt/orthrus/integration/serving-pair.patch

RUN --mount=type=secret,id=pypto_proxy --mount=type=secret,id=pypto_ca --mount=type=secret,id=pypto_pypi set -e; \
    source /opt/orthrus/integration/build-network.sh; \
    source /opt/pypto/bin/pypto-activate.sh; \
    if [[ -s /run/secrets/pypto_pypi ]]; then \
        UV_DEFAULT_INDEX="$(cat /run/secrets/pypto_pypi)"; export UV_DEFAULT_INDEX; \
    fi; \
    uv pip install --python /opt/pypto/venv/bin/python \
        -c "$PYPTO_SRC/build-constraints.txt" cmake ninja nanobind scikit-build-core; \
    uv pip install --python /opt/pypto/venv/bin/python \
        torch==2.10.0 --default-index https://download.pytorch.org/whl/cpu; \
    bash "$PYPTO_SRC/docker/scripts/pypto-install-ptoas.sh"; \
    PYPTO_BUILD_JOBS="$BUILD_JOBS" bash "$PYPTO_SRC/docker/scripts/pypto-build" --clean; \
    uv pip install --python /opt/pypto/venv/bin/python --no-build-isolation \
        numpy==2.2.6 transformers==4.51.0 safetensors==0.8.0 tokenizers==0.21.4 \
        msgspec==0.22.0 fastapi==0.162.5 uvicorn==0.54.0 httpx==0.28.1 \
        prometheus-client==0.26.0 pydantic==2.14.3 starlette==1.7.0 \
        setuptools==80.9.0; \
    uv pip install --python /opt/pypto/venv/bin/python --no-deps --no-build-isolation \
        -e "$PYPTO_SERVING_ROOT"

COPY online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh /opt/orthrus/integration/cache-pto-isa.sh
RUN --network=none bash /opt/orthrus/integration/cache-pto-isa.sh

COPY online-speculation /opt/orthrus/online-speculation
RUN cp "$PYPTO_SRC/docker/scripts/pypto-activate.sh" /opt/pypto/bin/pypto-activate.sh && \
    source /opt/pypto/bin/pypto-activate.sh && \
    uv pip install --python /opt/pypto/venv/bin/python --no-deps --no-build-isolation \
    -e /opt/orthrus/online-speculation

COPY online-speculation/runtime/pypto_dsv41_a3/docker/entrypoint.sh /opt/orthrus/entrypoint.sh
RUN chmod +x /opt/orthrus/entrypoint.sh
ENV PYPTO_MOTD_SHOWN=1
ENTRYPOINT ["/opt/orthrus/entrypoint.sh"]
CMD ["doctor"]
RUN /opt/orthrus/entrypoint.sh doctor > /opt/orthrus/environment-build.json
