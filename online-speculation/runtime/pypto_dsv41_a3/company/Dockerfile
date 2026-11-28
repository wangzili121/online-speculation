ARG BASE_IMAGE
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
WORKDIR /opt/orthrus-a3
ENV PYPTO_HOME=/opt/pypto \
    PYPTO_SRC=/opt/orthrus-a3/pypto \
    PYPTO_LIB_SRC=/opt/orthrus-a3/pypto-lib \
    PYPTO_LIB_ROOT=/opt/orthrus-a3/pypto-lib \
    PYPTO_PLATFORM=a2a3 \
    PYPTO_MOTD_SHOWN=1 \
    CMAKE_BUILD_TYPE=Release \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    BASH_ENV=/dev/null \
    PYTHONPATH=/opt/orthrus-a3/pypto/python:/opt/orthrus-a3/pypto-lib

COPY pypto ./pypto
COPY runtime-release.patch ./
RUN git -C pypto/runtime apply /opt/orthrus-a3/runtime-release.patch
RUN set -e; source /opt/pypto/bin/pypto-activate.sh; \
    uv pip install --python /opt/pypto/venv/bin/python \
      -c "$PYPTO_SRC/build-constraints.txt" cmake ninja nanobind scikit-build-core; \
    bash "$PYPTO_SRC/docker/scripts/pypto-install-ptoas.sh"; \
    PYPTO_RUNTIME_BUILD_TARGETS=build_package_a2a3 PYPTO_BUILD_JOBS=8 \
      bash "$PYPTO_SRC/docker/scripts/pypto-build"; \
    cp "$PYPTO_SRC/docker/scripts/pypto-activate.sh" /opt/pypto/bin/pypto-activate.sh; \
    source /opt/pypto/bin/pypto-activate.sh; \
    python -c 'from simpler_setup.pto_isa import ensure_pto_isa_root; print(ensure_pto_isa_root())'

COPY sources/pypto-lib ./pypto-lib
COPY sources/serving ./DeepSeek-V4.1-Flash-A3
COPY sources/commits.env ./commits.env
COPY study ./study
COPY start-serving.sh /opt/orthrus/start-serving.sh
RUN set -e; source /opt/pypto/bin/pypto-activate.sh; \
    uv pip install --python /opt/pypto/venv/bin/python \
      -r DeepSeek-V4.1-Flash-A3/pypto-serving-deepseek-v4.1-flash-a3/requirements.txt; \
    bash -n /opt/orthrus/start-serving.sh
ENTRYPOINT []
CMD ["bash", "/opt/orthrus/start-serving.sh"]
