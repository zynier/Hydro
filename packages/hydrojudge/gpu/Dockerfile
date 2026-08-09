ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG NSIGHT_COMPUTE_VERSION=2025.4.1

LABEL org.opencontainers.image.title="Hydro GPU operator judge"

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV NSIGHT_COMPUTE_ROOT=/opt/nvidia/nsight-compute/${NSIGHT_COMPUTE_VERSION}
ENV PATH=${NSIGHT_COMPUTE_ROOT}:${PATH}
WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends nsight-compute-${NSIGHT_COMPUTE_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Keep the Python GPU language toolchains reproducible. JIT compilation happens
# inside the isolated GPU container during the un-timed warmup phase.
ARG TILELANG_VERSION=0.1.13
ARG TIRX_VERSION=0.25.0.post1
ARG TRITON_VERSION=3.7.1
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check --break-system-packages \
    "tilelang==${TILELANG_VERSION}" \
    "apache-tvm==${TIRX_VERSION}" \
    "triton==${TRITON_VERSION}"

LABEL org.opencontainers.image.description="Isolated CUDA C++, TileLang 0.1.13, TIRx 0.25.0.post1, Triton 3.7.1, and PyTorch runtime for Hydro GPU operator submissions"
