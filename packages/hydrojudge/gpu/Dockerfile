ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG NSIGHT_COMPUTE_VERSION=2025.4.1

LABEL org.opencontainers.image.title="Hydro GPU operator judge"
LABEL org.opencontainers.image.description="Isolated CUDA C++, TileLang 0.1.13, and PyTorch runtime for Hydro GPU operator submissions"

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV NSIGHT_COMPUTE_ROOT=/opt/nvidia/nsight-compute/${NSIGHT_COMPUTE_VERSION}
ENV PATH=${NSIGHT_COMPUTE_ROOT}:${PATH}
WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends nsight-compute-${NSIGHT_COMPUTE_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Keep the judge language version reproducible. TileLang JIT compilation happens
# inside the isolated GPU container during the un-timed warmup phase.
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check --break-system-packages tilelang==0.1.13
