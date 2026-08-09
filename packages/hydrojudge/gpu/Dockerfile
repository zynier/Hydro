ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
FROM ${PYTORCH_IMAGE}

LABEL org.opencontainers.image.title="Hydro GPU operator judge"
LABEL org.opencontainers.image.description="Isolated CUDA C++, TileLang 0.1.13, and PyTorch runtime for Hydro GPU operator submissions"

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /work

# Keep the judge language version reproducible. TileLang JIT compilation happens
# inside the isolated GPU container during the un-timed warmup phase.
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check --break-system-packages tilelang==0.1.13
