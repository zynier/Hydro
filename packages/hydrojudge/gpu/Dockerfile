ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
FROM ${PYTORCH_IMAGE}

LABEL org.opencontainers.image.title="Hydro GPU operator judge"
LABEL org.opencontainers.image.description="Isolated CUDA C++ and PyTorch baseline runtime for Hydro GPU operator submissions"

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
