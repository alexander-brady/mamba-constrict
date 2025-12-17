FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace
COPY . /workspace/

# (optional but helps) avoid the numpy warning you saw
RUN python -m pip install -U pip numpy

# IMPORTANT: build extensions against the container's CUDA torch, not pip's CPU torch
RUN pip install --no-build-isolation .

RUN pip uninstall -y torchao