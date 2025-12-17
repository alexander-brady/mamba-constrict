FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace
COPY . /workspace/

ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE

# IMPORTANT: build extensions against the container's CUDA torch, not pip's CPU torch
RUN pip install --no-build-isolation .

RUN pip uninstall -y torchao