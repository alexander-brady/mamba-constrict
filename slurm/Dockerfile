FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace
COPY . /workspace/

# toolchain + numpy early (avoids your earlier numpy warning during builds)
RUN python -m pip install -U pip setuptools wheel numpy

# (optional safety) if kernels ever sneaks in via other deps, remove it
RUN pip uninstall -y kernels || true

# Build/install package + deps, but avoid isolated build env
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
RUN python -m pip install --no-build-isolation .

# Cleanup
RUN pip uninstall -y torchao || true

# sanity check (run-time GPU availability will still depend on podman run flags)
RUN python -c "import torch; print(torch.__version__, torch.version.cuda)"