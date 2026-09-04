# Lean DoseRAD2026 proton CT/MRI invoke image.
#
# Unlike the original multi-stage image, this starts with the exact integrated
# PyTorch/CUDA runtime used by inference. This avoids installing a second 4+ GB
# set of NVIDIA Python wheels and excludes training/evaluation-only packages.
#
# Build from the repository root:
#   docker build -f container/Dockerfile.lean \
#       -t doserad-proton-ct:lean-runtime-20260830 .
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app:/opt/app/src \
    MODEL_DIR=/opt/ml/model \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# Inductor invokes gcc during the unmetered torch.compile warm-up. libc6-dev is
# required for the generated extension headers; g++ is retained for compiler
# probes and any C++ source emitted by future Torch versions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
COPY container/requirements.runtime.txt /tmp/requirements.runtime.txt
# The PyTorch Ubuntu 24.04 image marks its system Python as externally managed
# (PEP 668). This image is itself the isolated environment, so installing the
# pinned serving wheels into it is intentional.
RUN python -m pip install --break-system-packages --no-cache-dir \
        -r /tmp/requirements.runtime.txt \
    && rm /tmp/requirements.runtime.txt

RUN groupadd -r algorithm \
    && useradd -r -g algorithm -u 1001 -m algorithm \
    && mkdir -p /opt/app /opt/ml/model \
    && chown -R algorithm:algorithm /opt/app /opt/ml

# The project is imported through PYTHONPATH, so it does not need to be
# installed and cannot pull the broad pyproject.toml dependency set.
COPY --chown=algorithm:algorithm src/       ./src/
COPY --chown=algorithm:algorithm training/  ./training/
COPY --chown=algorithm:algorithm scripts/doserad_proton_utils.py ./scripts/doserad_proton_utils.py
COPY --chown=algorithm:algorithm container/ ./container/

USER algorithm

ARG GC_TASK=proton_ct
ENV GC_TASK=${GC_TASK}

LABEL org.grand-challenge.api-method="invoke"

WORKDIR /opt/app/container
EXPOSE 4743
CMD ["python", "app.py"]
