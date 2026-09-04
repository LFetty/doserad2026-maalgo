# Slimmer build of the DoseRAD2026 algorithm container (Grand Challenge invoke API).
#   docker build -f container/Dockerfile.slim -t doserad-proton-ct:slim .
#
# Two size problems in the single-stage Dockerfile:
#  1. `chmod -R`/`chown -R` over /opt/app/.venv created an 8.36 GB layer — overlayfs copies a
#     whole file when only its metadata changes, so that duplicated the entire venv.
#     Fixed by COPY --chown from the builder: ownership is set as the files land, once.
#  2. build-essential + git + the apt lists + the uv binary stayed in the final image even
#     though they are only needed to install dependencies. A builder stage leaves them behind.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/app/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

WORKDIR /opt/app
COPY pyproject.toml uv.lock README.md ./
# Dependencies only (no project install), then the GC server deps. Done in one layer so no
# intermediate copy of the venv is kept.
RUN uv sync --frozen --no-install-project \
    && uv pip install fastapi "uvicorn[standard]" \
    && find /opt/app/.venv /opt/uv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/uv /opt/app/.venv


# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/app/.venv/bin:$PATH \
    PYTHONPATH=/opt/app:/opt/app/src \
    MODEL_DIR=/opt/ml/model \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# torch.compile needs a C compiler AT RUNTIME: Triton/Inductor shells out to cc to build
# each kernel on the first forward. The builder stage's build-essential does not survive
# into this stage by design, so without this the compile path dies with
#   InductorError: RuntimeError: Failed to find C compiler
# and — because only the warmup catches that — the first metered /invoke returns HTTP 500
# with no dose written. Measured 2026-08-07: invoke 500 in 28.7 s, 0/10 slots.
# gcc + g++ + libc6-dev is ~150 MB against an 18.4 GB image; build-essential would also
# pull make/dpkg-dev, which Inductor does not use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create the user first so everything below can be owned correctly on arrival — no
# recursive chown/chmod afterwards (that is what produced the 8.36 GB layer).
RUN groupadd -r algorithm && useradd -r -g algorithm -u 1001 -m algorithm \
    && mkdir -p /opt/app /opt/ml/model \
    && chown -R algorithm:algorithm /opt/app /opt/ml

# Interpreter + dependencies, single copy, ownership set inline.
COPY --from=builder --chown=algorithm:algorithm /opt/uv /opt/uv
COPY --from=builder --chown=algorithm:algorithm /opt/app/.venv /opt/app/.venv

# Project code. Note we rely on PYTHONPATH=/opt/app rather than `uv sync` installing the
# project, so a code change only invalidates these small layers (and never re-runs a sync).
WORKDIR /opt/app
COPY --chown=algorithm:algorithm src/       ./src/
COPY --chown=algorithm:algorithm training/  ./training/
COPY --chown=algorithm:algorithm scripts/doserad_proton_utils.py ./scripts/doserad_proton_utils.py
COPY --chown=algorithm:algorithm container/ ./container/

USER algorithm

# Task selector: "proton_ct" (default) or "proton_mri".
ARG GC_TASK=proton_ct
ENV GC_TASK=${GC_TASK}

LABEL org.grand-challenge.api-method="invoke"

WORKDIR /opt/app/container
EXPOSE 4743
CMD ["python", "app.py"]
