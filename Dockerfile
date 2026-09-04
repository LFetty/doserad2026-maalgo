# Training image: the CUDA + PyTorch environment the correction network was trained in.
# Build and push to your own registry, e.g.
#   docker build -t <registry>/pydose_rt:latest . && docker push <registry>/pydose_rt:latest
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/pydose-rt/.venv \
    PATH=/opt/pydose-rt/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv (pinned via release URL — no network at run time)
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

WORKDIR /opt/pydose-rt

# Install dependencies first for layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Copy the project last so code changes don't bust the dep layer.
COPY src/   ./src/
COPY training/ ./training/
COPY scripts/  ./scripts/
COPY example_data/ ./example_data/
COPY tests/    ./tests/

RUN uv sync --frozen

CMD ["bash"]
