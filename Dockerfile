FROM python:3.12-slim-bookworm

# libgl1/libglib2.0-0: needed by opencv-python at import time.
# build-essential: needed to build pycocotools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so they're cached even when the source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY sam3 ./sam3
COPY assets ./assets
COPY demo ./demo
COPY README.md pyproject.toml uv.lock ./

RUN uv sync --locked

ENV PATH="/app/.venv/bin:${PATH}"
# Hugging Face cache is expected to be mounted here with the sam3 weights
# already in place (see README.md "Model weights").
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

CMD ["uv", "run", "demo/server.py"]
