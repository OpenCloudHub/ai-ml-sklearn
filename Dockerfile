# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG RAY_VERSION=2.48.0
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
ARG RAY_PY_TAG=py${PYTHON_MAJOR}${PYTHON_MINOR}
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}

#==============================================================================#
# Stage: Base with UV (stays as root, just tools)
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv_base
WORKDIR /workspace/project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

#==============================================================================#
# Stage: Development (for devcontainer - no venv in image)
FROM uv_base AS dev
COPY pyproject.toml uv.lock ./
# Don't create venv here - let devcontainer do it at runtime
ENV ENVIRONMENT=development

#==============================================================================#
# Stage: TRAINING (slim base + ray[train])
FROM python:3.12-slim-bookworm AS training
WORKDIR /workspace/project

# Install system deps for building packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra training --no-dev --no-install-project

COPY src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=training

#==============================================================================#
# Stage: SERVING (slim base + ray[serve])
FROM python:3.12-slim-bookworm AS serving
WORKDIR /workspace/project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra serving --no-dev --no-install-project

COPY src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=production