# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG RAY_VERSION=2.48.0
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm

# Compose tags
ARG RAY_PY_TAG=py${PYTHON_MAJOR}${PYTHON_MINOR}
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}

#==============================================================================#
# Stage: UV binaries
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv

WORKDIR /workspace/project

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#==============================================================================#
# Stage: Development environment
FROM uv AS dev

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --dev

ENV ENVIRONMENT=development

#==============================================================================#
# Stage: Training - Ray base with training code
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS training

WORKDIR /workspace/project

COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project/src" \
    ENVIRONMENT=production

#==============================================================================#
# Stage: Serving - Includes baked-in model weights
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS serving

ARG MLFLOW_TRACKING_URI
ARG MODEL_NAME
ARG MODEL_VERSION

WORKDIR /workspace/project

COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copy only serving code (not training)
COPY src/serving/ ./src/serving/
COPY src/_utils/ ./src/_utils/

# Set environment variables
ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project/src" \
    MODEL_PATH="/workspace/project/model" \
    MODEL_NAME="${MODEL_NAME}" \
    ENVIRONMENT=production

# Download staging model from MLflow at build time
RUN python -c "import mlflow; mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}'); mlflow.artifacts.download_artifacts(artifact_uri='models:/staging.${MODEL_NAME}/${MODEL_VERSION}', dst_path='/workspace/project/model'); print('✅ Model ${MODEL_NAME} v${MODEL_VERSION} downloaded')"
