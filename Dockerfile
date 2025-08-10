# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.5.19

#==============================================================================#
# Stage: UV binaries
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

#==============================================================================#
# Stage: Base image with UV
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Copy uv binaries from the uv stage
COPY --from=uv /uv /uvx /bin/

# Set working directory
WORKDIR /mlflow/projects/code

# Environment variables for UV and Python
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/mlflow/projects/code/.venv/bin:$PATH" \
    PYTHONPATH="/mlflow/projects/code"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#==============================================================================#
# Stage: Development environment
FROM base AS dev

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install ALL dependencies including dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --dev

# Set environment for development
ENV ENVIRONMENT=development

#==============================================================================#
# Stage: Production - Pure environment, code mounted at runtime
FROM base AS prod

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Create non-root user
RUN useradd -m -u 1000 mlflow

# Change ownership of dependency files only
RUN chown -R mlflow:mlflow pyproject.toml uv.lock

# Switch to non-root user for runtime
USER mlflow

# Set environment for production
ENV ENVIRONMENT=production
