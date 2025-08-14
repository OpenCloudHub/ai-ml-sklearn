# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
ARG PYTHON_VERSION=3.12
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
WORKDIR /workspace/project

# Environment variables for UV and Python
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src"

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

# Copy only necessary source code
COPY src/ ./src/

# Create a non-root user for running the app and change ownership of project files
RUN useradd -m -u 1001 rayuser && \
    chown -R rayuser:rayuser /workspace/project

# Switch to non-root user for runtime
USER rayuser

# Set environment for production
ENV ENVIRONMENT=production
