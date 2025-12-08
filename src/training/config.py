# ==============================================================================
# Training Configuration
# ==============================================================================
#
# Pydantic-based configuration management for training.
#
# Two configuration classes:
#   - WorkflowTags: CI/CD data contract (ARGO_WORKFLOW_UID, DOCKER_IMAGE_TAG,
#                   DVC_DATA_VERSION) - required for reproducible training
#   - TrainingConfig: Application settings (MLflow, DVC paths, random state)
#
# Configuration is loaded from environment variables automatically.
# For local development, create a .env file with required values.
#
# IMPORTANT - Lazy Loading:
#   WorkflowTags are lazy-loaded via get_workflow_tags() to prevent import
#   failures in serving code that imports training modules but doesn't need
#   the workflow environment variables.
#
# MLflow Tagging:
#   All WorkflowTags are applied to MLflow runs, enabling filtering by:
#   - Argo workflow run ID
#   - Docker image version
#   - Dataset version
#
# ==============================================================================

from pydantic_settings import BaseSettings


class WorkflowTags(BaseSettings):
    """⚠️ Data Contract for CI/CD Workflows:
    ============================================================
    When running in Argo Workflows, the following environment variables MUST be set
    and will take precedence over CLI arguments:

    - ARGO_WORKFLOW_UID: Unique identifier for the Argo workflow run
    - DOCKER_IMAGE_TAG: Docker image tag used for training (for reproducibility)
    - DVC_DATA_VERSION: Data version from DVC (takes precedence over --data-version arg)

    For local development, set these to "DEV" in your .env file.
    """

    # CI/CD Data Contract - Required fields set by Argo Workflows
    # These values are used for MLflow tagging and reproducibility
    argo_workflow_uid: str  # Must be set (use "DEV" for local development)
    docker_image_tag: str  # Must be set (use "DEV" for local development)
    dvc_data_version: str  # Takes precedence over --data-version CLI arg


class TrainingConfig(BaseSettings):
    """Configuration for the model training application."""

    environment: str = "development"  # "development" or "production"
    log_level: str = "INFO"

    # For experiment tracking
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "wine-quality"
    mlflow_registered_model_name: str = "dev.wine-classifier"

    # DVC repository URL
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    dvc_data_path: str = "data/wine-quality/processed/wine-quality.csv"
    dvc_metrics_path: str = "data/wine-quality/metadata.json"

    random_state: int = 42


# Singleton config instance - always available
TRAINING_CONFIG = TrainingConfig()


def get_workflow_tags() -> WorkflowTags:
    """Get workflow tags. Only call this during training, not at import time."""
    return WorkflowTags()
