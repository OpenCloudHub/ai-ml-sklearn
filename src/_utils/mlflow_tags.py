"""Utility functions for MLflow tagging and tracking."""

import os
from typing import Optional

import mlflow


def _get_lifecycle_tags() -> dict:
    """
    Get lifecycle tags from environment variables.

    Returns:
        Dict of tag key-value pairs (only includes tags that are set)
    """
    env_vars = {
        "argo_workflow_name": "ARGO_WORKFLOW_NAME",
        "argo_workflow_uid": "ARGO_WORKFLOW_UID",
        "git_repo": "GIT_REPO",
        "git_commit": "GIT_COMMIT",
    }

    tags = {}
    for tag_key, env_var in env_vars.items():
        value = os.getenv(env_var)
        if value:
            tags[tag_key] = value

    return tags


def set_mlflow_experiment_tags(additional_tags: Optional[dict] = None) -> None:
    """
    Set standard lifecycle tags for MLflow runs.

    These tags enable traceability between Argo workflows, git commits,
    and MLflow experiments/models.

    Args:
        additional_tags: Optional dict of additional tags to set
    """
    tags = _get_lifecycle_tags()

    if additional_tags:
        tags.update(additional_tags)

    mlflow.set_tags(tags)
