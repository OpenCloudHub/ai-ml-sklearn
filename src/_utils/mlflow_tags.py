"""Utility functions for MLflow tagging and tracking."""

import os
from typing import Optional

import mlflow


def set_mlflow_tags(additional_tags: Optional[dict] = None) -> None:
    """
    Set standard lifecycle tags for MLflow runs.

    These tags enable traceability between Argo workflows, git commits,
    and MLflow experiments/models.

    Args:
        additional_tags: Optional dict of additional tags to set
    """
    # Get required environment variables
    argo_workflow_name = os.getenv("ARGO_WORKFLOW_NAME")
    argo_workflow_uid = os.getenv("ARGO_WORKFLOW_UID")
    git_repo = os.getenv("GIT_REPO")
    git_commit = os.getenv("GIT_COMMIT")

    # Build tags dict
    tags = {}

    # Add optional tags if they exist
    if argo_workflow_name:
        tags["argo_workflow_name"] = argo_workflow_name

    if argo_workflow_uid:
        tags["argo_workflow_uid"] = argo_workflow_uid

    if git_repo:
        tags["git_repo"] = git_repo

    if git_commit:
        tags["git_commit"] = git_commit

    # Add any additional project-specific tags
    if additional_tags:
        tags.update(additional_tags)

    # Set all tags
    mlflow.set_tags(tags)
