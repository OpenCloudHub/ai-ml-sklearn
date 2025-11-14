import json
import os
from typing import Tuple

import dvc.api
import pandas as pd
import s3fs
from sklearn.model_selection import train_test_split

from src._utils.logging import get_logger, log_section
from src.config import BASE_CNFG

logger = get_logger(__name__)


def load_data(
    version: str, train_size: float = 0.8, shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Load and transform datasets from DVC.

    Args:
        version: DVC version tag (e.g., 'wine-quality-v0.1.1')
        train_size: Proportion of the dataset to include in the train split
        shuffle: Whether to shuffle the data before splitting

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, metadata)
    """
    log_section(f"Loading Data Version {version}", "📦")
    logger.info(f"DVC repo: [cyan]{BASE_CNFG.dvc_repo}[/cyan]")

    # Get URLs from DVC
    data_path = dvc.api.get_url(
        BASE_CNFG.dvc_data_path,
        repo=BASE_CNFG.dvc_repo,
        rev=version,
    )

    metadata_content = dvc.api.read(
        BASE_CNFG.dvc_metrics_path, repo=BASE_CNFG.dvc_repo, rev=version
    )
    metadata = json.loads(metadata_content)

    logger.info(
        f"Loaded dataset: [bold]{metadata['dataset']['name']}[/bold] [green]({version})[/green]"
    )

    # Configure S3 filesystem with SSL verification disabled
    s3_client = s3fs.S3FileSystem(
        anon=False,
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        client_kwargs={
            "verify": False,  # Disable SSL verification for self-signed certs
        },
    )

    # Open and read CSV using s3fs
    with s3_client.open(data_path, "rb") as f:
        df = pd.read_csv(f)

    # Drop index column if present
    if "__index_level_0__" in df.columns:
        df.drop(columns=["__index_level_0__"], inplace=True)

    # Prepare data
    feature_cols = [col for col in df.columns if col != "quality"]
    X, y = df[feature_cols], df["quality"]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=train_size,
        shuffle=shuffle,
        random_state=BASE_CNFG.random_state,
    )

    logger.success("✨ Data loaded and split")

    return X_train, y_train, X_val, y_val, metadata
