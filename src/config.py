from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    # For experiment tracking
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "wine-quality"
    mlflow_registered_model_name: str = "dev.wine-classifier"

    # DVC repository URL
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    dvc_data_path: str = "data/wine-quality/processed/wine-quality.csv"
    dvc_metrics_path: str = "data/wine-quality/processed/metadata.json"

    # MLflow tags for workflow tagging
    argo_workflow_uid: str
    docker_image_tag: str

    random_state: int = 42


# Singleton config instance
BASE_CNFG = BaseConfig()
