# ==============================================================================
# Training Script
# ==============================================================================
#
# Main training entrypoint for the Wine Quality Classifier.
#
# This script orchestrates the full training pipeline:
#   1. Loads data from DVC (versioned in S3/MinIO)
#   2. Trains a LogisticRegression model with StandardScaler pipeline
#   3. Uses Ray + joblib for distributed training
#   4. Logs all metrics, parameters, and artifacts to MLflow
#   5. Registers the model in MLflow Model Registry
#
# Usage:
#   python src/training/train.py --C 0.9 --max-iter 100
#
# Environment Variables Required:
#   - MLFLOW_TRACKING_URI: MLflow server URL
#   - ARGO_WORKFLOW_UID: Workflow identifier (use "DEV" locally)
#   - DOCKER_IMAGE_TAG: Image tag for traceability (use "DEV" locally)
#   - DVC_DATA_VERSION: Dataset version (e.g., wine-quality-v0.2.0)
#   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL: MinIO creds
#
# MLflow Integration:
#   - Uses autolog() for automatic metric capture
#   - Tags runs with workflow metadata for full traceability
#   - Logs DVC metadata as artifacts for data provenance
#
# ==============================================================================

import argparse
from datetime import datetime

import joblib
import mlflow
import ray
from ray.util.joblib import register_ray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG, WORKFLOW_TAGS
from src.training.data import load_data

logger = get_logger(__name__)


def train(X_train, y_train, X_val, y_val, C, max_iter, solver):
    """Train logistic regression model with Ray + joblib backend."""
    # Create pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver=solver,
                    random_state=TRAINING_CONFIG.random_state,
                ),
            ),
        ]
    )

    # 🚨 Need joblib to train with Ray backend
    with joblib.parallel_backend("ray"):
        pipeline.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_val, pipeline.predict(X_val))

    return pipeline, accuracy, X_val, y_val


def main():
    """Run training pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    log_section("Training Configuration", "⚙️")
    logger.info(f"C: {args.C}")
    logger.info(f"Max iterations: {args.max_iter}")
    logger.info(f"Solver: {args.solver}")

    log_section("CI/CD Data Contract from ENV", "📋")
    logger.info(f"Argo Workflow UID: {WORKFLOW_TAGS.argo_workflow_uid}")
    logger.info(f"Docker image tag: {WORKFLOW_TAGS.docker_image_tag}")
    logger.info(f"DVC data version: {WORKFLOW_TAGS.dvc_data_version}")

    # Setup Ray
    ray.init(address="auto", ignore_reinit_error=True)
    register_ray()

    # Load data
    logger.info(f"Loading data version: {WORKFLOW_TAGS.dvc_data_version}")
    X_train, y_train, X_val, y_val, metadata = load_data(WORKFLOW_TAGS.dvc_data_version)
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Setup MLflow
    mlflow.set_experiment(TRAINING_CONFIG.mlflow_experiment_name)
    run_name = args.run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ⚠️ IMPORTANT: Tag training rune with workflow tags
    workflow_tags = {
        "argo_workflow_uid": WORKFLOW_TAGS.argo_workflow_uid,
        "docker_image_tag": WORKFLOW_TAGS.docker_image_tag,
        "dvc_data_version": WORKFLOW_TAGS.dvc_data_version,
    }

    with mlflow.start_run(
        run_name=run_name, log_system_metrics=True, tags=workflow_tags
    ):
        log_section("Starting MLflow Run", "🚀")
        mlflow.sklearn.autolog(
            log_models=False, silent=TRAINING_CONFIG.environment == "production"
        )
        # Log DVC metadata as parameters for traceability
        mlflow.log_params(
            {
                "dvc_repo": TRAINING_CONFIG.dvc_repo,
                "dvc_data_path": TRAINING_CONFIG.dvc_data_path,
                "dvc_metrics_path": TRAINING_CONFIG.dvc_metrics_path,
                "dvc_data_version": WORKFLOW_TAGS.dvc_data_version,
            }
        )
        # Log data metrics from DVC as a JSON artifact
        mlflow.log_dict(metadata, "data_metadata.json")

        # Train
        logger.info(
            f"Training with C={args.C}, max_iter={args.max_iter}, solver={args.solver}"
        )
        pipeline, accuracy, X_val, y_val = train(
            X_train,
            y_train,
            X_val,
            y_val,
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
        )

        log_section("Training Results", "📊")
        logger.info(f"Train accuracy: {accuracy:.4f}")
        mlflow.log_metric("train_acc", accuracy)

        # Log model
        log_section("Logging Model to MLflow", "💾")
        registered_model_name = TRAINING_CONFIG.mlflow_registered_model_name
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=registered_model_name,
            input_example=X_val[:1],
        )
        logger.info(f"Model registered as: {registered_model_name}")

    logger.success("✅ Done!")


if __name__ == "__main__":
    main()
