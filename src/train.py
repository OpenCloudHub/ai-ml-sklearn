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

from src._utils.logging import get_logger
from src.config import BASE_CNFG
from src.data import load_data

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
                    random_state=BASE_CNFG.random_state,
                    multi_class="auto",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-version",
        required=True,
        help="Version of the data to use for training(e.g. wine-quality-v0.0.2)",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    # Setup Ray
    ray.init(address="auto", ignore_reinit_error=True)
    register_ray()

    # Load data
    logger.info(f"Loading data version: {args.data_version}")
    X_train, y_train, X_val, y_val, metadata = load_data(args.data_version)
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Setup MLflow
    mlflow.set_experiment(BASE_CNFG.mlflow_experiment_name)
    run_name = args.run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create workflow tags
    workflow_tags = {
        "argo_workflow_uid": BASE_CNFG.argo_workflow_uid,
        "docker_image_tag": BASE_CNFG.docker_image_tag,
        "dvc_data_version": args.data_version,
    }

    with mlflow.start_run(
        run_name=run_name, log_system_metrics=True, tags=workflow_tags
    ):
        mlflow.sklearn.autolog(log_models=False)
        # Log DVC metadata as parameters for traceability
        mlflow.log_params(
            {
                "dvc_repo": BASE_CNFG.dvc_repo,
                "dvc_data_path": BASE_CNFG.dvc_data_path,
                "dvc_metrics_path": BASE_CNFG.dvc_metrics_path,
                "dvc_data_version": args.data_version,
            }
        )
        # Log data metrics from DVC as a JSON artifact
        mlflow.log_dict(metadata, "data_metadata.json")

        # Train
        print(
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

        print(f"Train accuracy: {accuracy:.4f}")
        mlflow.log_metric("train_acc", accuracy)

        # Log model
        registered_model_name = BASE_CNFG.mlflow_registered_model_name
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=registered_model_name,
            input_example=X_val[:1],
            # tags=
        )
        print(f"Model registered as: {registered_model_name}")

    print("✅ Done!")


if __name__ == "__main__":
    main()
