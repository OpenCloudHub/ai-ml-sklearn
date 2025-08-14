import argparse
import os
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import ray
from mlflow.models import infer_signature
from ray.util.joblib import register_ray
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _utils.get_or_create_experiment import get_or_create_experiment
from _utils.logging_config import setup_logging

# Set up ray
# Always use 'auto' - works everywhere
ray.init(address="auto", ignore_reinit_error=True)
register_ray()


logger = setup_logging()


def load_data(test_size, random_state):
    """Load the wine dataset."""
    wine = datasets.load_wine()

    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=test_size,
        random_state=random_state,
        stratify=wine.target,
    )

    eval_data = pd.DataFrame(X_test, columns=wine.feature_names)
    eval_data["label"] = y_test
    return X_train, X_test, y_train, y_test, eval_data


def train_model(
    X_train,
    y_train,
    C=1.0,
    max_iter=100,
    solver="lbfgs",
    random_state=42,
):
    """Train a model with given parameters."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver=solver,
                    random_state=random_state,
                    multi_class="auto",
                ),
            ),
        ]
    )

    logger.info(f"Training with C={C}, max_iter={max_iter}, solver={solver}")

    # Necessary to run distributed sklearn
    with joblib.parallel_backend("ray"):
        pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(pipeline, X_test, y_test, eval_data, log_model=True):
    """Evaluate a trained model and optionally log it to MLflow."""
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Test accuracy: {accuracy:.4f}")

    model_uri = None
    if log_model:
        signature = infer_signature(X_test, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
        )
        model_uri = model_info.model_uri

        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
        )
        return result, model_uri
    else:
        mlflow.log_metric("accuracy_score", accuracy)
        return None, model_uri


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    eval_data,
    C=1.0,
    max_iter=100,
    solver="lbfgs",
    random_state=42,
    log_model=True,
):
    """Combined training and evaluation."""
    mlflow.sklearn.autolog(
        log_models=False,
        log_model_signatures=False,
        log_input_examples=False,
        silent=True,
    )

    pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )

    result, model_uri = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        eval_data=eval_data,
        log_model=log_model,
    )

    return result, model_uri, pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train wine classifier")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        help="Solver algorithm",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_arguments()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Wine Classification")
    experiment_id = get_or_create_experiment(experiment_name)

    timestamp = datetime.now()
    run_name = f"ray_wine_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tags(
            {
                "project": "Wine Classification",
                "model_family": "LogisticRegression",
                "run_type": "ray_job",
                "execution_mode": "ray",
            }
        )

        # Load data
        X_train, X_test, y_train, y_test, eval_data = load_data(
            test_size=args.test_size, random_state=args.random_state
        )

        # Train and evaluate
        result, model_uri, pipeline = train_and_evaluate(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            eval_data=eval_data,
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            random_state=args.random_state,
            log_model=True,
        )

        # Register the model only if MLFLOW_REGISTERED_MODEL_NAME is set
        registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
        if registered_model_name:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            try:
                mlflow.register_model(model_uri, registered_model_name)
                logger.info(
                    f"Model registered successfully as '{registered_model_name}'"
                )
            except Exception as e:
                logger.warning(f"Could not register model: {e}")
        else:
            logger.info(
                "MLFLOW_REGISTERED_MODEL_NAME not set - skipping model registration"
            )

        if result:
            logger.info(f"Training completed. Final metrics: {result.metrics}")
            print(f"🎉 Job completed! Model URI: {model_uri}")


if __name__ == "__main__":
    main()
