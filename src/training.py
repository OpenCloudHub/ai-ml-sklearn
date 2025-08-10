import argparse
import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _utils.get_or_create_experiment import get_or_create_experiment
from _utils.logging_config import setup_logging

# Remove duplicate logger setup and just use:
logger = setup_logging()


def load_data(test_size, random_state):
    """Load the wine dataset."""
    wine = datasets.load_wine()

    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=test_size,
        random_state=random_state,
        stratify=wine.target,  # Add stratification
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
    """
    Train a model with given parameters without evaluation.
    Returns the trained pipeline.
    """
    # Build and train pipeline
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
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(pipeline, X_test, y_test, eval_data, log_model=True):
    """
    Evaluate a trained model and optionally log it to MLflow.
    Returns evaluation results and model_uri if logged.
    """
    # Calculate basic metrics
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Test accuracy: {accuracy:.4f}")

    model_uri = None
    if log_model:
        # Log model with signature
        signature = infer_signature(X_test, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
        )
        model_uri = model_info.model_uri

        # Run MLflow evaluate - let it fail if it fails
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
        )
        return result, model_uri
    else:
        # Just log accuracy without model
        mlflow.log_metric("accuracy", accuracy)
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
    """
    Combined function that trains and evaluates with proper autologging.
    This is what we'll use for both standalone and hyperparameter tuning.
    """
    # Configure autologging based on context
    mlflow.sklearn.autolog(
        log_models=False,  # We'll log model manually for better control
        log_model_signatures=False,
        log_input_examples=False,
        silent=True,
    )

    # Train model
    pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )

    # Evaluate model
    result, model_uri = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        eval_data=eval_data,
        log_model=log_model,
    )

    return result, model_uri, pipeline


def register_model(
    model_uri: str,
    registered_model_name: str,
    accuracy: float,
    threshold: float | None = 0.8,
) -> bool:
    """
    Register model if threshold is met.

    Args:
        model_uri: MLflow model URI (e.g., runs:/run_id/model)
        accuracy: Model accuracy score
        threshold: Minimum accuracy required for registration

    Returns:
        True if registered, False otherwise
    """

    # Check threshold if set
    if threshold:
        threshold = float(threshold)
        if accuracy < threshold:
            logger.info(
                f"Accuracy {accuracy:.4f} below threshold {threshold:.4f}, skipping registration"
            )
            return False

    # Register the model
    try:
        logger.info(
            f"Registering model to {registered_model_name} with accuracy {accuracy:.4f}"
        )
        model_version = mlflow.register_model(model_uri, registered_model_name)
        logger.info(
            f"Model registered as {registered_model_name} version {model_version.version}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return False


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
    run_name = f"standalone_wine_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Set tags
        mlflow.set_tags(
            {
                "project": "Wine Classification",
                "model_family": "LogisticRegression",
                "run_type": "standalone",
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

        if result:
            logger.info(f"Training completed. Final metrics: {result.metrics}")

    # Check if we want to register model
    accuracy = result.metrics.get("accuracy_score")
    registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
    if not registered_model_name:
        logger.debug("Registered model name not set, skipping registration")
        return False
    else:
        register_model(
            model_uri=model_uri,
            registered_model_name=registered_model_name,
            accuracy=accuracy,
        )


if __name__ == "__main__":
    main()
