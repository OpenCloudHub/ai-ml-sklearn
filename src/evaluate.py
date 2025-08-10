import argparse
import logging
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(test_size: float = 0.2, random_state: int = 42):
    """Load and split the wine dataset."""
    wine = datasets.load_wine()

    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, wine.feature_names, wine.target_names


def evaluate_model(model_uri: str, X_test, y_test, feature_names, target_names):
    """
    Evaluate a model using MLflow's evaluation framework.

    Args:
        model_uri: URI of the model to evaluate (e.g., runs:/<run_id>/model)
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features
        target_names: Names of target classes
    """

    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None

    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
    }


def evaluate_with_mlflow(model_uri: str, X_test, y_test, feature_names):
    """
    Evaluate model using MLflow's built-in evaluate function.
    """

    # Create evaluation dataset
    eval_data = pd.DataFrame(X_test, columns=feature_names)
    eval_data["label"] = y_test

    logger.info("Running MLflow model evaluation...")

    # Use MLflow's evaluate function
    result = mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

    return result


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")

    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="Model URI (e.g., runs:/<run_id>/model)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to evaluate (alternative to model-uri)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for data splitting"
    )
    parser.add_argument(
        "--tracking-uri", type=str, default=None, help="MLflow tracking URI"
    )
    parser.add_argument(
        "--use-mlflow-evaluate",
        action="store_true",
        help="Use MLflow's built-in evaluate function",
    )
    parser.add_argument(
        "--log-to-mlflow", action="store_true", help="Log evaluation results to MLflow"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="wine-evaluation",
        help="MLflow experiment name for logging results",
    )

    args = parser.parse_args()

    # Set tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Determine model URI
    if args.run_id:
        model_uri = f"runs:/{args.run_id}/model"
    else:
        model_uri = args.model_uri

    # Load data
    logger.info("Loading wine dataset...")
    X_train, X_test, y_test, y_test, feature_names, target_names = load_data(
        test_size=args.test_size, random_state=args.random_state
    )

    if args.use_mlflow_evaluate:
        # Use MLflow's built-in evaluation
        result = evaluate_with_mlflow(model_uri, X_test, y_test, feature_names)

        logger.info("\n" + "=" * 60)
        logger.info("MLflow Evaluation Results")
        logger.info("=" * 60)

        for metric_name, metric_value in result.metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")

        logger.info("\nGenerated Artifacts:")
        for artifact_name, path in result.artifacts.items():
            logger.info(f"  {artifact_name}: {path}")

    else:
        # Use custom evaluation
        results = evaluate_model(model_uri, X_test, y_test, feature_names, target_names)

        logger.info("\n" + "=" * 60)
        logger.info("Model Evaluation Results")
        logger.info("=" * 60)

        # Print metrics
        logger.info("\nMetrics:")
        for metric_name, metric_value in results["metrics"].items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        # Print confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info(results["confusion_matrix"])

        # Print classification report
        logger.info("\nClassification Report:")
        for class_name in target_names:
            if class_name in results["classification_report"]:
                class_metrics = results["classification_report"][class_name]
                logger.info(
                    f"  {class_name}: "
                    f"precision={class_metrics['precision']:.3f}, "
                    f"recall={class_metrics['recall']:.3f}, "
                    f"f1={class_metrics['f1-score']:.3f}"
                )

        # Log to MLflow if requested
        if args.log_to_mlflow:
            mlflow.set_experiment(args.experiment_name)

            with mlflow.start_run(run_name=f"evaluation_{model_uri.split('/')[-2]}"):
                # Log metrics
                mlflow.log_params(
                    {
                        "model_uri": model_uri,
                        "test_size": args.test_size,
                        "random_state": args.random_state,
                    }
                )

                mlflow.log_metrics(results["metrics"])

                # Log confusion matrix as artifact
                cm_df = pd.DataFrame(
                    results["confusion_matrix"],
                    index=target_names,
                    columns=target_names,
                )
                cm_df.to_csv("confusion_matrix.csv")
                mlflow.log_artifact("confusion_matrix.csv")

                # Log classification report
                report_df = pd.DataFrame(results["classification_report"]).T
                report_df.to_csv("classification_report.csv")
                mlflow.log_artifact("classification_report.csv")

                logger.info(
                    f"\nEvaluation logged to MLflow run: {mlflow.active_run().info.run_id}"
                )


if __name__ == "__main__":
    main()
