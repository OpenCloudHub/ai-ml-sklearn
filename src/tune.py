# import argparse
# import os
# from datetime import datetime
# from functools import partial
# from typing import Any, Dict, Tuple

# import mlflow
# import numpy.typing as npt
# import optuna
# import pandas as pd
# from optuna import Study, Trial
# from sklearn.metrics import accuracy_score

# from _utils.get_or_create_experiment import get_or_create_experiment
# from _utils.logging_config import setup_logging
# from _utils.mlflow_tags import set_mlflow_experiment_tags
# from training.train import load_data, train_and_evaluate

# logger = setup_logging()


# def objective(
#     trial: Trial,
#     X_train: npt.NDArray,
#     X_test: npt.NDArray,
#     y_train: npt.NDArray,
#     y_test: npt.NDArray,
#     eval_data: pd.DataFrame,
#     parent_run_id: str,
#     random_state: int,
# ) -> float:
#     """
#     Objective function for optuna with explicit parent_run_id for proper nesting.

#     Args:
#         trial: Optuna trial object
#         X_train: Training features
#         X_test: Test features
#         y_train: Training labels
#         y_test: Test labels
#         eval_data: DataFrame with test data and labels
#         parent_run_id: MLflow parent run ID for nesting
#         random_state: Random seed for reproducibility

#     Returns:
#         Accuracy score for this trial
#     """
#     try:
#         # Start child run with explicit parent
#         with mlflow.start_run(
#             run_name=f"trial_{trial.number}", nested=True, parent_run_id=parent_run_id
#         ):
#             # Set trial-specific tags
#             mlflow.set_tags(
#                 {"trial_number": trial.number, "trial_type": "hyperparameter_search"}
#             )

#             # Define hyperparameters
#             C: float = trial.suggest_float("C", 1e-3, 100.0, log=True)
#             max_iter: int = trial.suggest_int("max_iter", 100, 1000)
#             solver: str = trial.suggest_categorical(
#                 "solver", ["lbfgs", "liblinear", "saga"]
#             )

#             # Train and evaluate
#             result, model_uri, pipeline = train_and_evaluate(
#                 X_train=X_train,
#                 X_test=X_test,
#                 y_train=y_train,
#                 y_test=y_test,
#                 eval_data=eval_data,
#                 C=C,
#                 max_iter=max_iter,
#                 solver=solver,
#                 random_state=random_state,
#                 log_model=False,
#             )

#             # Extract accuracy from result or use a default
#             accuracy: float
#             if result and result.metrics:
#                 accuracy = result.metrics.get("accuracy_score", 0.0)
#             else:
#                 # Fallback: calculate accuracy directly
#                 y_pred = pipeline.predict(X_test)
#                 accuracy = accuracy_score(y_test, y_pred)

#             # Log trial-specific metric
#             mlflow.log_metric("trial_accuracy", accuracy)

#             return accuracy

#     except Exception as e:
#         logger.error(f"Trial {trial.number} failed: {e}")
#         return 0.0


# def hyperparameter_search(
#     n_trials: int, test_size: float, random_state: int
# ) -> Tuple[Dict[str, Any], float]:
#     """
#     Run hyperparameter optimization using Optuna with parent-child relationship.

#     Args:
#         n_trials: Number of optimization trials to run
#         test_size: Proportion of dataset to include in test split
#         random_state: Random seed for reproducibility

#     Returns:
#         Tuple of (best parameters dict, best score)
#     """
#     # Set the current active MLflow experiment
#     experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality")
#     experiment_id = get_or_create_experiment(experiment_name)

#     timestamp = datetime.now()
#     run_name = f"optuna_optimization_{timestamp.strftime('%Y%m%d_%H%M%S')}"

#     # Start parent run and capture its ID
#     with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as parent_run:
#         parent_run_id: str = parent_run.info.run_id

#         logger.info(f"Started parent run with ID: {parent_run_id}")

#         # Load data once before optimization
#         logger.info("Loading data...")
#         X_train, X_test, y_train, y_test, eval_data = load_data(
#             test_size=test_size, random_state=random_state
#         )

#         # Log parent run configuration
#         mlflow.log_params(
#             {
#                 "n_trials": n_trials,
#                 "test_size": test_size,
#                 "random_state": random_state,
#                 "optimization_direction": "maximize",
#                 "sampler": "TPESampler",
#             }
#         )

#         # Set parent run tags
#         set_mlflow_experiment_tags(
#             {
#                 "project": "Wine Classification",
#                 "optimizer_engine": "optuna",
#                 "model_family": "logistic_regression",
#                 "feature_set_version": "1",
#                 "run_type": "parent_optimization",
#             }
#         )

#         # Create objective with all context including parent_run_id
#         objective_with_context = partial(
#             objective,
#             X_train=X_train,
#             X_test=X_test,
#             y_train=y_train,
#             y_test=y_test,
#             eval_data=eval_data,
#             parent_run_id=parent_run_id,
#             random_state=random_state,
#         )

#         # Initialize and run Optuna study
#         study: Study = optuna.create_study(
#             direction="maximize",
#             study_name="wine_classifier_study",
#             sampler=optuna.samplers.TPESampler(seed=random_state),
#         )

#         logger.info("Running local optimization")
#         study.optimize(
#             objective_with_context,
#             n_trials=n_trials,
#             n_jobs=1,  # Single job for proper parent-child relationship
#             show_progress_bar=True,
#         )

#         # Log best results to parent run
#         best_params: Dict[str, Any] = study.best_params
#         mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
#         mlflow.log_metric("best_accuracy", study.best_value)
#         mlflow.log_metric("n_trials_completed", len(study.trials))

#         logger.info(f"Best trial: {study.best_trial.number}")
#         logger.info(f"Best accuracy: {study.best_value:.4f}")
#         logger.info(f"Best parameters: {best_params}")

#         # Train final model with best parameters in a new child run
#         logger.info("\nTraining final model with best parameters...")
#         with mlflow.start_run(
#             run_name="best_model", nested=True, parent_run_id=parent_run_id
#         ):
#             # Set tags for final model
#             set_mlflow_experiment_tags(
#                 {
#                     "project": "Wine Classification",
#                     "model_family": "LogisticRegression",
#                     "training_framework": "scikit-learn",
#                     "run_type": "final_model_training",
#                     "best_trial_number": str(study.best_trial.number),
#                 }
#             )

#             # Train with best parameters and full logging
#             result, model_uri, pipeline = train_and_evaluate(
#                 X_train=X_train,
#                 X_test=X_test,
#                 y_train=y_train,
#                 y_test=y_test,
#                 eval_data=eval_data,
#                 C=best_params["C"],
#                 max_iter=best_params["max_iter"],
#                 solver=best_params["solver"],
#                 random_state=random_state,
#                 log_model=True,  # Log the final model
#             )

#             # Register the model only if MLFLOW_REGISTERED_MODEL_NAME is set
#             registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
#             if registered_model_name:
#                 run_id = mlflow.active_run().info.run_id
#                 model_uri_str = f"runs:/{run_id}/model"

#                 try:
#                     mlflow.register_model(model_uri_str, registered_model_name)
#                     logger.info(
#                         f"Model registered successfully as '{registered_model_name}'"
#                     )
#                 except Exception as e:
#                     logger.warning(f"Could not register model: {e}")
#             else:
#                 logger.info(
#                     "MLFLOW_REGISTERED_MODEL_NAME not set - skipping model registration"
#                 )

#             if result:
#                 logger.info(f"Final model evaluation metrics: {result.metrics}")

#         logger.info("\nHyperparameter optimization complete!")

#         return best_params, study.best_value


# def parse_arguments() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Hyperparameter tuning for wine classifier"
#     )
#     parser.add_argument(
#         "--n_trials", type=int, default=5, help="Number of optimization trials"
#     )
#     parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
#     parser.add_argument("--random_state", type=int, default=42, help="Random seed")
#     return parser.parse_args()


# def main() -> None:
#     """Main hyperparameter search function."""
#     args = parse_arguments()

#     best_params, best_score = hyperparameter_search(
#         n_trials=args.n_trials, test_size=args.test_size, random_state=args.random_state
#     )

#     print("\n" + "=" * 50)
#     print("Hyperparameter Search Complete!")
#     print("=" * 50)
#     print(f"Best Score: {best_score:.4f}")
#     print("Best Parameters:")
#     for param, value in best_params.items():
#         print(f"  {param}: {value}")


# if __name__ == "__main__":
#     main()
