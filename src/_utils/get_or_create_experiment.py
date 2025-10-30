# src/_utils/get_or_create_experiment.py
import mlflow

from _utils.logging_config import setup_logging

logger = setup_logging()


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        logger.info(
            f"Experiment '{experiment_name}' found with ID: {experiment.experiment_id}"
        )
        return experiment.experiment_id
    else:
        logger.info(
            f"Experiment '{experiment_name}' not found. Creating a new experiment."
        )
        return mlflow.create_experiment(experiment_name)
