# _utils/logging_config.py
import logging
import warnings


def setup_logging(level=logging.INFO, library_level=logging.WARNING):
    """
    Configure logging for the entire project.
    Call this at the start of your main scripts.
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Reconfigure even if already configured
    )

    # Suppress verbose libraries
    logging.getLogger("mlflow").setLevel(library_level)
    logging.getLogger("mlflow.sklearn").setLevel(library_level)
    logging.getLogger("mlflow.tracking").setLevel(library_level)
    logging.getLogger("mlflow.utils.autologging_utils").setLevel(library_level)

    # Optuna logging
    import optuna

    optuna.logging.set_verbosity(library_level)

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    return logging.getLogger(__name__)
