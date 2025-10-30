# src/_utils/logging_config.py
import logging
import warnings

import optuna
import urllib3


def setup_logging(level=logging.INFO, library_level=logging.WARNING):
    """
    Configure logging for the entire project.
    Sets log levels across libraries and disables some warnings.
    Call this at the start of your main scripts.
    Adjust to your individual requirements.
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

    # Suppress urllib3 SSL warnings
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Optuna logging
    optuna.logging.set_verbosity(library_level)

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    return logging.getLogger(__name__)
