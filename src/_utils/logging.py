# ==============================================================================
# Logging Configuration
# ==============================================================================
#
# Rich-based logging setup for pretty console output.
#
# Features:
#   - Color-coded log levels with custom theme
#   - Custom SUCCESS level (between INFO and WARNING)
#   - Section headers with rule() for visual separation
#   - Ray-compatible (works in both driver and worker processes)
#   - Auto-suppresses warnings in production/training environments
#
# Usage:
#   from src._utils.logging import get_logger, log_section
#
#   logger = get_logger(__name__)
#   logger.info("Standard info message")
#   logger.success("Success message in green")  # Custom level
#   log_section("Section Title", "🚀")  # Visual separator
#
# Ray Compatibility:
#   - Configures logging after ray.init() but before workers spawn
#   - Uses propagate=False to avoid duplicate handlers
#   - Uses print() for section headers (more reliable with Ray)
#
# Environment-based Warning Suppression:
#   When ENVIRONMENT is "production" or "training", Python warnings
#   and noisy library loggers (MLflow, urllib3, etc.) are silenced.
#
# ==============================================================================

import logging
import os
import sys
import warnings

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ==============================================================================
# Suppress warnings in production/training environments
# ==============================================================================
_ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

if _ENVIRONMENT in ("production", "training"):
    # Suppress Python warnings (DeprecationWarning, FutureWarning, etc.)
    warnings.filterwarnings("ignore")

    # Silence noisy library loggers
    for _logger_name in (
        "mlflow",
        "mlflow.tracking",
        "mlflow.utils",
        "mlflow.sklearn",
        "mlflow.models",
        "urllib3",
        "urllib3.connectionpool",
        "git",
        "git.cmd",
        "fsspec",
        "s3fs",
        "botocore",
        "boto3",
    ):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Custom theme
CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "success": "green bold",
        "debug": "blue",
    }
)

# Global console instance
console = Console(
    theme=CUSTOM_THEME,
    file=sys.stdout,
    force_terminal=True,
    force_jupyter=False,
    force_interactive=False,
    color_system="truecolor",  # Use full color support
    legacy_windows=False,
)

# Add custom SUCCESS level
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, message, *args, **kwargs):
    """Log a success message."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = success


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with RichHandler for pretty formatting.

    According to Ray docs, you should configure logging AFTER ray.init()
    but BEFORE creating workers. This function returns a logger that
    will work in both driver and worker processes.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            enable_link_path=False,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def log_section(title: str, emoji: str = "📌") -> None:
    """Print a section header - uses print() for compatibility with Ray."""
    console.rule(f"{emoji} {title}", style="bold cyan")
