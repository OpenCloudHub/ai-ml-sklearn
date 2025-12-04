# ==============================================================================
# Serving Configuration
# ==============================================================================
#
# Pydantic-based configuration for the serving application.
#
# Settings:
#   - expected_num_features: Number of input features (12 for wine dataset)
#   - request_max_length: Maximum batch size for predictions (DOS protection)
#
# Configuration is loaded from environment variables automatically.
# Override by setting EXPECTED_NUM_FEATURES or REQUEST_MAX_LENGTH env vars.
#
# ==============================================================================

from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    """Configuration for the model serving application."""

    expected_num_features: int = 12
    request_max_length: int = 1000


SERVING_CONFIG = ServingConfig()
