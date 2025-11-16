"""Schema definitions for the serving module."""

from datetime import datetime
from enum import StrEnum, auto
from typing import Annotated, List

import numpy as np
from pydantic import AfterValidator, BaseModel, Field

from src._utils.logging import get_logger
from src.serving.config import SERVING_CONFIG

logger = get_logger(__name__)


def validate_feature_vectors(features: List[List[float]]) -> List[List[float]]:
    """Validate input feature vectors."""
    for i, feature_vec in enumerate(features):
        # Check length of each vector
        if len(feature_vec) != SERVING_CONFIG.expected_num_features:
            raise ValueError(
                f"Sample {i}: Expected {SERVING_CONFIG.expected_num_features} features, got {len(feature_vec)}"
            )

        # Check for NaN or infinite values in this vector
        if any(not np.isfinite(f) for f in feature_vec):
            raise ValueError(f"Sample {i}: Features contain NaN or infinite values")

    return features


class PredictionRequest(BaseModel):
    """Input model for predictions with validation."""

    features: Annotated[
        List[List[float]],
        AfterValidator(validate_feature_vectors),
        Field(
            min_length=1,
            max_length=SERVING_CONFIG.request_max_length,  # Prevent DOS attacks with huge batches
            description="List of wine feature vectors. Each vector must contain exactly 12 float values: "
            "[fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, "
            "free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type]. "
            "wine_type: 0 = red, 1 = white",
            example=[
                [
                    7.4,
                    0.7,
                    0.0,
                    1.9,
                    0.076,
                    11.0,
                    34.0,
                    0.9978,
                    3.51,
                    0.56,
                    9.4,
                    0,
                ],
                [
                    7.0,
                    0.27,
                    0.36,
                    20.7,
                    0.045,
                    45.0,
                    170.0,
                    1.001,
                    3.0,
                    0.45,
                    8.8,
                    1,
                ],
            ],
        ),
    ]


class Prediction(BaseModel):
    """Single prediction result."""

    quality_score: int = Field(
        ..., description="Predicted quality score (3-9)", ge=3, le=9
    )
    confidence: float = Field(
        ..., description="Prediction confidence (0-1)", ge=0.0, le=1.0
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predictions: List[Prediction] = Field(
        ..., description="List of predictions for each input wine sample"
    )
    model_uri: str = Field(..., description="URI of the model used")
    timestamp: datetime = Field(..., description="Prediction timestamp UTC")
    processing_time_ms: float = Field(..., description="Time taken to process request")


class ModelInfo(BaseModel):
    """Model metadata information."""

    model_uri: str = Field(..., description="URI of the model used")
    model_uuid: str = Field(..., description="MLflow model UUID")
    run_id: str = Field(..., description="MLflow run ID associated with the model")
    model_signature: dict | None = Field(None, description="MLflow model signature")
    data_version: str | None = Field(
        None, description="DVC data version used for training"
    )
    training_timestamp: datetime | None = Field(
        None, description="When the model was trained"
    )


class APIStatus(StrEnum):
    """API status enumeration."""

    LOADING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    NOT_READY = auto()


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: APIStatus = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_uri: str | None = Field(None, description="Current model URI")
    uptime_seconds: float | None = Field(None, description="Service uptime in seconds")


class RootResponse(BaseModel):
    """Response model for root endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    docs: str = Field(..., description="URL to API documentation")
    health: str = Field(..., description="URL to health check endpoint")


class ErrorDetail(BaseModel):
    """Error detail model."""

    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: ErrorDetail = Field(..., description="Error details")
