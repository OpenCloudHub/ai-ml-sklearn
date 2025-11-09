import os
from typing import List

import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve

from _utils.logging_config import setup_logging

logger = setup_logging()

app = FastAPI(
    title="🍷 Wine Classifier API",
    description="ML model endpoint example for wine classification using MLflow and Ray Serve",
    version="1.0.0",
)


# Pydantic models for request/response validation
class WineFeatures(BaseModel):
    """Input features for wine classification"""

    features: List[List[float]] = Field(
        ...,
        description="List of wine feature vectors. Each vector should contain 13 float values.",
        example=[
            [
                13.2,
                2.77,
                2.51,
                18.5,
                1015.0,
                2.95,
                3.33,
                0.29,
                10.2,
                0.56,
                1.68,
                5.0,
                1.04,
            ],
            [
                12.37,
                1.17,
                1.92,
                19.6,
                162.0,
                1.45,
                2.52,
                0.24,
                11.8,
                0.61,
                1.65,
                3.8,
                0.61,
            ],
        ],
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: List[int] = Field(
        ..., description="Predicted wine classes for the input features"
    )
    model_name: str


class RootResponse(BaseModel):
    """Response model for root endpoint"""

    status: str
    model: str
    features_expected: int
    output_classes: List[str]


@serve.deployment(
    # Example autoscaling configuration
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "max_replicas": 10,
    #     "target_num_ongoing_requests_per_replica": 5,  # Scale when avg requests > 5
    # }
)
@serve.ingress(app)
class WineClassifier:
    """Wine Classifier API.

    Serves a pre-trained wine classification model from MLflow and provides
    endpoints for health checks and predictions."""

    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "/workspace/project/model")
        self.model_name = os.getenv("MODEL_NAME", "unknown")
        self.model = mlflow.sklearn.load_model(model_path)
        logger.info(f"Loaded model '{self.model_name}' from {model_path}")

    @app.get("/", response_model=RootResponse, summary="Health Check")
    async def root(self):
        return {
            "status": "healthy",
            "model": self.model_name,
            "features_expected": 13,
            "output_classes": ["0", "1", "2"],
        }

    @app.post(
        "/predict", response_model=PredictionResponse, summary="Predict Wine Class"
    )
    async def predict(self, wine_features: WineFeatures):
        """
        Predict wine class based on features

        - **wine_features**: List of wine feature vectors
        """
        logger.debug(f"Received features for prediction: {wine_features.features}")
        features = np.array(wine_features.features)
        predictions = self.model.predict(features)
        logger.debug(f"Predictions: {predictions}")

        return PredictionResponse(
            predictions=predictions.tolist(),
            model_name=self.model_name,
        )


# Ray Serve deployment
deployment = WineClassifier.bind()
