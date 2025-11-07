# src/serving/wine_classifier.py
import os
from typing import List

import mlflow.tracking._tracking_service.client
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve

from _utils.logging_config import setup_logging

logger = setup_logging()

# FastAPI app
app = FastAPI(
    title="🍷 Wine Classifier API",
    description="ML model for wine classification using MLflow and Ray Serve",
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
    model_version: str


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
    """Wine Classifier API with baked-in model"""

    # def __init__(self, model_name: str | None = None, model_alias: str | None = None):
    #     client = MlflowClient()

    #     # Set MLflow tracking URI
    #     mlflow_uri = os.getenv(
    #         "MLFLOW_TRACKING_URI", "https://mlflow.ai.internal.opencloudhub.org"
    #     )
    #     logger.debug(f"Setting MLflow tracking URI to: {mlflow_uri}")

    #     # Handle SSL for local dev
    #     if os.getenv("MLFLOW_TRACKING_INSECURE_TLS"):
    #         mlflow.tracking._tracking_service.client.VERIFY = False
    #         logger.warning("MLflow TLS verification is disabled")

    #     set_tracking_uri(mlflow_uri)

    #     # Load model from MLflow
    #     try:
    #         model_version = client.get_model_version_by_alias(model_name, model_alias)
    #         model_uri = f"models:/{model_name}@{model_alias}"
    #         self.model = mlflow.sklearn.load_model(model_uri)
    #         self.model_name = model_name
    #         self.model_version = model_version.version
    #         logger.info(
    #             f"Loaded model '{self.model_name}' version '{self.model_version}' from MLflow"
    #         )
    #     except Exception as e:
    #         logger.error(f"Error loading model: {e}")
    #         raise

    def __init__(self):
        # Load model from local filesystem (baked into image)
        model_path = os.getenv("MODEL_PATH", "/workspace/project/model")
        self.model_name = os.getenv(
            "MODEL_NAME", "unknown"
        )  # TODO: currently this is staging.wine_classifier as this is needed for loading model in the bilding phase but ye that's bad maybe?
        self.model_version = os.getenv("MODEL_VERSION", "unknown")

        logger.info(f"Loading model from {model_path}")
        self.model = mlflow.sklearn.load_model(model_path)
        logger.info(f"Loaded model '{self.model_name}' version '{self.model_version}'")

    @app.get("/", summary="Health Check")
    async def root(self):
        """Health check endpoint"""
        return {"message": "Wine Classifier API is running", "status": "healthy"}

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
            model_version=self.model_version,
        )

    @app.get("/model/info", summary="Model Information")
    async def model_info(self):
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "features_expected": 13,
            "output_classes": ["0", "1", "2"],
        }


deployment = WineClassifier.bind()
