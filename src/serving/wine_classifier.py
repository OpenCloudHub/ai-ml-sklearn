import os
from typing import List

import numpy as np
from fastapi import FastAPI
from mlflow import MlflowClient, set_tracking_uri
from pydantic import BaseModel
from ray import serve

# FastAPI app
app = FastAPI(
    title="🍷 Wine Classifier API",
    description="ML model for wine classification using MLflow and Ray Serve",
    version="1.0.0",
)


# Pydantic models for request/response
class WineFeatures(BaseModel):
    features: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[int]
    model_name: str
    model_version: str


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class WineClassifier:
    def __init__(
        self, model_name: str = "prod.wine-classifier", model_alias: str = "champion"
    ):
        client = MlflowClient()

        # Set MLflow tracking URI
        mlflow_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "https://mlflow.ai.internal.opencloudhub.org"
        )

        # Handle SSL for local dev
        if os.getenv("MLFLOW_TRACKING_INSECURE_TLS"):
            import mlflow.tracking._tracking_service.client

            mlflow.tracking._tracking_service.client.VERIFY = False

        set_tracking_uri(mlflow_uri)

        # Load model from MLflow
        try:
            model_version = client.get_model_version_by_alias(model_name, model_alias)
            model_uri = f"models:/{model_name}@{model_alias}"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_name = model_name
            self.model_version = model_version.version
        except Exception as e:
            print(f"Error loading model: {e}")

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

        - **features**: List of wine feature vectors (each with 13 features)
        """
        features = np.array(wine_features.features)
        predictions = self.model.predict(features)

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


# Bind the deployment
deployment = WineClassifier.bind()
