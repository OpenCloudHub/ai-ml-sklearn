import json
from datetime import datetime, timezone
from typing import List

import dvc.api
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from src._utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="🍷 Wine Quality Classifier API",
    description="Wine Quality classification using Ray Serve + MLflow",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="List of wine feature vectors. Each vector should contain 12 float values: "
        "[fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, "
        "free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type]",
        example=[
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0],
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8, 0],
        ],
    )


class Prediction(BaseModel):
    """Single prediction result"""

    quality_score: int = Field(..., description="Predicted quality score (3-9)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: List[Prediction] = Field(
        ..., description="List of predictions for each input wine sample"
    )
    model_uri: str = Field(..., description="URI of the model used")
    timestamp: datetime = Field(..., description="Prediction timestamp UTC")


class ModelInfo(BaseModel):
    model_uri: str = Field(..., description="URI of the model used")
    model_uuid: str = Field(..., description="MLflow model UUID")
    run_id: str = Field(..., description="MLflow run ID associated with the model")
    data_version: str = Field(..., description="DVC data version used for training")
    model_signature: str | None = Field(None, description="MLflow model signature")
    expected_features: List[str] = Field(
        default=[
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "wine_type",
        ],
        description="Expected feature names in order",
    )
    quality_range: List[int] = Field(
        default=[3, 9],
        description="Possible quality score range",
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(..., description="API health status")
    model_info: ModelInfo | None = Field(
        None, description="Information about the loaded model"
    )


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 3},
)
@serve.ingress(app)
class WineClassifier:
    def __init__(self, model_uri: str | None = None):
        """Initialize the classifier, optionally with a model URI."""
        self.model = None
        self.model_info: ModelInfo | None = None
        self.data_version: str | None = None
        self.metadata: dict | None = None

        # Load model if URI provided at init
        if model_uri:
            self._load_model(model_uri)

    def _load_model(self, model_uri: str):
        """Internal method to load model and fetch metadata."""
        logger.info(f"📦 Loading model from: {model_uri}")

        # Get model info from MLflow
        info = mlflow.models.get_model_info(model_uri)

        # Fetch the data version from the model's training run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(info.run_id)

        # Get data version from run parameters
        self.data_version = run.data.params.get("dvc_data_version")
        if not self.data_version:
            raise ValueError(
                f"Model at {model_uri} was trained without data_version parameter. "
                "Cannot determine which data version was used."
            )

        logger.info(f"📊 Using data version: {self.data_version}")

        # Fetch metadata from DVC to get feature info

        metadata_content = dvc.api.read(
            "data/wine-quality/processed/metadata.json",
            repo="https://github.com/OpenCloudHub/data-registry",
            rev=self.data_version,
        )
        self.metadata = json.loads(metadata_content)

        logger.info(f"📊 Dataset: {self.metadata['dataset']['name']}")
        logger.info(f"   Features: {self.metadata['summary']['num_features']}")
        logger.info(f"   Quality range: {self.metadata['schema']['target']['range']}")

        # Load the model
        self.model = mlflow.sklearn.load_model(model_uri)

        # Build ModelInfo
        feature_names = list(self.metadata["schema"]["features"].keys())
        quality_range = self.metadata["schema"]["target"]["range"]

        self.model_info = ModelInfo(
            model_uri=model_uri,
            model_uuid=info.model_uuid,
            run_id=info.run_id,
            data_version=self.data_version,
            model_signature=str(info.signature) if info.signature else None,
            expected_features=feature_names,
            quality_range=quality_range,
        )

        logger.info("✅ Model loaded successfully")
        logger.info(f"   Model UUID: {self.model_info.model_uuid}")
        logger.info(f"   Run ID: {self.model_info.run_id}")
        logger.info(f"   Data version: {self.data_version}")

    def reconfigure(self, config: dict) -> None:
        """Handle model updates without restarting the deployment.

        Update via: serve.run(..., user_config={"model_uri": "new_uri"})
        """
        new_model_uri = config.get("model_uri")

        if not new_model_uri:
            logger.warning("⚠️ No model_uri provided in config")
            return

        # If no model loaded yet (first time), load it
        if self.model is None or self.model_info is None:
            logger.info("🆕 Initial model load")
            self._load_model(new_model_uri)
            return

        # If model already loaded, check if URI changed
        if new_model_uri == self.model_info.model_uri:
            logger.info("ℹ️ No model update needed (same URI)")
            return

        # URI changed, reload model
        logger.info(
            f"🔄 Updating model from {self.model_info.model_uri} to {new_model_uri}"
        )
        self._load_model(new_model_uri)

    @app.get("/", response_model=HealthResponse, summary="Health Check")
    async def root(self):
        """Health check endpoint."""
        if self.model_info is None:
            return HealthResponse(
                status="not_ready",
                model_info=None,
            )

        return HealthResponse(
            status="healthy",
            model_info=self.model_info,
        )

    @app.post(
        "/predict", response_model=PredictionResponse, summary="Predict Wine Quality"
    )
    async def predict(self, request: PredictionRequest):
        """
        Predict wine quality from physicochemical properties.

        Input should be a list of feature vectors with 12 values each:
        [fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
         free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type]

        wine_type: 0 = red, 1 = white
        """
        # Check if model is loaded
        if self.model is None or self.model_info is None:
            raise HTTPException(
                503, "Model not loaded. Configure the deployment with a model_uri."
            )

        try:
            # Convert to numpy array
            features = np.array(request.features, dtype=np.float64)

            # Validate input shape
            if features.shape[1] != 12:
                raise ValueError(
                    f"Expected 12 features, got {features.shape[1]}. "
                    f"Expected features: {self.model_info.expected_features}"
                )

            # Get predictions and probabilities
            quality_scores = self.model.predict(features)
            probabilities = self.model.predict_proba(features)

            # Get confidence for each prediction (max probability)
            confidences = np.max(probabilities, axis=1)

            # Build predictions
            predictions = [
                Prediction(
                    quality_score=int(score),
                    confidence=float(conf),
                )
                for score, conf in zip(quality_scores, confidences)
            ]

            return PredictionResponse(
                predictions=predictions,
                model_uri=self.model_info.model_uri,
                timestamp=datetime.now(timezone.utc),
            )

        except HTTPException:
            raise
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise HTTPException(500, f"Prediction failed: {e}")


class AppBuilderArgs(BaseModel):
    model_uri: str | None = Field(
        None,
        description="MLflow model URI to load (e.g., models:/wine-classifier/1)",
    )


def app_builder(args: AppBuilderArgs) -> Application:
    """Helper function to build the deployment with optional model URI."""
    return WineClassifier.bind(model_uri=args.model_uri)
