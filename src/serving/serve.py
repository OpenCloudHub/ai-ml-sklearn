from datetime import datetime, timezone

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from src._utils.logging import get_logger
from src.serving.schemas import (
    APIStatus,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)

logger = get_logger(__name__)

app = FastAPI(
    title="🍷 Wine Quality Classifier API",
    description="Wine Quality classification using Ray Serve + MLflow",
    version="1.0.0",
)


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 3},
)
@serve.ingress(app)
class WineClassifier:
    def __init__(self, model_uri: str | None = None) -> None:
        """Initialize the classifier, optionally with a model URI."""
        logger.info("🍷 Initializing Wine Classifier Service")
        self.status = APIStatus.NOT_READY
        self.model = None
        self.model_info: ModelInfo | None = None
        self.start_time = datetime.now(timezone.utc)

        # Load model if URI provided at init
        if model_uri:
            try:
                self._load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load model during initialization: {e}")
                self.status = APIStatus.UNHEALTHY

    def _load_model(self, model_uri: str) -> None:
        """Internal method to load model and fetch metadata."""
        logger.info(f"📦 Loading model from: {model_uri}")
        self.status = APIStatus.LOADING

        try:
            # Get model info first to validate URI
            info = mlflow.models.get_model_info(model_uri)

            # Load the model
            self.model = mlflow.sklearn.load_model(model_uri)

            # Get training run metadata
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(info.run_id)

            # Extract data version and training timestamp
            data_version = run.data.params.get("dvc_data_version")
            training_timestamp = datetime.fromtimestamp(
                run.info.start_time / 1000.0, tz=timezone.utc
            )

            # Build ModelInfo
            self.model_info = ModelInfo(
                model_uri=model_uri,
                model_uuid=info.model_uuid,
                run_id=info.run_id,
                model_signature=info.signature.to_dict() if info.signature else None,
                data_version=data_version,
                training_timestamp=training_timestamp,
            )

            self.status = APIStatus.HEALTHY
            logger.success("✅ Model loaded successfully")
            logger.info(f"   Model UUID: {self.model_info.model_uuid}")
            logger.info(f"   Run ID: {self.model_info.run_id}")
            if data_version:
                logger.info(f"   Data version: {data_version}")

        except mlflow.exceptions.MlflowException as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ MLflow error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model from MLflow: {str(e)}",
            )
        except Exception as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ Unexpected error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error loading model: {str(e)}",
            )

    def reconfigure(self, config: dict) -> None:
        """Handle model updates without restarting the deployment.

        Check: https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html

        Update via: serve.run(..., user_config={"model_uri": "new_uri"})
        """
        new_model_uri = config.get("model_uri")

        if not new_model_uri:
            logger.warning("⚠️ No model_uri provided in config")
            return

        # If no model loaded yet, load it
        if self.model_info is None:
            logger.info("🆕 Initial model load via reconfigure")
            self._load_model(new_model_uri)
            return

        # Check if URI changed
        if self.model_info.model_uri != new_model_uri:
            logger.info(
                f"🔄 Updating model from {self.model_info.model_uri} to {new_model_uri}"
            )
            self._load_model(new_model_uri)
        else:
            logger.info("ℹ️ Model URI unchanged, skipping reload")

    @app.get(
        "/",
        response_model=RootResponse,
        summary="Root endpoint",
        responses={
            200: {"description": "Service information"},
            503: {"description": "Service not healthy"},
        },
    )
    async def root(self):
        """Root endpoint with basic info."""
        return RootResponse(
            service="Wine Quality Classifier API",
            version="1.0.0",
            status=self.status.value,
            docs="/docs",
            health="/health",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        responses={
            200: {"description": "Service is healthy"},
            503: {"description": "Service is not ready or unhealthy"},
        },
    )
    async def health(self):
        """Health check endpoint."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        response = HealthResponse(
            status=self.status,
            model_loaded=self.model is not None,
            model_uri=self.model_info.model_uri if self.model_info else None,
            uptime_seconds=int(uptime),
        )

        # Return 503 if not healthy
        if self.status != APIStatus.HEALTHY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.model_dump(),
            )

        return response

    @app.get(
        "/info",
        response_model=ModelInfo,
        summary="Model Information",
        responses={
            200: {"description": "Model information"},
            503: {"description": "Model not loaded", "model": ErrorResponse},
        },
    )
    async def info(self):
        """Get detailed model information."""
        if self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please configure the deployment with a model_uri.",
            )
        return self.model_info

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict Wine Quality",
        responses={
            200: {"description": "Successful prediction"},
            400: {"description": "Invalid input", "model": ErrorResponse},
            503: {"description": "Model not loaded", "model": ErrorResponse},
            500: {"description": "Internal server error", "model": ErrorResponse},
        },
    )
    async def predict(self, request: PredictionRequest):
        """
        Predict wine quality from physicochemical properties.

        Input should be a list of feature vectors with 12 values each:
        [fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
         free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type]

        wine_type: 0 = red, 1 = white

        The model returns quality scores between 3 and 9, along with confidence scores.
        """
        # Check if model is loaded
        if self.model is None or self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Configure the deployment with a model_uri.",
            )

        start_time = datetime.now(timezone.utc)

        try:
            # Convert to numpy array (validation already done by Pydantic)
            features = np.array(request.features, dtype=np.float64)

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

            # Calculate processing time
            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return PredictionResponse(
                predictions=predictions,
                model_uri=self.model_info.model_uri,
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_time,
            )

        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input: {str(e)}",
            )
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )


class AppBuilderArgs(BaseModel):
    """Arguments for building the Ray Serve application."""

    model_uri: str | None = Field(
        None,
        description="MLflow model URI to load (e.g., models:/wine-classifier/1 or runs:/run_id/model)",
    )


def app_builder(args: AppBuilderArgs) -> Application:
    """Helper function to build the deployment with optional model URI.

    This can be useful for testing out deployments locally like:
    serve run src.serve:app_builder model_uri="models:/ci.wine-classifier/7"

    Or with hot reload during development:
    serve run src.serve:app_builder model_uri="models:/ci.wine-classifier/7" --reload
    """
    return WineClassifier.bind(model_uri=args.model_uri)
