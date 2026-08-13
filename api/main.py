"""FastAPI application for serving the exported Ames Housing model."""

from __future__ import annotations

import logging
import math
import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError

from api.schema import (
    FeatureValue,
    HealthResponse,
    PredictionResponse,
    build_prediction_request_model,
)
from src.config import CONFIG, Config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingArtifacts:
    """The P2 artifacts needed by the P3 serving layer."""

    model: object
    raw_columns: list[str]
    model_version: str


class PredictionLogger(Protocol):
    """The minimal P3 interface for persisting a completed prediction."""

    def __call__(
        self,
        *,
        input_features: dict[str, FeatureValue],
        prediction: float,
        timestamp: datetime,
        model_version: str,
    ) -> None: ...


@dataclass(frozen=True)
class DatabricksDeltaLogger:
    """Write P3 prediction records to a configured Databricks Delta table."""

    host: str
    token: str
    warehouse_id: str
    table: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> "DatabricksDeltaLogger | None":
        """Create a logger only when all required Databricks settings exist."""

        settings = {
            "host": os.getenv("DATABRICKS_HOST", "").rstrip("/"),
            "token": os.getenv("DATABRICKS_TOKEN", ""),
            "warehouse_id": os.getenv("DATABRICKS_WAREHOUSE_ID", ""),
            "table": os.getenv("DATABRICKS_PREDICTIONS_TABLE", ""),
        }

        if not all(settings.values()):
            return None

        return cls(**settings)

    def __call__(
        self,
        *,
        input_features: dict[str, FeatureValue],
        prediction: float,
        timestamp: datetime,
        model_version: str,
    ) -> None:
        """Insert one prediction row using the Statement Execution API."""

        statement = (
            f"INSERT INTO {self.table} "
            "(input_features, prediction, prediction_timestamp, model_version) "
            "VALUES (:input_features, :prediction, :prediction_timestamp, :model_version)"
        )
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": statement,
            "parameters": [
                {
                    "name": "input_features",
                    "value": json.dumps(input_features, separators=(",", ":")),
                    "type": "STRING",
                },
                {"name": "prediction", "value": str(prediction), "type": "DOUBLE"},
                {
                    "name": "prediction_timestamp",
                    "value": timestamp.isoformat(),
                    "type": "TIMESTAMP",
                },
                {"name": "model_version", "value": model_version, "type": "STRING"},
            ],
            "wait_timeout": "10s",
            "on_wait_timeout": "CANCEL",
        }
        response = requests.post(
            f"{self.host}/api/2.0/sql/statements",
            headers={"Authorization": f"Bearer {self.token}"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        execution = response.json().get("status", {})
        if execution.get("state") != "SUCCEEDED":
            raise RuntimeError("Databricks did not complete the prediction insert")


def load_serving_artifacts(config: Config) -> ServingArtifacts:
    """Load and validate P2's exported model and raw feature contract."""

    model = joblib.load(config.model_path)
    contract = joblib.load(config.features_path)

    if not isinstance(contract, dict) or "raw_columns" not in contract:
        raise ValueError("features artifact does not contain a raw_columns contract")

    raw_columns = contract["raw_columns"]

    if not isinstance(raw_columns, list):
        raise ValueError("raw_columns contract must be a list")

    if not callable(getattr(model, "predict", None)):
        raise ValueError("model artifact does not provide predict")

    # The schema factory performs the remaining column validation, and keeping
    # it here ensures a corrupt contract makes the whole service unavailable.
    build_prediction_request_model(raw_columns)

    model_version = hashlib.sha256(config.model_path.read_bytes()).hexdigest()

    return ServingArtifacts(
        model=model,
        raw_columns=raw_columns,
        model_version=model_version,
    )


def create_app(
    config: Config = CONFIG,
    prediction_logger: PredictionLogger | None = None,
) -> FastAPI:
    """Create the P3 API application with its artifact-loading lifecycle."""

    # Local credentials belong in the git-ignored .env file. Existing system
    # environment variables keep priority, which is useful in deployments.
    load_dotenv(config.project_root / ".env")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            artifacts = load_serving_artifacts(config)
            application.state.model = artifacts.model
            application.state.raw_columns = artifacts.raw_columns
            application.state.model_version = artifacts.model_version
            application.state.request_model = build_prediction_request_model(
                artifacts.raw_columns
            )
            application.state.prediction_logger = (
                prediction_logger or DatabricksDeltaLogger.from_environment()
            )
            application.state.ready = True
            LOGGER.info("model-serving artifacts loaded")
            if application.state.prediction_logger is None:
                LOGGER.warning("Databricks prediction logging is not configured")
        except Exception:
            application.state.ready = False
            LOGGER.exception("model-serving artifacts could not be loaded")

        yield

    app = FastAPI(
        title="Ames Housing Model API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.ready = False
    app.state.model = None
    app.state.raw_columns = []
    app.state.model_version = None
    app.state.request_model = None
    app.state.prediction_logger = None

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response) -> HealthResponse:
        """Report whether P2's model artifacts are ready to serve."""

        if not app.state.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(status="unavailable")

        return HealthResponse(status="ready")

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: dict[str, FeatureValue], request: Request
    ) -> PredictionResponse:
        """Validate one complete raw-house payload and return its sale price."""

        application = request.app

        if not application.state.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is unavailable.",
            )

        try:
            validated = application.state.request_model(**payload)
        except Exception as error:
            # Convert the artifact-derived Pydantic validation result into
            # FastAPI's standard 422 response format.
            if hasattr(error, "errors"):
                raise RequestValidationError(error.errors()) from error
            raise

        try:
            frame = pd.DataFrame(
                [validated.model_dump()], columns=application.state.raw_columns
            )
            predictions = application.state.model.predict(frame)
            prediction = float(predictions[0])

            if not math.isfinite(prediction):
                raise ValueError("model returned a non-finite prediction")
        except Exception as error:
            LOGGER.exception("prediction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction could not be generated.",
            ) from error

        if application.state.prediction_logger is not None:
            try:
                application.state.prediction_logger(
                    input_features=validated.model_dump(),
                    prediction=prediction,
                    timestamp=datetime.now(UTC),
                    model_version=application.state.model_version,
                )
            except Exception:
                LOGGER.exception("Databricks prediction logging failed")

        return PredictionResponse(prediction=prediction)

    return app


app = create_app()
