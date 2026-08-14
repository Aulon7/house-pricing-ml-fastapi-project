"""Persist FastAPI prediction records to a Databricks Delta table."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import requests

FeatureValue = str | int | float | bool | None


class PredictionLogger(Protocol):
    """Minimal interface for persisting a completed prediction."""

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
    """Write prediction records to a configured Databricks Delta table."""

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
