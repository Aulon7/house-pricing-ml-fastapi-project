"""Contract tests for the P3 FastAPI model-serving layer."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import DatabricksDeltaLogger, create_app
from src.config import CONFIG


@pytest.fixture(scope="module")
def valid_payload() -> dict[str, object]:
    """Build a complete request from P2's exported input contract."""

    raw_columns = joblib.load(CONFIG.features_path)["raw_columns"]
    house = pd.read_csv(CONFIG.test_csv, nrows=1)[raw_columns]

    # Pandas/NumPy scalar values are converted to JSON-native Python values.
    return json.loads(house.to_json(orient="records"))[0]


@pytest.fixture
def client():
    """Start a fresh application so artifact loading follows its lifecycle."""

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_is_ready_when_model_artifacts_load(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_predict_accepts_the_exported_raw_feature_contract(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"prediction"}
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])


def test_predict_rejects_a_missing_required_feature(client, valid_payload):
    incomplete_payload = valid_payload.copy()
    incomplete_payload.pop("OverallQual")

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422


@pytest.mark.parametrize("forbidden_field", ["Id", "SalePrice", "UnexpectedField"])
def test_predict_rejects_unknown_or_excluded_features(
    client, valid_payload, forbidden_field
):
    invalid_payload = {**valid_payload, forbidden_field: 1}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_missing_artifacts_make_the_service_unavailable(tmp_path, valid_payload):
    unavailable_config = replace(CONFIG, models_dir=tmp_path)

    with TestClient(create_app(config=unavailable_config)) as unavailable_client:
        health = unavailable_client.get("/health")
        prediction = unavailable_client.post("/predict", json=valid_payload)

    assert health.status_code == 503
    assert health.json() == {"status": "unavailable"}
    assert prediction.status_code == 503


def test_predict_logs_the_required_databricks_record(valid_payload):
    logged_records: list[dict[str, object]] = []

    def record_prediction(**record: object) -> None:
        logged_records.append(record)

    with TestClient(create_app(prediction_logger=record_prediction)) as logging_client:
        response = logging_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert len(logged_records) == 1

    record = logged_records[0]
    assert record["input_features"] == valid_payload
    assert record["prediction"] == response.json()["prediction"]
    assert isinstance(record["model_version"], str)
    assert record["model_version"]
    assert isinstance(record["timestamp"], datetime)
    assert record["timestamp"].tzinfo is UTC


def test_logger_failure_does_not_prevent_a_prediction(valid_payload):
    def failing_logger(**record: object) -> None:
        raise RuntimeError("Databricks is unavailable")

    with TestClient(create_app(prediction_logger=failing_logger)) as logging_client:
        response = logging_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def test_databricks_logger_submits_a_parameterized_delta_insert(monkeypatch):
    captured: dict[str, object] = {}

    class SuccessfulResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"status": {"state": "SUCCEEDED"}}

    def fake_post(*args, **kwargs):
        captured["url"] = args[0]
        captured.update(kwargs)
        return SuccessfulResponse()

    monkeypatch.setattr("api.main.requests.post", fake_post)

    logger = DatabricksDeltaLogger(
        host="https://example.cloud.databricks.com",
        token="test-token",
        warehouse_id="warehouse-1",
        table="workspace.default.iowa_predictions",
    )
    logger(
        input_features={"OverallQual": 8},
        prediction=365000.0,
        timestamp=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
        model_version="abc123",
    )

    assert captured["url"] == "https://example.cloud.databricks.com/api/2.0/sql/statements"
    assert captured["timeout"] == 10.0
    assert captured["headers"] == {"Authorization": "Bearer test-token"}

    request = captured["json"]
    assert request["warehouse_id"] == "warehouse-1"
    assert request["statement"] == (
        "INSERT INTO workspace.default.iowa_predictions "
        "(input_features, prediction, prediction_timestamp, model_version) "
        "VALUES (:input_features, :prediction, :prediction_timestamp, :model_version)"
    )
    assert request["parameters"] == [
        {
            "name": "input_features",
            "value": '{"OverallQual":8}',
            "type": "STRING",
        },
        {"name": "prediction", "value": "365000.0", "type": "DOUBLE"},
        {
            "name": "prediction_timestamp",
            "value": "2026-08-12T12:00:00+00:00",
            "type": "TIMESTAMP",
        },
        {"name": "model_version", "value": "abc123", "type": "STRING"},
    ]
