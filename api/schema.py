"""Pydantic schemas for the P3 serving API.

The raw input columns belong to P2's exported feature contract, not to this
module. ``build_prediction_request_model`` therefore creates the request
schema from the artifact at application construction time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


# Raw Ames data contains categorical strings, numeric measurements and a few
# yes/no values. Existing P1 preprocessing is responsible for interpreting and
# imputing them; P3 only validates that they are JSON scalar values.
FeatureValue = str | int | float | bool | None


class PredictionResponse(BaseModel):
    """Successful prediction response, in US dollars."""

    prediction: float


class HealthResponse(BaseModel):
    """Public readiness response without internal error details."""

    status: Literal["ready", "unavailable"]


def build_prediction_request_model(raw_columns: list[str]) -> type[BaseModel]:
    """Create a strict flat request schema from P2's raw feature contract.

    Every artifact column is required in the JSON object, but each value may
    be ``null`` so P1's fitted imputer can handle a genuinely missing value.
    Extra fields—including ``Id`` and ``SalePrice``—are rejected.
    """

    if not raw_columns:
        raise ValueError("the raw feature contract is empty")

    if len(raw_columns) != len(set(raw_columns)):
        raise ValueError("the raw feature contract contains duplicate columns")

    if any(not isinstance(column, str) or not column for column in raw_columns):
        raise ValueError("the raw feature contract contains an invalid column name")

    fields = {
        column: (
            FeatureValue,
            Field(..., description=f"Raw model feature: {column}"),
        )
        for column in raw_columns
    }

    return create_model(
        "PredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
