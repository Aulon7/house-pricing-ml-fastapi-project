"""Streamlit dashboard for Ames Housing price predictions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# "streamlit run" puts this file's own directory on sys.path rather than the
# project root, so the src package is invisible unless we add it ourselves.
# This has to happen before the first src import, hence the placement.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402

PRIMARY_FIELDS = {
    "OverallQual": "Overall quality (1–10)",
    "GrLivArea": "Above-grade living area (sq ft)",
    "YearBuilt": "Year built",
    "GarageCars": "Garage capacity (cars)",
    "TotRmsAbvGrd": "Total rooms above grade",
}

BOOLEAN_FIELDS = {"CentralAir", "PavedDrive"}


def project_root() -> Path:
    return CONFIG.project_root


def load_environment() -> None:
    load_dotenv(project_root() / ".env")


def api_base_url() -> str:
    return os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data
def load_defaults() -> dict[str, object]:
    defaults_path = CONFIG.models_dir / "defaults.json"

    with defaults_path.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_raw_columns() -> list[str]:
    contract = joblib.load(CONFIG.features_path)
    return contract["raw_columns"]


@st.cache_data
def load_training_reference() -> pd.DataFrame:
    return pd.read_csv(CONFIG.train_csv)


@st.cache_data
def load_categorical_options() -> dict[str, list[str]]:
    reference = load_training_reference()
    defaults = load_defaults()
    options: dict[str, list[str]] = {}

    for column, default_value in defaults.items():
        if column not in reference.columns:
            continue

        if isinstance(default_value, str):
            values = sorted(reference[column].dropna().astype(str).unique().tolist())
            if "None" not in values:
                values = ["None", *values]
            options[column] = values

    return options


def check_api_health() -> bool:
    try:
        response = requests.get(f"{api_base_url()}/health", timeout=5)
        response.raise_for_status()
        return response.json().get("status") == "ready"
    except requests.RequestException:
        return False


def row_to_payload(row: pd.Series, raw_columns: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {}

    for column in raw_columns:
        value = row[column]
        if pd.isna(value) or value == "":
            payload[column] = None
        elif isinstance(value, (pd.Timestamp,)):
            payload[column] = value.isoformat()
        else:
            payload[column] = value.item() if hasattr(value, "item") else value

    return payload


def predict_payload(payload: dict[str, object]) -> float:
    response = requests.post(
        f"{api_base_url()}/predict",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return float(response.json()["prediction"])


def render_primary_inputs(
    defaults: dict[str, object],
) -> dict[str, object]:
    values = defaults.copy()
    col_left, col_right = st.columns(2)

    with col_left:
        values["OverallQual"] = st.slider(
            PRIMARY_FIELDS["OverallQual"],
            min_value=1,
            max_value=10,
            value=int(defaults["OverallQual"]),
        )
        values["GrLivArea"] = st.number_input(
            PRIMARY_FIELDS["GrLivArea"],
            min_value=300,
            max_value=8000,
            value=int(defaults["GrLivArea"]),
            step=50,
        )
        values["YearBuilt"] = st.number_input(
            PRIMARY_FIELDS["YearBuilt"],
            min_value=1872,
            max_value=2010,
            value=int(defaults["YearBuilt"]),
            step=1,
        )

    with col_right:
        values["GarageCars"] = st.number_input(
            PRIMARY_FIELDS["GarageCars"],
            min_value=0,
            max_value=4,
            value=int(defaults["GarageCars"]),
            step=1,
        )
        values["TotRmsAbvGrd"] = st.number_input(
            PRIMARY_FIELDS["TotRmsAbvGrd"],
            min_value=2,
            max_value=15,
            value=int(defaults["TotRmsAbvGrd"]),
            step=1,
        )

    return values


def render_field_input(
    column: str,
    value: object,
    categorical_options: dict[str, list[str]],
) -> object:
    if column in BOOLEAN_FIELDS:
        choices = ["Y", "N"]
        current = str(value) if value is not None else "N"
        index = choices.index(current) if current in choices else 0
        return st.selectbox(column, choices, index=index)

    if column in categorical_options:
        choices = categorical_options[column]
        current = str(value) if value is not None else choices[0]
        index = choices.index(current) if current in choices else 0
        return st.selectbox(column, choices, index=index)

    if isinstance(value, float):
        return st.number_input(column, value=float(value))

    return st.number_input(column, value=int(value), step=1)


def render_advanced_inputs(
    defaults: dict[str, object],
    current_values: dict[str, object],
    categorical_options: dict[str, list[str]],
) -> dict[str, object]:
    advanced_values = current_values.copy()

    with st.expander("Advanced house details"):
        # current_values starts life as a copy of every default, so testing
        # membership in it would exclude every column and leave the expander
        # empty. The primary inputs are the only ones already rendered.
        remaining_columns = [
            column for column in defaults if column not in PRIMARY_FIELDS
        ]

        for index in range(0, len(remaining_columns), 2):
            columns = st.columns(2)
            for offset, container in enumerate(columns):
                column_index = index + offset
                if column_index >= len(remaining_columns):
                    continue

                column = remaining_columns[column_index]
                with container:
                    advanced_values[column] = render_field_input(
                        column,
                        defaults[column],
                        categorical_options,
                    )

    return advanced_values


def render_single_prediction_tab() -> None:
    defaults = load_defaults()
    raw_columns = load_raw_columns()
    categorical_options = load_categorical_options()

    st.subheader("Single house prediction")
    st.caption(
        "Adjust the main features below. Remaining values start from "
        "`models/defaults.json`, which P4 prepared as the baseline house profile."
    )

    current_values = render_primary_inputs(defaults)
    current_values = render_advanced_inputs(
        defaults,
        current_values,
        categorical_options,
    )

    for column, default_value in defaults.items():
        current_values.setdefault(column, default_value)

    payload = {column: current_values[column] for column in raw_columns}

    if st.button("Predict sale price", type="primary"):
        try:
            prediction = predict_payload(payload)
        except requests.HTTPError as error:
            st.error(f"Prediction request failed: {error.response.text}")
            return
        except requests.RequestException as error:
            st.error(f"Could not reach the API at {api_base_url()}: {error}")
            return

        st.success(f"Predicted sale price: ${prediction:,.0f}")


def render_batch_prediction_tab() -> None:
    raw_columns = load_raw_columns()

    st.subheader("Batch prediction")
    st.caption(
        "Upload a CSV with the same raw feature columns used by the model. "
        "Each row is sent to FastAPI and logged to Databricks when configured."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        return

    frame = pd.read_csv(uploaded_file)
    missing_columns = [column for column in raw_columns if column not in frame.columns]

    if missing_columns:
        st.error(
            "The uploaded file is missing required columns: "
            + ", ".join(missing_columns)
        )
        return

    st.dataframe(frame[raw_columns].head(), use_container_width=True)

    if st.button("Run batch prediction", type="primary"):
        predictions: list[float] = []
        progress = st.progress(0.0, text="Sending rows to FastAPI...")

        try:
            for index, (_, row) in enumerate(frame.iterrows(), start=1):
                predictions.append(
                    predict_payload(row_to_payload(row, raw_columns))
                )
                progress.progress(
                    index / len(frame),
                    text=f"Processed {index} of {len(frame)} houses",
                )
        except requests.HTTPError as error:
            st.error(f"Prediction request failed: {error.response.text}")
            return
        except requests.RequestException as error:
            st.error(f"Could not reach the API at {api_base_url()}: {error}")
            return

        result = frame.copy()
        result["PredictedSalePrice"] = predictions
        st.success("Batch prediction completed.")
        st.dataframe(result, use_container_width=True)

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )


def main() -> None:
    load_environment()

    st.set_page_config(
        page_title="Ames Housing Price Predictor",
        page_icon="🏡",
        layout="wide",
    )
    st.title("Ames Housing Price Predictor")
    st.write(
        "Predict house sale prices through the FastAPI service. "
        "Successful predictions can be stored in Databricks Delta Lake."
    )

    if check_api_health():
        st.success(f"API ready at {api_base_url()}")
    else:
        st.warning(
            f"The API at {api_base_url()} is unavailable. "
            "Start it with `uvicorn api.main:app --reload` before predicting."
        )

    single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

    with single_tab:
        render_single_prediction_tab()

    with batch_tab:
        render_batch_prediction_tab()


if __name__ == "__main__":
    main()
