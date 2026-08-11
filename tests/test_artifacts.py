"""Tests for the exported model and feature artifacts."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.train import (
    load_training_data,
    main,
    required_input_columns,
    save_artifacts,
    train_final_model,
)


@pytest.fixture(scope="module")
def small_dataset():
    features, target = load_training_data()

    return features.head(200), target.head(200)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, small_dataset):
    """Train once and export, then reuse across the assertions below."""

    features, target = small_dataset
    directory = tmp_path_factory.mktemp("models")

    model = train_final_model("random_forest", features, target)
    model_path, features_path = save_artifacts(
        model, required_input_columns(features), directory
    )

    return model, model_path, features_path


def test_required_columns_exclude_the_identifier(small_dataset):
    features, _ = small_dataset
    columns = required_input_columns(features)

    assert CONFIG.id_column not in columns
    assert CONFIG.target not in columns
    assert len(columns) == features.shape[1] - 1


def test_both_artifacts_are_written(exported):
    _, model_path, features_path = exported

    assert model_path.name == "model.pkl"
    assert features_path.name == "features.pkl"
    assert model_path.stat().st_size > 0
    assert features_path.stat().st_size > 0


def test_reloaded_model_predicts_identically(exported, small_dataset):
    model, model_path, _ = exported
    features, _ = small_dataset

    reloaded = joblib.load(model_path)

    np.testing.assert_allclose(reloaded.predict(features), model.predict(features))


def test_reloaded_model_takes_a_raw_dataframe_and_returns_dollars(
    exported, small_dataset
):
    _, model_path, _ = exported
    features, _ = small_dataset

    predictions = joblib.load(model_path).predict(features)

    assert predictions.min() > 10_000
    assert predictions.max() < 1_000_000


def test_features_artifact_describes_both_ends_of_the_pipeline(exported):
    _, _, features_path = exported

    saved = joblib.load(features_path)

    assert set(saved) == {"raw_columns", "encoded_columns"}
    assert CONFIG.id_column not in saved["raw_columns"]
    assert CONFIG.target not in saved["raw_columns"]

    # One-hot encoding turns 79 raw columns into a few hundred.
    assert len(saved["encoded_columns"]) > len(saved["raw_columns"])


def test_the_identifier_is_not_needed_at_prediction_time(exported, small_dataset):
    _, model_path, features_path = exported
    features, _ = small_dataset

    model = joblib.load(model_path)
    raw_columns = joblib.load(features_path)["raw_columns"]

    without_id = features[raw_columns]

    np.testing.assert_allclose(
        model.predict(without_id), model.predict(features)
    )


def test_a_complete_single_row_can_be_predicted(exported, small_dataset):
    """The Streamlit form sends one house at a time."""

    _, model_path, _ = exported
    features, _ = small_dataset

    prediction = joblib.load(model_path).predict(features.head(1))

    assert prediction.shape == (1,)
    assert 10_000 < prediction[0] < 1_000_000


def test_a_missing_categorical_sent_as_none_is_tolerated(exported, small_dataset):
    """A field the Streamlit form leaves blank arrives as None, not NaN.

    None keeps the column object-typed, so clean_data still sees a category
    and the unknown value is encoded as all zeros.
    """

    _, model_path, _ = exported
    features, _ = small_dataset

    row = features.head(1).copy()
    row["Electrical"] = None

    prediction = joblib.load(model_path).predict(row)

    assert 10_000 < prediction[0] < 1_000_000


@pytest.mark.xfail(
    reason=(
        "P1: a categorical column that is empty for every submitted row is "
        "read as NaN and typed float64, so clean_data treats it as numeric "
        "and fills it with 0, which the OneHotEncoder then rejects. This is "
        "the batch CSV upload path in P4. Flips to passing once fixed."
    ),
    strict=False,
)
def test_an_uploaded_csv_with_an_empty_column_can_be_predicted(
    exported, small_dataset, tmp_path
):
    _, model_path, _ = exported
    features, _ = small_dataset

    upload = features.head(3).copy()
    upload["Electrical"] = ""

    path = tmp_path / "upload.csv"
    upload.to_csv(path, index=False)

    predictions = joblib.load(model_path).predict(pd.read_csv(path))

    assert (predictions > 10_000).all()


def test_main_exports_both_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.train.CONFIG",
        type(CONFIG)(models_dir=tmp_path, reports_dir=tmp_path),
    )

    assert main(["--models", "random_forest", "--folds", "2"]) == 0

    assert (tmp_path / "model.pkl").is_file()
    assert (tmp_path / "features.pkl").is_file()
    assert (tmp_path / "model_comparison.csv").is_file()
