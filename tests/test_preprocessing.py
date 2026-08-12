import numpy as np
import pandas as pd
import pytest

from src.preprocessing import HousePricingPreprocessor


def test_preprocessor_returns_model_ready_data(sample_houses):
    X_train = sample_houses.iloc[:2].drop(columns="SalePrice")
    X_valid = sample_houses.iloc[2:].drop(columns="SalePrice")

    preprocessor = HousePricingPreprocessor()

    train_ready = preprocessor.fit_transform(X_train)
    valid_ready = preprocessor.transform(X_valid)

    assert train_ready.isna().sum().sum() == 0
    assert valid_ready.isna().sum().sum() == 0

    assert list(train_ready.columns) == list(valid_ready.columns)

    assert "Id" not in train_ready.columns
    assert "SalePrice" not in train_ready.columns

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in train_ready.dtypes
    )


def test_preprocessor_requires_fit_before_transform(sample_houses):
    preprocessor = HousePricingPreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_houses)

def test_preprocessor_preserves_row_count(sample_houses):
    preprocessor = HousePricingPreprocessor()

    model_ready_df = preprocessor.fit_transform(sample_houses)

    assert len(model_ready_df) == len(sample_houses)

def test_preprocessor_returns_same_output_for_training_data(sample_houses):
    X_train = sample_houses.iloc[:2].drop(columns="SalePrice")

    preprocessor = HousePricingPreprocessor()

    first_result = preprocessor.fit_transform(X_train)
    second_result = preprocessor.transform(X_train)

    pd.testing.assert_frame_equal(first_result, second_result)


def test_preprocessor_uses_training_median(sample_houses):
    X_train = sample_houses.iloc[[1, 2]].drop(columns="SalePrice")
    X_new = sample_houses.iloc[[3]].drop(columns="SalePrice").copy()

    # The training values are 80 and 70, so their median is 75.
    X_new["LotFrontage"] = np.nan

    preprocessor = HousePricingPreprocessor()
    preprocessor.fit(X_train)

    result = preprocessor.transform(X_new)

    assert result["LotFrontage"].iloc[0] == 75


def test_preprocessor_handles_single_row_with_none_category(sample_houses):
    X_train = sample_houses.iloc[1:].drop(columns="SalePrice")
    X_new = sample_houses.iloc[[0]].drop(columns="SalePrice").copy()

    # FastAPI can create a one-row DataFrame containing Python None.
    X_new["Electrical"] = None

    preprocessor = HousePricingPreprocessor()
    preprocessor.fit(X_train)

    result = preprocessor.transform(X_new)

    assert len(result) == 1
    assert result.isna().sum().sum() == 0
