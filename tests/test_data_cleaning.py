import pandas as pd

from src.data_cleaning import clean_data


def test_clean_data_fills_absent_categorical_features(sample_houses):
    cleaned_df = clean_data(sample_houses)

    assert cleaned_df.loc[0, "Alley"] == "None"
    assert cleaned_df.loc[3, "GarageType"] == "None"


def test_clean_data_zero_fills_absent_numeric_features():
    raw_df = pd.DataFrame(
        {
            "TotalBsmtSF": [None],
            "BsmtFinSF1": [None],
            "BsmtFinSF2": [None],
            "BsmtUnfSF": [None],
            "BsmtFullBath": [None],
            "BsmtHalfBath": [None],
            "MasVnrArea": [None],
        }
    )

    cleaned_df = clean_data(raw_df)

    assert cleaned_df.loc[0, "TotalBsmtSF"] == 0
    assert cleaned_df.loc[0, "BsmtFinSF1"] == 0
    assert cleaned_df.loc[0, "BsmtFinSF2"] == 0
    assert cleaned_df.loc[0, "BsmtUnfSF"] == 0
    assert cleaned_df.loc[0, "BsmtFullBath"] == 0
    assert cleaned_df.loc[0, "BsmtHalfBath"] == 0
    assert cleaned_df.loc[0, "MasVnrArea"] == 0


def test_clean_data_leaves_general_missing_values_for_preprocessor(
    sample_houses,
):
    cleaned_df = clean_data(sample_houses)

    assert pd.isna(cleaned_df.loc[0, "LotFrontage"])
    assert pd.isna(cleaned_df.loc[0, "Electrical"])


def test_clean_data_does_not_change_original_dataframe(sample_houses):
    clean_data(sample_houses)

    assert pd.isna(sample_houses.loc[0, "Alley"])
    assert pd.isna(sample_houses.loc[0, "LotFrontage"])
