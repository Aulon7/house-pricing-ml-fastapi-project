import pandas as pd
from src.data_cleaning import clean_data


def test_clean_data_removes_missing_values(sample_houses):
    cleaned_df = clean_data(sample_houses)

    assert cleaned_df.isna().sum().sum() == 0


def test_clean_data_uses_none_for_absent_amenities(sample_houses):
    cleaned_df = clean_data(sample_houses)

    assert cleaned_df.loc[0, "Alley"] == "None"
    assert cleaned_df.loc[0, "GarageType"] == "None"


def test_clean_data_does_not_change_original_dataframe(sample_houses):
    clean_data(sample_houses)

    assert pd.isna(sample_houses.loc[0, "Alley"])
    assert pd.isna(sample_houses.loc[0, "LotFrontage"])

def test_clean_data_fills_numeric_values_with_median(sample_houses):
    cleaned_df = clean_data(sample_houses)

    # Median of existing LotFrontage values: 80 and 70.
    assert cleaned_df.loc[0, "LotFrontage"] == 75
    assert cleaned_df.loc[3, "LotFrontage"] == 75


def test_clean_data_fills_categorical_values_with_mode(sample_houses):
    cleaned_df = clean_data(sample_houses)

    # SBrkr appears twice and is the mode.
    assert cleaned_df.loc[0, "Electrical"] == "SBrkr"


def test_clean_data_handles_completely_empty_columns():
    raw_df = pd.DataFrame(
        {
            "EmptyNumeric": [float("nan"), float("nan")],
            "EmptyCategory": [None, None],
            "Alley": [None, None],
        }
    )

    cleaned_df = clean_data(raw_df)

    assert cleaned_df["EmptyNumeric"].tolist() == [0, 0]
    assert cleaned_df["EmptyCategory"].tolist() == [
        "Unknown",
        "Unknown",
    ]
    assert cleaned_df["Alley"].tolist() == ["None", "None"]