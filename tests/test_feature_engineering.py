from src.data_cleaning import clean_data
from src.feature_engineering import create_features


def test_create_features_adds_expected_columns(sample_houses):
    cleaned_df = clean_data(sample_houses)
    features_df = create_features(cleaned_df)

    expected_columns = [
        "TotalSF",
        "TotalBath",
        "TotalPorchSF",
        "HouseAge",
        "RemodAge",
        "IsRemodeled",
        "HasGarage",
        "HasBasement",
        "HasFireplace",
    ]

    for column in expected_columns:
        assert column in features_df.columns


def test_create_features_calculates_correct_values(sample_houses):
    features_df = create_features(clean_data(sample_houses))

    assert features_df.loc[0, "TotalSF"] == 1900
    assert features_df.loc[0, "TotalBath"] == 3.5
    assert features_df.loc[0, "TotalPorchSF"] == 35
    assert features_df.loc[0, "HouseAge"] == 10
    assert features_df.loc[0, "RemodAge"] == 5
    assert features_df.loc[0, "IsRemodeled"] == 1
    assert features_df.loc[0, "HasGarage"] == 1
    assert features_df.loc[0, "HasBasement"] == 1
    assert features_df.loc[0, "HasFireplace"] == 1

def test_create_features_does_not_change_input_dataframe(sample_houses):
    cleaned_df = clean_data(sample_houses)

    create_features(cleaned_df)

    assert "TotalSF" not in cleaned_df.columns
    assert "TotalBath" not in cleaned_df.columns


def test_binary_features_are_zero_when_amenity_is_missing(sample_houses):
    cleaned_df = clean_data(sample_houses)

    cleaned_df.loc[0, "GarageArea"] = 0
    cleaned_df.loc[0, "TotalBsmtSF"] = 0
    cleaned_df.loc[0, "Fireplaces"] = 0

    features_df = create_features(cleaned_df)

    assert features_df.loc[0, "HasGarage"] == 0
    assert features_df.loc[0, "HasBasement"] == 0
    assert features_df.loc[0, "HasFireplace"] == 0


def test_is_remodeled_is_zero_when_years_are_equal(sample_houses):
    cleaned_df = clean_data(sample_houses)

    cleaned_df.loc[0, "YearRemodAdd"] = cleaned_df.loc[0, "YearBuilt"]

    features_df = create_features(cleaned_df)

    assert features_df.loc[0, "IsRemodeled"] == 0