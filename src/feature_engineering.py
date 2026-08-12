import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataset with useful derived housing features."""

    features_df = df.copy()

    # Total indoor square footage.
    features_df["TotalSF"] = (
        features_df["TotalBsmtSF"]
        + features_df["1stFlrSF"]
        + features_df["2ndFlrSF"]
    )

    # Total bathrooms, including basement bathrooms.
    features_df["TotalBath"] = (
        features_df["FullBath"]
        + 0.5 * features_df["HalfBath"]
        + features_df["BsmtFullBath"]
        + 0.5 * features_df["BsmtHalfBath"]
    )

    # Total usable outdoor area.
    features_df["TotalPorchSF"] = (
        features_df["OpenPorchSF"]
        + features_df["EnclosedPorch"]
        + features_df["3SsnPorch"]
        + features_df["ScreenPorch"]
        + features_df["WoodDeckSF"]
    )

    # House age and time since its last remodel at the date of sale.
    features_df["HouseAge"] = (
        features_df["YrSold"] - features_df["YearBuilt"]
    )

    features_df["RemodAge"] = (
        features_df["YrSold"] - features_df["YearRemodAdd"]
    )

    # Helpful yes/no amenity features.
    features_df["IsRemodeled"] = (
        features_df["YearRemodAdd"] > features_df["YearBuilt"]
    ).astype(int)

    features_df["HasGarage"] = (
        features_df["GarageType"] != "None"
    ).astype(int)

    features_df["HasBasement"] = (
        features_df["TotalBsmtSF"] > 0
    ).astype(int)

    features_df["HasFireplace"] = (
        features_df["Fireplaces"] > 0
    ).astype(int)

    return features_df