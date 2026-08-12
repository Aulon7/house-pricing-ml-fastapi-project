import pandas as pd

# Categorical features that are missing because the physical feature is absent should be filled with "None".
ABSENCE_COLUMNS = ["Alley",
                    "BsmtQual",
                    "BsmtCond",
                    "BsmtExposure",
                    "BsmtFinType1",
                    "BsmtFinType2",
                    "FireplaceQu",
                    "GarageType",
                    "GarageFinish",
                    "GarageQual",
                    "GarageCond",
                    "PoolQC",
                    "Fence",
                    "MiscFeature",
                    "MasVnrType"]

# Numerical features that are missing because the physical feature is absent should be filled with 0.
ZERO_FILL_COLUMNS = ["BsmtFinSF1",
                    "BsmtFinSF2",
                    "BsmtUnfSF",
                    "TotalBsmtSF",
                    "BsmtFullBath",
                    "BsmtHalfBath",
                    "MasVnrArea"]



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned copy of Ames Housing dataset.
    General missing values are handled in HousePricingPreprocessor where values are learned from the training data.
    """

    # Copy the original dataframe to avoid modifying it.
    cleaned_df = df.copy()

    # Missing means the categorical feature does not physically exist, fill with "None".
    for column in ABSENCE_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].fillna("None")


    # Missing means the numeric feature is absent, fill with 0.
    for column in ZERO_FILL_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            ).fillna(0)

    return cleaned_df
