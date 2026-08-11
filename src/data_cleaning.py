import pandas as pd

"""According to data_description.txt, an empty value in these columns means the
 corresponding physical feature is absent, not that its value is unknown. """

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



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned copy of Ames Housing dataset

    Rules:
    - Absence-related categorical values become "None".
    - Numeric missing values use the column median.
    - Other categorical missing values use the column mode.
    """

    # Copy the original dataframe to avoid modifying it.
    cleaned_df = df.copy()

    # Missing means the physical feature does not exist.
    for column in ABSENCE_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].fillna("None")

    # Fill every numeric missing value with the column median.
    numeric_columns = cleaned_df.select_dtypes(include="number").columns

    for column in numeric_columns:
        if cleaned_df[column].isna().any():
            non_missing_values = cleaned_df[column].dropna()
            if non_missing_values.empty:
                median = 0
            else:
                median = non_missing_values.median()
            cleaned_df[column] = cleaned_df[column].fillna(median)

    # Fill other categorical missing values with the most common value (mode).
    categorical_columns = cleaned_df.select_dtypes(exclude="number").columns

    for column in categorical_columns:
        if cleaned_df[column].isna().any():
            mode = cleaned_df[column].mode(dropna=True)

            cleaned_df[column] = cleaned_df[column].fillna(
                mode.iloc[0] if not mode.empty else "Unknown"
            )

    return cleaned_df
