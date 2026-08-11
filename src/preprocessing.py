import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.data_cleaning import clean_data
from src.feature_engineering import create_features


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, engineer, and select model input features."""

    prepared_df = clean_data(df)
    prepared_df = create_features(prepared_df)

    # SalePrice is the target; Id is only a row identifier.
    prepared_df = prepared_df.drop(
        columns=["SalePrice", "Id"],
        errors="ignore",
    )

    # MSSubClass is a category code, not a numerical measurement.
    if "MSSubClass" in prepared_df.columns:
        prepared_df["MSSubClass"] = prepared_df["MSSubClass"].astype(str)

    return prepared_df


class HousePricingPreprocessor:
    """One-hot encode categorical Ames Housing features."""

    def __init__(self):
        self.transformer: ColumnTransformer | None = None

    def fit(self, df: pd.DataFrame) -> "HousePricingPreprocessor":
        """Learn training-data categories for one-hot encoding."""

        prepared_df = prepare_dataframe(df)

        numeric_columns = prepared_df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = prepared_df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        self.transformer = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", numeric_columns),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=int,
                    ),
                    categorical_columns,
                ),
            ],
            verbose_feature_names_out=False,
        )

        self.transformer.fit(prepared_df)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply training encoding rules to new data."""

        if self.transformer is None:
            raise RuntimeError("Run fit() before transform().")

        prepared_df = prepare_dataframe(df)

        transformed_data = np.asarray(
            self.transformer.transform(prepared_df)
        )

        return pd.DataFrame(
            transformed_data,
            columns=self.transformer.get_feature_names_out(),
            index=prepared_df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform training data."""

        self.fit(df)

        return self.transform(df)