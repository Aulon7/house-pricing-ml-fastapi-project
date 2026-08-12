import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.data_cleaning import clean_data
from src.feature_engineering import create_features


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, engineer, and select model input features."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    prepared_df = clean_data(df)

    # FastAPI inputs can contain Python None values. Normalize them to
    # np.nan so that scikit-learn's imputers recognize them as missing.
    # The string "None", used for absent physical features, is unchanged.
    prepared_df = prepared_df.where(prepared_df.notna(), np.nan)

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
    """Learn and apply preprocessing steps for the Ames Housing dataset and reuse them on new data.
    - Numerical features are filled with the training-column median,
    - Categorical features are filled with the training-column mode and then one-hot encoded.
    """

    def __init__(self):
        self.transformer: ColumnTransformer | None = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "HousePricingPreprocessor":
        """Learn training-data categories for one-hot encoding from training data."""

        prepared_df = prepare_dataframe(df)

        # Save training feature names and their respective order.

        self.feature_columns = prepared_df.columns.tolist()

        self.numeric_columns = prepared_df.select_dtypes(
            include="number"
        ).columns.tolist()

        self.categorical_columns = prepared_df.select_dtypes(
            exclude="number"
        ).columns.tolist()


        numerical_pipeline = Pipeline (
            steps=[
                (
                    "imputer", SimpleImputer(strategy="median")
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer", SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
                )
            ]
        )
        self.transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numerical_pipeline,
                    self.numeric_columns),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.transformer.fit(prepared_df)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using rules learned from training data. Raises an error if fit() has not been called first."""

        if self.transformer is None:
            raise RuntimeError("Run fit() before transform().")

        prepared_df = prepare_dataframe(df)

        missing_columns = [
            column for column in self.feature_columns
            if column not in prepared_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}."
            )

        prepared_df = prepared_df[self.feature_columns]

        transformed_data = np.asarray(
            self.transformer.transform(prepared_df)
        )

        return pd.DataFrame(
            transformed_data,
            columns=self.transformer.get_feature_names_out(),
            index=prepared_df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and transform it."""

        self.fit(df)

        return self.transform(df)
