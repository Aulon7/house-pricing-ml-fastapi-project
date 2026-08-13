"""The candidate models and how they are assembled.

Each candidate is a full estimator, not a bare regressor: cleaning, feature
engineering and one-hot encoding are steps inside it, and the log target is
handled internally. That has two consequences worth stating.

Cross-validation refits the whole thing per fold, so the encoder never sees
the fold it is scored on. And the exported model takes a raw dataframe and
returns sale prices in dollars, so serving needs no knowledge of how the
target was transformed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from src.config import CONFIG
from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.preprocessing import HousePricingPreprocessor


class PreprocessorStep(BaseEstimator, TransformerMixin):
    """Adapts HousePricingPreprocessor to the scikit-learn transformer API.

    The preprocessor from P1 takes fit(df) while scikit-learn calls
    fit(X, y). This wrapper bridges the two without touching that module, and
    building the preprocessor inside fit keeps the step cloneable, which is
    what cross-validation relies on.

    Set engineer_features=False to score the same models without the derived
    features. The engineered columns are dropped after preprocessing rather
    than never created, which comes to the same thing: they are numeric
    additions, so removing them afterwards leaves every other column, every
    category and every imputed value exactly as it was.
    """

    def __init__(self, engineer_features: bool = True):
        self.engineer_features = engineer_features

    def _engineered_columns(self, X: pd.DataFrame) -> list[str]:
        """The columns create_features adds, asked of the module itself.

        Derived rather than hard-coded so the list stays correct when P1
        adds or renames a feature.
        """

        cleaned = clean_data(X)

        return [
            column
            for column in create_features(cleaned).columns
            if column not in cleaned.columns
        ]

    def fit(self, X: pd.DataFrame, y=None) -> "PreprocessorStep":
        self.preprocessor_ = HousePricingPreprocessor()
        self.preprocessor_.fit(X)

        self.dropped_columns_ = (
            [] if self.engineer_features else self._engineered_columns(X)
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = self.preprocessor_.transform(X)

        # Read defensively: a model.pkl exported before the ablation existed
        # unpickles without this attribute, and serving should keep working
        # rather than fail on an artifact that predates the feature.
        dropped = getattr(self, "dropped_columns_", ())

        if dropped:
            encoded = encoded.drop(columns=list(dropped), errors="ignore")

        # The encoded names are read off the output rather than off the
        # preprocessor's internal ColumnTransformer, which is private to P1
        # and is exactly what gets restructured when the cleaning schema is
        # frozen. Recording them on the first transform costs nothing, since
        # a Pipeline transforms the training data on the way through anyway.
        #
        # Recorded once and never rewritten: after export this same object
        # serves predictions, and a transform that mutates fitted state is
        # not safe to share across concurrent requests.
        if not hasattr(self, "feature_names_"):
            self.feature_names_ = list(encoded.columns)

        return encoded

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise AttributeError(
                "feature names are recorded on the first transform; "
                "fit the surrounding pipeline before asking for them"
            )

        return np.asarray(self.feature_names_, dtype=object)


def _xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=CONFIG.random_seed,
        n_jobs=-1,
    )


def _lightgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        # 1460 rows overfit quickly, so the trees stay deliberately small.
        num_leaves=16,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=CONFIG.random_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=CONFIG.random_seed,
        n_jobs=-1,
    )


def _mlp() -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=800,
        # 1460 rows against 325 encoded columns overfit almost immediately,
        # so training stops as soon as the held-out loss stalls.
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=CONFIG.random_seed,
    )


@dataclass(frozen=True)
class ModelSpec:
    """How to build a candidate and what it needs around it."""

    builder: Callable[[], RegressorMixin]

    #: Trees split on raw values; a neural network needs comparable scales.
    #: Scaling uses MinMax rather than standardisation on purpose. Most of
    #: the 325 encoded columns are one-hot dummies, and standardising a
    #: category that appears once in 1460 rows sends that column to 34
    #: standard deviations, which is enough to blow the first layer up:
    #: measured rmse_log 0.88 standardised against 0.14 with MinMax.
    needs_scaling: bool = False


#: The candidates compared during training. Adding one is a single entry.
MODEL_SPECS: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(_xgboost),
    "lightgbm": ModelSpec(_lightgbm),
    "random_forest": ModelSpec(_random_forest),
    "mlp": ModelSpec(_mlp, needs_scaling=True),
}


def available_models() -> list[str]:
    """Return the candidate names, in the order they are compared."""

    return list(MODEL_SPECS)


def build_model(
    name: str, engineer_features: bool = True
) -> TransformedTargetRegressor:
    """Assemble one candidate: preprocessing, regressor and target transform.

    The returned estimator is fitted on a raw dataframe and predicts sale
    prices in dollars. Pass engineer_features=False for the ablation run.
    """

    if name not in MODEL_SPECS:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        )

    spec = MODEL_SPECS[name]
    steps: list[tuple[str, object]] = [
        ("preprocessor", PreprocessorStep(engineer_features=engineer_features))
    ]

    if spec.needs_scaling:
        steps.append(("scaler", MinMaxScaler()))

    steps.append(("regressor", spec.builder()))

    pipeline = Pipeline(steps=steps)

    # Sale prices are right-skewed; log1p makes the errors comparable across
    # price levels, and expm1 puts predictions back into dollars.
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def feature_names(model: TransformedTargetRegressor) -> list[str]:
    """Return the encoded column names a fitted model was trained on."""

    step = model.regressor_.named_steps["preprocessor"]

    return list(step.feature_names_)
