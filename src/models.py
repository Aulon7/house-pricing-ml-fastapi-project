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

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import CONFIG
from src.preprocessing import HousePricingPreprocessor


class PreprocessorStep(BaseEstimator, TransformerMixin):
    """Adapts HousePricingPreprocessor to the scikit-learn transformer API.

    The preprocessor from P1 takes fit(df) while scikit-learn calls
    fit(X, y). This wrapper bridges the two without touching that module, and
    building the preprocessor inside fit keeps the step cloneable, which is
    what cross-validation relies on.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "PreprocessorStep":
        self.preprocessor_ = HousePricingPreprocessor()
        self.preprocessor_.fit(X)

        self.feature_names_ = list(
            self.preprocessor_.transformer.get_feature_names_out()
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
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


#: The candidates compared during training. Adding one is a single entry.
MODEL_BUILDERS: dict[str, Callable[[], RegressorMixin]] = {
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "random_forest": _random_forest,
}


def available_models() -> list[str]:
    """Return the candidate names, in the order they are compared."""

    return list(MODEL_BUILDERS)


def build_model(name: str) -> TransformedTargetRegressor:
    """Assemble one candidate: preprocessing, regressor and target transform.

    The returned estimator is fitted on a raw dataframe and predicts sale
    prices in dollars.
    """

    if name not in MODEL_BUILDERS:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", PreprocessorStep()),
            ("regressor", MODEL_BUILDERS[name]()),
        ]
    )

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
