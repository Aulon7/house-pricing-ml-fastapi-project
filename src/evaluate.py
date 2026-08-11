"""Evaluation metrics and the model comparison report.

Everything here works on sale prices in dollars, not on the log target the
models are trained against. Predictions arrive already converted back, so a
metric never has to know how the target was transformed.

Two error metrics are reported on purpose. RMSE on the log scale is the
Kaggle metric and treats a $20k error on a cheap house as seriously as on an
expensive one, which is what makes it right for comparing models. RMSE in
dollars is the one to quote to a person, because it answers "how far off is
this prediction".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def _as_price_array(values, name: str) -> np.ndarray:
    """Validate and coerce a vector of sale prices."""

    array = np.asarray(values, dtype=float).ravel()

    if array.size == 0:
        raise ValueError(f"{name} is empty")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return array


@dataclass(frozen=True)
class RegressionMetrics:
    """The scores of one model on one set of predictions."""

    rmse_log: float
    rmse: float
    mae: float
    mape: float
    r2: float

    @classmethod
    def from_predictions(cls, y_true, y_pred) -> "RegressionMetrics":
        """Score predicted sale prices against the true ones."""

        actual = _as_price_array(y_true, "y_true")
        predicted = _as_price_array(y_pred, "y_pred")

        if actual.shape != predicted.shape:
            raise ValueError(
                f"y_true has {actual.size} rows but y_pred has {predicted.size}"
            )

        if (actual <= 0).any():
            raise ValueError("y_true contains non-positive sale prices")

        # A model is free to predict a negative price; log1p is not. Clipping
        # keeps the log metric defined and still punishes the bad prediction.
        safe_predicted = np.clip(predicted, 0.0, None)

        return cls(
            rmse_log=float(
                np.sqrt(
                    mean_squared_error(np.log1p(actual), np.log1p(safe_predicted))
                )
            ),
            rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
            mae=float(mean_absolute_error(actual, predicted)),
            mape=float(mean_absolute_percentage_error(actual, predicted)),
            r2=float(r2_score(actual, predicted)),
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


#: Ordered so the comparison table always reads the same way.
METRIC_NAMES: tuple[str, ...] = tuple(RegressionMetrics.__dataclass_fields__)

#: The metric that decides which model is exported. Lower is better.
SELECTION_METRIC = "rmse_log"


def aggregate_folds(folds: Sequence[RegressionMetrics]) -> dict[str, float]:
    """Average cross-validation folds, keeping the spread.

    The standard deviation matters as much as the mean here: two models whose
    means differ by less than a fold's spread have not been told apart.
    """

    if not folds:
        raise ValueError("no folds to aggregate")

    summary: dict[str, float] = {}

    for metric in METRIC_NAMES:
        scores = np.array([getattr(fold, metric) for fold in folds], dtype=float)

        summary[f"{metric}_mean"] = float(scores.mean())
        summary[f"{metric}_std"] = float(scores.std(ddof=0))

    summary["n_folds"] = len(folds)

    return summary


def comparison_table(
    results: Mapping[str, Sequence[RegressionMetrics]],
    fit_seconds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the model comparison report, best model first."""

    if not results:
        raise ValueError("no models to compare")

    rows = []

    for model_name, folds in results.items():
        row = {"model": model_name, **aggregate_folds(folds)}

        if fit_seconds is not None and model_name in fit_seconds:
            row["fit_seconds"] = float(fit_seconds[model_name])

        rows.append(row)

    table = pd.DataFrame(rows)

    return table.sort_values(
        f"{SELECTION_METRIC}_mean", ascending=True
    ).reset_index(drop=True)


def best_model_name(table: pd.DataFrame) -> str:
    """Return the winning model from a comparison table."""

    if table.empty:
        raise ValueError("comparison table is empty")

    return str(table.loc[table[f"{SELECTION_METRIC}_mean"].idxmin(), "model"])


def format_comparison(table: pd.DataFrame) -> str:
    """Render the comparison table for the console."""

    display = pd.DataFrame(
        {
            "model": table["model"],
            # ASCII only: this table is logged to the Windows console, which
            # is not UTF-8 by default and would raise on a "±".
            "rmse_log": [
                f"{mean:.4f} +/- {std:.4f}"
                for mean, std in zip(table["rmse_log_mean"], table["rmse_log_std"])
            ],
            "rmse_$": [f"{value:,.0f}" for value in table["rmse_mean"]],
            "mae_$": [f"{value:,.0f}" for value in table["mae_mean"]],
            "mape": [f"{value:.1%}" for value in table["mape_mean"]],
            "r2": [f"{value:.3f}" for value in table["r2_mean"]],
        }
    )

    if "fit_seconds" in table.columns:
        display["fit_s"] = [f"{value:.1f}" for value in table["fit_seconds"]]

    return display.to_string(index=False)
