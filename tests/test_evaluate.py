import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    METRIC_NAMES,
    RegressionMetrics,
    aggregate_folds,
    best_model_name,
    comparison_table,
    format_comparison,
)

PRICES = [200_000.0, 150_000.0, 300_000.0, 175_000.0]


def metrics_with(rmse_log: float) -> RegressionMetrics:
    """A metrics object where only the selection metric matters."""

    return RegressionMetrics(rmse_log=rmse_log, rmse=0.0, mae=0.0, mape=0.0, r2=0.0)


def test_perfect_predictions_score_perfectly():
    scores = RegressionMetrics.from_predictions(PRICES, PRICES)

    assert scores.rmse_log == pytest.approx(0.0)
    assert scores.rmse == pytest.approx(0.0)
    assert scores.mae == pytest.approx(0.0)
    assert scores.mape == pytest.approx(0.0)
    assert scores.r2 == pytest.approx(1.0)


def test_dollar_metrics_match_hand_calculation():
    actual = [100_000.0, 200_000.0]
    predicted = [110_000.0, 180_000.0]

    scores = RegressionMetrics.from_predictions(actual, predicted)

    # Errors are 10,000 and 20,000.
    assert scores.mae == pytest.approx(15_000.0)
    assert scores.rmse == pytest.approx(np.sqrt((10_000**2 + 20_000**2) / 2))
    assert scores.mape == pytest.approx((0.10 + 0.10) / 2)


def test_log_metric_weights_cheap_and_expensive_houses_equally():
    """The same relative error on two price levels must score the same."""

    cheap = RegressionMetrics.from_predictions(
        [100_000.0, 120_000.0], [110_000.0, 132_000.0]
    )
    expensive = RegressionMetrics.from_predictions(
        [500_000.0, 600_000.0], [550_000.0, 660_000.0]
    )

    assert cheap.rmse_log == pytest.approx(expensive.rmse_log, rel=1e-3)
    assert expensive.rmse > cheap.rmse * 4


def test_negative_predictions_are_clipped_for_the_log_metric():
    scores = RegressionMetrics.from_predictions(
        [200_000.0, 250_000.0], [-5_000.0, 250_000.0]
    )

    assert np.isfinite(scores.rmse_log)
    assert scores.rmse == pytest.approx(205_000.0 / np.sqrt(2))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="rows"):
        RegressionMetrics.from_predictions([1.0, 2.0], [1.0])


def test_missing_values_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        RegressionMetrics.from_predictions([1.0, np.nan], [1.0, 2.0])


def test_non_positive_actual_prices_are_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        RegressionMetrics.from_predictions([0.0, 200_000.0], [1.0, 2.0])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        RegressionMetrics.from_predictions([], [])


def test_aggregate_folds_reports_mean_and_spread():
    folds = [metrics_with(0.10), metrics_with(0.14)]

    summary = aggregate_folds(folds)

    assert summary["rmse_log_mean"] == pytest.approx(0.12)
    assert summary["rmse_log_std"] == pytest.approx(0.02)
    assert summary["n_folds"] == 2

    for metric in METRIC_NAMES:
        assert f"{metric}_mean" in summary
        assert f"{metric}_std" in summary


def test_aggregate_folds_rejects_an_empty_run():
    with pytest.raises(ValueError, match="no folds"):
        aggregate_folds([])


def test_comparison_table_puts_the_best_model_first():
    table = comparison_table(
        {
            "random_forest": [metrics_with(0.15)],
            "xgboost": [metrics_with(0.12)],
            "lightgbm": [metrics_with(0.13)],
        }
    )

    assert list(table["model"]) == ["xgboost", "lightgbm", "random_forest"]
    assert best_model_name(table) == "xgboost"


def test_comparison_table_carries_fit_times_when_given():
    table = comparison_table(
        {"xgboost": [metrics_with(0.12)]}, fit_seconds={"xgboost": 3.5}
    )

    assert table.loc[0, "fit_seconds"] == pytest.approx(3.5)


def test_comparison_table_rejects_an_empty_run():
    with pytest.raises(ValueError, match="no models"):
        comparison_table({})


def test_best_model_name_rejects_an_empty_table():
    with pytest.raises(ValueError, match="empty"):
        best_model_name(pd.DataFrame())


def test_format_comparison_renders_every_model():
    table = comparison_table(
        {
            "xgboost": [
                RegressionMetrics.from_predictions(PRICES, PRICES),
                RegressionMetrics.from_predictions(
                    PRICES, [price * 1.05 for price in PRICES]
                ),
            ],
            "random_forest": [
                RegressionMetrics.from_predictions(
                    PRICES, [price * 1.10 for price in PRICES]
                )
            ],
        }
    )

    rendered = format_comparison(table)

    assert "xgboost" in rendered
    assert "random_forest" in rendered
    assert "+/-" in rendered
    assert rendered.isascii()
