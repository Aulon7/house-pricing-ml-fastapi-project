"""Cross-validate the candidate models and pick a winner.

Run with:

    python -m src.train
    python -m src.train --models xgboost lightgbm --folds 3

Model selection uses k-fold cross-validation rather than a single hold-out
split. With 1460 training rows a single split moves the score by more than
the gap between two candidates, so the winner of one split is partly the
winner of one lucky seed.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from src.config import CONFIG
from src.evaluate import (
    RegressionMetrics,
    best_model_name,
    comparison_table,
    format_comparison,
)
from src.models import available_models, build_model
from src.utils import get_logger, set_seed

LOGGER = get_logger("train")


@dataclass
class CandidateResult:
    """The cross-validation outcome for one model."""

    name: str
    folds: list[RegressionMetrics] = field(default_factory=list)
    fit_seconds: float = 0.0


def load_training_data(
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read the Kaggle training file and split off the target."""

    path = csv_path or CONFIG.train_csv

    if not path.is_file():
        raise FileNotFoundError(f"training data not found at {path}")

    frame = pd.read_csv(path)

    if CONFIG.target not in frame.columns:
        raise ValueError(f"{path.name} has no {CONFIG.target} column")

    features = frame.drop(columns=[CONFIG.target])
    target = frame[CONFIG.target]

    return features, target


def cross_validate(
    name: str,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int | None = None,
) -> CandidateResult:
    """Score one candidate across the folds.

    The estimator is cloned and refitted per fold, which means the encoder
    and every imputation statistic are learned from the training folds only.
    """

    splits = n_splits or CONFIG.n_splits

    if splits < 2:
        raise ValueError(f"cross-validation needs at least 2 folds, got {splits}")

    folds = KFold(n_splits=splits, shuffle=True, random_state=CONFIG.random_seed)
    result = CandidateResult(name=name)

    for number, (train_index, valid_index) in enumerate(folds.split(features), 1):
        X_train = features.iloc[train_index]
        X_valid = features.iloc[valid_index]
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]

        model = clone(build_model(name))

        started = time.perf_counter()
        model.fit(X_train, y_train)
        result.fit_seconds += time.perf_counter() - started

        scores = RegressionMetrics.from_predictions(y_valid, model.predict(X_valid))
        result.folds.append(scores)

        LOGGER.info(
            "%s fold %d/%d: rmse_log=%.4f rmse=$%s",
            name,
            number,
            splits,
            scores.rmse_log,
            f"{scores.rmse:,.0f}",
        )

    return result


def compare_models(
    names: list[str] | None = None,
    features: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    n_splits: int | None = None,
) -> pd.DataFrame:
    """Cross-validate every candidate and return the comparison table."""

    candidates = names or available_models()

    unknown = [name for name in candidates if name not in available_models()]

    if unknown:
        raise KeyError(
            f"unknown model(s) {', '.join(unknown)}; "
            f"available: {', '.join(available_models())}"
        )

    if features is None or target is None:
        features, target = load_training_data()

    results = {}
    fit_seconds = {}

    for name in candidates:
        LOGGER.info("cross-validating %s", name)

        result = cross_validate(name, features, target, n_splits)

        results[name] = result.folds
        fit_seconds[name] = result.fit_seconds

    return comparison_table(results, fit_seconds)


def write_report(table: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the comparison table so a run can be revisited later."""

    destination = path or CONFIG.comparison_report_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(destination, index=False)

    return destination


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_models(),
        default=available_models(),
        help="candidates to compare (default: all)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=CONFIG.n_splits,
        help=f"number of cross-validation folds (default: {CONFIG.n_splits})",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    set_seed()

    features, target = load_training_data()

    LOGGER.info(
        "training on %d rows and %d raw columns", len(features), features.shape[1]
    )

    table = compare_models(
        arguments.models, features, target, n_splits=arguments.folds
    )

    LOGGER.info("comparison over %d folds:\n%s", arguments.folds, format_comparison(table))

    report_path = write_report(table)
    LOGGER.info("report written to %s", report_path)

    winner = best_model_name(table)
    LOGGER.info("best model: %s", winner)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
