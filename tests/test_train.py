import pandas as pd
import pytest

from src.config import CONFIG
from src.evaluate import SELECTION_METRIC
from src.models import available_models
from src.train import (
    compare_models,
    cross_validate,
    load_training_data,
    main,
    parse_arguments,
    write_report,
)


@pytest.fixture(scope="module")
def small_dataset():
    """A slice of the real training data, small enough to fit repeatedly."""

    features, target = load_training_data()

    return features.head(200), target.head(200)


def test_load_training_data_separates_the_target():
    features, target = load_training_data()

    assert CONFIG.target not in features.columns
    assert target.name == CONFIG.target
    assert len(features) == len(target) == 1460


def test_load_training_data_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="training data"):
        load_training_data(tmp_path / "absent.csv")


def test_load_training_data_reports_a_missing_target(tmp_path):
    path = tmp_path / "no_target.csv"
    pd.DataFrame({"Id": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match=CONFIG.target):
        load_training_data(path)


def test_cross_validate_scores_every_fold(small_dataset):
    features, target = small_dataset

    result = cross_validate("random_forest", features, target, n_splits=3)

    assert result.name == "random_forest"
    assert len(result.folds) == 3
    assert result.fit_seconds > 0

    for fold in result.folds:
        assert 0 < fold.rmse_log < 1
        assert fold.rmse > 0


def test_cross_validate_rejects_a_single_fold(small_dataset):
    features, target = small_dataset

    with pytest.raises(ValueError, match="at least 2 folds"):
        cross_validate("random_forest", features, target, n_splits=1)


def test_cross_validate_is_reproducible(small_dataset):
    features, target = small_dataset

    first = cross_validate("lightgbm", features, target, n_splits=2)
    second = cross_validate("lightgbm", features, target, n_splits=2)

    assert [fold.rmse_log for fold in first.folds] == [
        fold.rmse_log for fold in second.folds
    ]


def test_compare_models_ranks_the_requested_candidates(small_dataset):
    features, target = small_dataset

    table = compare_models(
        ["random_forest", "lightgbm"], features, target, n_splits=2
    )

    assert set(table["model"]) == {"random_forest", "lightgbm"}
    assert table[f"{SELECTION_METRIC}_mean"].is_monotonic_increasing
    assert (table["n_folds"] == 2).all()
    assert (table["fit_seconds"] > 0).all()


def test_compare_models_rejects_unknown_candidates(small_dataset):
    features, target = small_dataset

    with pytest.raises(KeyError, match="catboost"):
        compare_models(["catboost"], features, target, n_splits=2)


def test_write_report_creates_a_readable_csv(tmp_path, small_dataset):
    features, target = small_dataset

    table = compare_models(["random_forest"], features, target, n_splits=2)
    destination = write_report(table, tmp_path / "nested" / "report.csv")

    assert destination.is_file()

    reloaded = pd.read_csv(destination)

    assert list(reloaded["model"]) == list(table["model"])


def test_arguments_default_to_every_model_and_the_configured_folds():
    arguments = parse_arguments([])

    assert arguments.models == available_models()
    assert "mlp" in arguments.models
    assert arguments.folds == CONFIG.n_splits


def test_export_is_off_unless_asked_for():
    assert parse_arguments([]).export is False
    assert parse_arguments(["--export"]).export is True


def test_arguments_reject_an_unknown_model():
    with pytest.raises(SystemExit):
        parse_arguments(["--models", "catboost"])


def test_main_runs_end_to_end_and_writes_the_report(monkeypatch, tmp_path):
    report = tmp_path / "model_comparison.csv"

    monkeypatch.setattr(
        "src.train.CONFIG",
        type(CONFIG)(models_dir=tmp_path, reports_dir=tmp_path),
    )

    assert main(["--models", "random_forest", "--folds", "2"]) == 0
    assert report.is_file()
