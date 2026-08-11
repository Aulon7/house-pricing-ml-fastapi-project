import dataclasses
from pathlib import Path

import pytest

from src.config import CONFIG


def test_paths_are_derived_from_the_project_root():
    assert CONFIG.train_csv == CONFIG.data_dir / "train.csv"
    assert CONFIG.model_path == CONFIG.models_dir / "model.pkl"
    assert CONFIG.features_path == CONFIG.models_dir / "features.pkl"


def test_project_root_contains_the_source_package():
    assert (CONFIG.project_root / "src" / "config.py").is_file()


def test_ensure_directories_creates_missing_folders(tmp_path):
    config = dataclasses.replace(
        CONFIG,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    config.ensure_directories()

    assert config.models_dir.is_dir()
    assert config.reports_dir.is_dir()
    assert config.model_path == tmp_path / "models" / "model.pkl"


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.random_seed = 1  # type: ignore[misc]


def test_training_data_is_present():
    train_csv: Path = CONFIG.train_csv

    assert train_csv.is_file(), f"Kaggle train.csv missing at {train_csv}"
