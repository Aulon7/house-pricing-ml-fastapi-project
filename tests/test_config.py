import dataclasses
from pathlib import Path

import pytest

from src.config import CONFIG, Config


def test_paths_are_derived_from_the_project_root():
    assert CONFIG.train_csv == CONFIG.data_dir / "train.csv"
    assert CONFIG.model_path == CONFIG.models_dir / "model.pkl"
    assert CONFIG.features_path == CONFIG.models_dir / "features.pkl"


def test_project_root_contains_the_source_package():
    assert (CONFIG.project_root / "src" / "config.py").is_file()


def test_directories_can_be_overridden_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOUSE_PRICING_MODELS_DIR", str(tmp_path / "artifacts"))

    config = Config()

    assert config.models_dir == (tmp_path / "artifacts").resolve()
    assert config.model_path.name == "model.pkl"


def test_ensure_directories_creates_missing_folders(monkeypatch, tmp_path):
    monkeypatch.setenv("HOUSE_PRICING_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HOUSE_PRICING_REPORTS_DIR", str(tmp_path / "reports"))

    config = Config()
    config.ensure_directories()

    assert config.models_dir.is_dir()
    assert config.reports_dir.is_dir()


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.random_seed = 1  # type: ignore[misc]


def test_training_data_is_present():
    train_csv: Path = CONFIG.train_csv

    assert train_csv.is_file(), f"Kaggle train.csv missing at {train_csv}"
