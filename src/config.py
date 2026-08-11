"""Central configuration for the Ames Housing pipeline.

Every path, column name and tuning constant lives here so that training,
evaluation and model serving all agree on where things are and how the data
is split. Directories can be overridden with environment variables, which is
what lets the FastAPI container point at a mounted models volume without any
code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _directory_from_env(variable: str, default: Path) -> Path:
    """Return the directory named by an environment variable, or the default."""

    configured = os.getenv(variable)

    if not configured:
        return default

    return Path(configured).expanduser().resolve()


@dataclass(frozen=True)
class Config:
    """Paths and constants shared by every stage of the pipeline."""

    project_root: Path = PROJECT_ROOT

    data_dir: Path = field(
        default_factory=lambda: _directory_from_env(
            "HOUSE_PRICING_DATA_DIR", PROJECT_ROOT / "data"
        )
    )

    models_dir: Path = field(
        default_factory=lambda: _directory_from_env(
            "HOUSE_PRICING_MODELS_DIR", PROJECT_ROOT / "models"
        )
    )

    reports_dir: Path = field(
        default_factory=lambda: _directory_from_env(
            "HOUSE_PRICING_REPORTS_DIR", PROJECT_ROOT / "reports"
        )
    )

    # The Kaggle target and the row identifier that must never reach the model.
    target: str = "SalePrice"
    id_column: str = "Id"

    # One seed for every source of randomness in the project.
    random_seed: int = 42

    # 1460 training rows make a single hold-out split noisier than the gap
    # between candidate models, so model selection uses cross-validation.
    n_splits: int = 5

    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "model.pkl"

    @property
    def features_path(self) -> Path:
        return self.models_dir / "features.pkl"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "metadata.json"

    @property
    def comparison_report_path(self) -> Path:
        return self.reports_dir / "model_comparison.csv"

    def ensure_directories(self) -> None:
        """Create the output directories if they do not exist yet."""

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
