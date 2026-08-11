import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor

from src.config import CONFIG
from src.models import (
    MODEL_SPECS,
    PreprocessorStep,
    available_models,
    build_model,
    feature_names,
)


@pytest.fixture(scope="module")
def houses() -> pd.DataFrame:
    """A small slice of the real training data.

    The four-row fixture from P1 is too thin for a tree to split on, so the
    model tests use real rows and keep the slice small for speed.
    """

    return pd.read_csv(CONFIG.train_csv).head(150)


@pytest.fixture(scope="module")
def features_and_target(houses):
    return houses.drop(columns=CONFIG.target), houses[CONFIG.target]


def test_the_planned_models_are_registered():
    assert available_models() == ["xgboost", "lightgbm", "random_forest", "mlp"]
    assert set(MODEL_SPECS) == set(available_models())


def test_only_the_neural_network_is_scaled():
    """Trees split on raw values, so scaling them would only cost time."""

    scaled = {name for name, spec in MODEL_SPECS.items() if spec.needs_scaling}

    assert scaled == {"mlp"}

    assert "scaler" in build_model("mlp").regressor.named_steps
    assert "scaler" not in build_model("xgboost").regressor.named_steps


def test_unknown_model_names_are_rejected_with_the_options():
    with pytest.raises(KeyError, match="xgboost"):
        build_model("catboost")


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "random_forest", "mlp"])
def test_every_candidate_is_cloneable(name):
    """Cross-validation clones the estimator before each fold."""

    model = build_model(name)

    assert isinstance(model, TransformedTargetRegressor)
    assert isinstance(clone(model), TransformedTargetRegressor)


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "random_forest", "mlp"])
def test_candidates_predict_plausible_sale_prices(name, features_and_target):
    X, y = features_and_target

    model = build_model(name).fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()

    # Dollars, not the log target: Ames prices live in the hundred thousands.
    assert predictions.min() > 10_000
    assert predictions.max() < 1_000_000


def test_the_pipeline_learns_something_beyond_the_mean(features_and_target):
    X, y = features_and_target

    model = build_model("xgboost").fit(X, y)
    predictions = model.predict(X)

    baseline_error = np.abs(y - y.mean()).mean()
    model_error = np.abs(y - predictions).mean()

    assert model_error < baseline_error / 2


def test_the_target_transform_is_applied_and_inverted(features_and_target):
    X, y = features_and_target

    model = build_model("random_forest").fit(X, y)

    # The inner regressor works in log space, the wrapper returns dollars.
    inner = model.regressor_.predict(X)

    assert inner.max() < 20
    assert model.predict(X).min() > 10_000


def test_identifier_and_target_never_reach_the_regressor(features_and_target):
    X, y = features_and_target

    model = build_model("random_forest").fit(X, y)
    names = feature_names(model)

    assert CONFIG.id_column not in names
    assert CONFIG.target not in names
    assert len(names) > 50


def test_training_twice_gives_identical_predictions(features_and_target):
    X, y = features_and_target

    first = build_model("lightgbm").fit(X, y).predict(X)
    second = build_model("lightgbm").fit(X, y).predict(X)

    np.testing.assert_allclose(first, second)


def test_preprocessor_step_is_fitted_per_fit_call(features_and_target):
    """A cloned step must not carry state from the estimator it came from."""

    X, _ = features_and_target

    step = PreprocessorStep().fit(X)
    fresh = clone(step)

    assert not hasattr(fresh, "preprocessor_")

    # Names are recorded on transform, so asking too early says so clearly.
    with pytest.raises(AttributeError, match="first transform"):
        step.get_feature_names_out()

    assert step.transform(X).shape[0] == len(X)
    assert list(step.get_feature_names_out()) == step.feature_names_
