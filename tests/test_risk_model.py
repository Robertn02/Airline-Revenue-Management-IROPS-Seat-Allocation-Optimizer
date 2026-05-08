"""Tests for the risk estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reroute.model.risk import (
    RiskModel,
    features_for_scenario,
    synthesize_misconnect_labels,
    train_from_scenarios,
)
from reroute.sim.generator import generate_dataset


@pytest.fixture(scope="module")
def trained_model():
    """Train once for all tests."""
    scns = generate_dataset(n_scenarios=100, seed=42)
    model, _, _ = train_from_scenarios(scns)
    return model, scns


def test_model_trains_with_acceptable_auc(trained_model):
    model, _ = trained_model
    assert model.train_results.auc > 0.85


def test_model_well_calibrated(trained_model):
    model, _ = trained_model
    assert model.train_results.brier < 0.10


def test_predictions_in_valid_range(trained_model):
    model, scns = trained_model
    df = features_for_scenario(scns[0])
    probs = model.predict_proba(df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_method(trained_model):
    """The .predict() convenience method should work."""
    model, scns = trained_model
    probs = model.predict(scns[0])
    assert len(probs) == len(scns[0].passengers)


def test_confidence_classification(trained_model):
    model, _ = trained_model
    assert model.confidence_class(0.05) == "H"
    assert model.confidence_class(0.95) == "H"
    assert model.confidence_class(0.25) == "M"
    assert model.confidence_class(0.50) == "L"


def test_label_artifact_fixed(trained_model):
    """REGRESSION TEST for the v1 label artifact.

    yield_usd should be at most a marginal feature in permutation
    importance. effective_buffer_min should be the dominant signal.
    """
    model, _ = trained_model
    fi = model.train_results.feature_importance
    # effective_buffer should be top
    top = max(fi.items(), key=lambda kv: kv[1])
    assert top[0] == "effective_buffer_min", \
        f"Expected effective_buffer_min as top feature, got {top[0]}"
    # yield_usd should be much less important than buffer
    assert fi["yield_usd"] < fi["effective_buffer_min"] * 0.1, \
        f"yield_usd importance ({fi['yield_usd']:.3f}) too high — label artifact may have returned"


def test_label_synthesis_independent_of_yield():
    """Labels should be uncorrelated with yield (correlation < 0.05)."""
    scns = generate_dataset(n_scenarios=100, seed=7)
    feature_dfs = [features_for_scenario(s) for s in scns]
    df = pd.concat(feature_dfs, ignore_index=True)
    rng = np.random.default_rng(0)
    labels = synthesize_misconnect_labels(df, rng)
    corr = abs(df["yield_usd"].corr(pd.Series(labels)))
    assert corr < 0.05, f"yield-label correlation {corr:.3f} too high"


def test_label_synthesis_correlated_with_buffer():
    """Labels should be strongly correlated with effective buffer."""
    scns = generate_dataset(n_scenarios=100, seed=7)
    feature_dfs = [features_for_scenario(s) for s in scns]
    df = pd.concat(feature_dfs, ignore_index=True)
    rng = np.random.default_rng(0)
    labels = synthesize_misconnect_labels(df, rng)
    corr = df["effective_buffer_min"].corr(pd.Series(labels))
    assert corr < -0.4, f"buffer-label correlation {corr:.3f} too weak"


def test_save_and_load(tmp_path, trained_model):
    model, _ = trained_model
    fpath = tmp_path / "model.pkl"
    model.save(str(fpath))
    loaded = RiskModel.load(str(fpath))
    assert loaded.train_results.auc == model.train_results.auc


def test_below_mct_increases_probability(trained_model):
    model, _ = trained_model
    base = {
        "delay_min": 90, "buffer_min": 60, "effective_buffer_min": -30,
        "below_mct": 1, "tier_EXP": 0, "tier_PLT": 0, "tier_GLD": 0,
        "cabin_F": 0, "cabin_Yplus": 0, "yield_usd": 400, "log_yield": 6.0,
        "has_ssr": 0, "is_um": 0,
    }
    safe = {**base, "effective_buffer_min": 90, "below_mct": 0, "buffer_min": 180}
    df = pd.DataFrame([base, safe])
    probs = model.predict_proba(df)
    assert probs[0] > probs[1]
