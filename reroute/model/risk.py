"""Misconnect risk estimator.

Trains a gradient boosted tree (LightGBM) to predict per-passenger misconnect
probability, then post-calibrates with Platt scaling so the probabilities are
suitable for direct use in the LP optimizer's expected-cost objective.

Calibration matters because raw GBT probabilities are typically miscalibrated
even when AUC is good (Niculescu-Mizil & Caruana 2005).

LABEL ARTIFACT FIX (v2):
    The earlier label generator made yield/cabin spuriously informative
    because high-yield passengers were correlated with tighter buffers in
    the data. The fixed version below generates labels that depend ONLY
    on operational features (buffer, delay, SSR, UM, tier) — not on yield
    or cabin. The model now has to learn the true causal structure, which
    is closer to what would actually be true in real airline data.

Author: Phuc Nguyen
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

from reroute.core.config import Config, default_config
from reroute.core.logging import get_logger
from reroute.core.types import ConfidenceClass, DisruptionScenario, Flight, Passenger

logger = get_logger(__name__)


@dataclass
class TrainResults:
    """Aggregate statistics from a model training run."""
    auc: float
    brier: float
    log_loss: float
    n_train: int
    n_test: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    label_base_rate: float = 0.0


def passenger_features(
    pax: Passenger,
    inbound: Flight,
    outbound_lookup: dict[str, Flight],
    config: Optional[Config] = None,
) -> dict[str, float]:
    """Compute features for a single passenger in a scenario context."""
    cfg = config or default_config()
    mct = cfg.operational.mct_domestic_min

    outbound = outbound_lookup.get(pax.outbound_flight_id)

    if outbound is None:
        # Original outbound not in recovery set (already departed)
        return {
            "delay_min": float(inbound.delay_min),
            "buffer_min": -999.0,
            "effective_buffer_min": -999.0,
            "below_mct": 1.0,
            "tier_EXP": float(pax.tier == "EXP"),
            "tier_PLT": float(pax.tier == "PLT"),
            "tier_GLD": float(pax.tier == "GLD"),
            "cabin_F": float(pax.cabin == "F"),
            "cabin_Yplus": float(pax.cabin == "Y+"),
            "yield_usd": float(pax.yield_usd),
            "log_yield": float(np.log1p(pax.yield_usd)),
            "has_ssr": float(pax.has_ssr),
            "is_um": float(pax.is_unaccompanied_minor),
        }

    effective = pax.sched_connection_min - inbound.delay_min
    return {
        "delay_min": float(inbound.delay_min),
        "buffer_min": float(pax.sched_connection_min),
        "effective_buffer_min": float(effective),
        "below_mct": float(effective < mct),
        "tier_EXP": float(pax.tier == "EXP"),
        "tier_PLT": float(pax.tier == "PLT"),
        "tier_GLD": float(pax.tier == "GLD"),
        "cabin_F": float(pax.cabin == "F"),
        "cabin_Yplus": float(pax.cabin == "Y+"),
        "yield_usd": float(pax.yield_usd),
        "log_yield": float(np.log1p(pax.yield_usd)),
        "has_ssr": float(pax.has_ssr),
        "is_um": float(pax.is_unaccompanied_minor),
    }


def features_for_scenario(scn: DisruptionScenario, config: Optional[Config] = None) -> pd.DataFrame:
    """Compute feature DataFrame for all passengers in a scenario."""
    outbound_lookup = {f.flight_id: f for f in scn.recovery_flights}
    rows = [passenger_features(p, scn.inbound_flight, outbound_lookup, config)
            for p in scn.passengers]
    df = pd.DataFrame(rows)
    df["pax_id"] = [p.pax_id for p in scn.passengers]
    df["scenario_id"] = scn.scenario_id
    return df


def synthesize_misconnect_labels(
    df: pd.DataFrame,
    rng: np.random.Generator,
    config: Optional[Config] = None,
) -> np.ndarray:
    """Generate synthetic ground-truth misconnect labels.

    LABEL MODEL (FIXED):
        Misconnect probability depends ONLY on operational features that
        would actually drive a real misconnect:
            - effective_buffer_min (the dominant signal)
            - has_ssr / is_um (these passengers need extra handling time)
            - tier (top-tier passengers get fast-track services that help marginally)

        Yield, cabin, and tier-as-yield-proxy are EXCLUDED from the label
        model. This forces the trained GBT to lean on the actual causal
        signals, fixing the artifact in the v1 label generator where
        yield_usd dominated feature importance.

    Returns:
        Binary array of length len(df), where 1 = misconnected.
    """
    cfg = config or default_config()
    mct = cfg.operational.mct_domestic_min
    eb = df["effective_buffer_min"].values

    # Buffer-driven baseline: piecewise logit
    base_logit = np.where(
        eb < 0, 4.5,                          # already missed: ~99%
        np.where(eb < mct, 2.2,               # below MCT: ~90%
        np.where(eb < 60, 0.2,                # tight: ~55%
        np.where(eb < 90, -1.6, -3.2)))       # comfortable: ~17% / safe: ~4%
    )

    # SSR / unaccompanied minor — extra handling time required
    base_logit += 1.0 * df["has_ssr"].values
    base_logit += 1.3 * df["is_um"].values

    # Top-tier marginal benefit (fast-track lanes, dedicated agents)
    base_logit -= 0.5 * df["tier_EXP"].values
    base_logit -= 0.25 * df["tier_PLT"].values

    # Noise
    noise = rng.normal(0, 0.55, size=len(eb))
    logit = base_logit + noise
    prob = 1.0 / (1.0 + np.exp(-logit))
    return (rng.uniform(0, 1, size=len(eb)) < prob).astype(int)


class RiskModel:
    """Calibrated misconnect probability estimator."""

    FEATURE_COLS = [
        "delay_min",
        "buffer_min",
        "effective_buffer_min",
        "below_mct",
        "tier_EXP",
        "tier_PLT",
        "tier_GLD",
        "cabin_F",
        "cabin_Yplus",
        "yield_usd",
        "log_yield",
        "has_ssr",
        "is_um",
    ]

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config()
        self.model: Any = None  # type: ignore
        self.calibrator: Any = None  # type: ignore
        self.train_results: Optional[TrainResults] = None

    def _build_base(self):
        hp = self.config.model
        if HAS_LGBM:
            return lgb.LGBMClassifier(
                n_estimators=hp.n_estimators,
                num_leaves=hp.num_leaves,
                learning_rate=hp.learning_rate,
                min_child_samples=hp.min_child_samples,
                random_state=hp.random_seed,
                verbose=-1,
            )
        return GradientBoostingClassifier(
            n_estimators=hp.n_estimators,
            max_depth=4,
            learning_rate=hp.learning_rate,
            random_state=hp.random_seed,
        )

    def fit(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> TrainResults:
        """Train + calibrate on a feature DataFrame and labels.

        Holds out `test_size` for evaluation. Calibrates with Platt scaling
        on a separate held-out fold via CalibratedClassifierCV.
        """
        hp = self.config.model
        X = df[self.FEATURE_COLS].values
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=hp.test_size,
            random_state=hp.random_seed, stratify=y,
        )

        # Train base for feature importance
        base = self._build_base()
        base.fit(X_train, y_train)
        self.model = base

        # Calibrate
        cal_base = self._build_base()
        cal = CalibratedClassifierCV(cal_base, method="sigmoid", cv=hp.calibration_cv_folds)
        cal.fit(X_train, y_train)
        self.calibrator = cal

        # Evaluate
        probs_calib = cal.predict_proba(X_test)[:, 1]

        # Use PERMUTATION importance, not split-count importance.
        # LightGBM's feature_importances_ is biased toward continuous features
        # with many unique values (they offer more split points). Permutation
        # importance measures actual causal contribution by shuffling each
        # feature and measuring AUC degradation.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            cal, X_test, y_test, n_repeats=5,
            random_state=hp.random_seed, n_jobs=1, scoring="roc_auc"
        )
        # Clip negatives to 0 (some features can hurt slightly)
        imp_vals = np.maximum(perm.importances_mean, 0)
        total = sum(imp_vals) or 1.0
        fi = {col: round(float(imp) / total, 4)
              for col, imp in zip(self.FEATURE_COLS, imp_vals)}

        self.train_results = TrainResults(
            auc=float(roc_auc_score(y_test, probs_calib)),
            brier=float(brier_score_loss(y_test, probs_calib)),
            log_loss=float(log_loss(y_test, probs_calib)),
            n_train=int(len(y_train)),
            n_test=int(len(y_test)),
            feature_importance=fi,
            label_base_rate=float(labels.mean()),
        )
        logger.info(
            f"Trained risk model on {len(y_train)} samples — "
            f"AUC={self.train_results.auc:.3f} "
            f"Brier={self.train_results.brier:.3f}"
        )
        return self.train_results

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return calibrated misconnect probabilities."""
        if self.calibrator is None:
            raise RuntimeError("Model not fitted — call fit() first")
        X = df[self.FEATURE_COLS].values
        return self.calibrator.predict_proba(X)[:, 1]

    def predict(self, scenario: DisruptionScenario) -> np.ndarray:
        """Convenience: predict for all passengers in a scenario."""
        df = features_for_scenario(scenario, self.config)
        return self.predict_proba(df)

    def confidence_class(self, prob: float) -> ConfidenceClass:
        """Classify uncertainty: H = clear, M = moderate, L = uncertain.

        Low-confidence cases should be routed to the human queue rather than
        producing automated recommendations.
        """
        d = abs(prob - 0.5)
        if d > 0.30:
            return "H"
        if d > 0.15:
            return "M"
        return "L"

    def save(self, path: str | Path) -> None:
        """Save model to disk via pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "calibrator": self.calibrator,
                "results": self.train_results,
                "config": self.config,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> RiskModel:
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(config=data.get("config"))
        m.model = data["model"]
        m.calibrator = data["calibrator"]
        m.train_results = data["results"]
        return m

    @classmethod
    def load_default(cls) -> RiskModel:
        """Load the default trained model from the package's results dir.

        Falls back to training a fresh model from defaults if no cached one
        exists.
        """
        from reroute.sim.generator import generate_dataset
        # Try cached model first
        candidates = [
            Path("results/model.pkl"),
            Path("../results/model.pkl"),
            Path(__file__).parent.parent.parent / "results" / "model.pkl",
        ]
        for c in candidates:
            if c.exists():
                return cls.load(c)
        # Train fresh
        logger.info("No cached model found — training a fresh one")
        cfg = default_config()
        scns = generate_dataset(n_scenarios=200, seed=42, config=cfg)
        m = cls(cfg)
        train_from_scenarios(scns, model=m)
        return m


def train_from_scenarios(
    scenarios: list[DisruptionScenario],
    seed: int = 42,
    config: Optional[Config] = None,
    model: Optional[RiskModel] = None,
) -> tuple[RiskModel, pd.DataFrame, np.ndarray]:
    """End-to-end: scenarios → features → labels → trained model.

    Args:
        scenarios: List of DisruptionScenario to train on.
        seed: RNG seed for label synthesis.
        config: Optional Config (defaults if None).
        model: Optional existing RiskModel to fit; creates new if None.

    Returns:
        (trained_model, feature_df, labels_array)
    """
    cfg = config or default_config()
    rng = np.random.default_rng(seed)
    feature_dfs = [features_for_scenario(s, cfg) for s in scenarios]
    df = pd.concat(feature_dfs, ignore_index=True)
    labels = synthesize_misconnect_labels(df, rng, cfg)
    if model is None:
        model = RiskModel(cfg)
    model.fit(df, labels)
    return model, df, labels


# Type stub for `Any`
from typing import Any  # noqa: E402
