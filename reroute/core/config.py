"""Configuration system for Reroute.

All tunable parameters — cost coefficients, calibration constants, MCT values,
yield distributions — live here. Configs can be loaded from YAML or built
programmatically. The default config matches the calibrations documented in
docs/DATA_SOURCES.md.

Author: Phuc Nguyen
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CostCoefficients:
    """Weights on the components of the LP objective function.

    The objective minimizes:
        Σ x_ijc · (alpha · yield_dilution + beta · spill + delta · harm)
        + Σ z_i · p_i · lambda · (miss_cost + delta · harm)

    Tuning these changes the trade-offs the optimizer makes:
    - Higher beta → more aggressive avoidance of cabin downgrades.
    - Higher lambda → more aggressive avoidance of misconnects.
    - Higher delta → extra protection for SSR / unaccompanied minor pax.
    """

    alpha_yield: float = 1.0
    beta_spill: float = 0.85
    delta_harm: float = 1.5
    lambda_miss: float = 1.0
    miss_fixed_cost_usd: float = 250.0


@dataclass
class CalibrationConstants:
    """Distributions used by the synthetic data generator.

    Calibrated to public BTS 2024 on-time performance data and typical
    industry-published distributions for tier mix and cabin mix. See
    docs/DATA_SOURCES.md for citations.
    """

    # BTS 2024 on-time performance
    bts_late_fraction: float = 0.22
    bts_delay_mean_min: float = 50.0
    bts_delay_lognormal_sigma: float = 0.85

    # Loyalty tier distribution (typical major US carrier)
    tier_distribution: dict[str, float] = field(default_factory=lambda: {
        "EXP": 0.04, "PLT": 0.09, "GLD": 0.18, "REG": 0.69
    })

    # Cabin distribution for narrow-body domestic
    cabin_distribution: dict[str, float] = field(default_factory=lambda: {
        "F": 0.08, "Y+": 0.18, "Y": 0.74
    })

    # Yield lognormal parameters (mu, sigma) per cabin
    yield_lognormal: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "F": (7.45, 0.32),
        "Y+": (6.60, 0.35),
        "Y": (5.95, 0.40),
    })

    # Tier yield premium multipliers (top tiers fly more / higher fares)
    tier_yield_multiplier: dict[str, float] = field(default_factory=lambda: {
        "EXP": 1.45, "PLT": 1.25, "GLD": 1.10, "REG": 1.00
    })

    # Recovery flight load factor range (peak-bank realistic)
    recovery_load_factor_low: float = 0.92
    recovery_load_factor_high: float = 0.98

    # Per-flight capacity (typical narrow-body)
    flight_capacity_total: int = 172
    flight_capacity_F: int = 12
    flight_capacity_Yplus: int = 24


@dataclass
class OperationalConstants:
    """Operational rules — MCT, cabin handling, etc."""

    # Minimum Connection Time (minutes) for domestic transfers at major US hubs
    mct_domestic_min: int = 35
    # Extended handling time required for SSR / unaccompanied minors
    ssr_handling_min: int = 60
    # Network destinations the synthetic generator draws from
    hub_airport: str = "DFW"
    destinations: list[str] = field(default_factory=lambda: [
        "TPA", "BOS", "SEA", "LAX", "ATL", "JFK", "DEN", "MCO"
    ])


@dataclass
class ModelHyperparams:
    """Hyperparameters for the gradient boosted risk estimator."""

    n_estimators: int = 200
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_child_samples: int = 20
    test_size: float = 0.2
    calibration_cv_folds: int = 3
    random_seed: int = 42


@dataclass
class Config:
    """Top-level configuration aggregating all subsystems."""

    cost: CostCoefficients = field(default_factory=CostCoefficients)
    calibration: CalibrationConstants = field(default_factory=CalibrationConstants)
    operational: OperationalConstants = field(default_factory=OperationalConstants)
    model: ModelHyperparams = field(default_factory=ModelHyperparams)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Reconstruct Config from a nested dict."""
        cost = CostCoefficients(**d.get("cost", {}))
        cal_d = d.get("calibration", {})
        # Tuple round-trip for yield_lognormal
        if "yield_lognormal" in cal_d:
            cal_d["yield_lognormal"] = {
                k: tuple(v) for k, v in cal_d["yield_lognormal"].items()
            }
        cal = CalibrationConstants(**cal_d)
        op = OperationalConstants(**d.get("operational", {}))
        mdl = ModelHyperparams(**d.get("model", {}))
        return cls(cost=cost, calibration=cal, operational=op, model=mdl)


def default_config() -> Config:
    """Return the default Config (calibrated to public BTS data)."""
    return Config()
