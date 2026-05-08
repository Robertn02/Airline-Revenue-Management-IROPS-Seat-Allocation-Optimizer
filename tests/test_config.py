"""Tests for the configuration system."""
from __future__ import annotations

from reroute.core.config import (
    CalibrationConstants,
    Config,
    CostCoefficients,
    OperationalConstants,
    default_config,
)


def test_default_config_loadable():
    cfg = default_config()
    assert cfg.cost.alpha_yield == 1.0
    assert cfg.operational.mct_domestic_min == 35
    assert cfg.calibration.bts_late_fraction == 0.22


def test_yaml_roundtrip(tmp_path):
    cfg = default_config()
    fpath = tmp_path / "config.yaml"
    cfg.to_yaml(str(fpath))
    loaded = Config.from_yaml(str(fpath))
    assert loaded.cost.alpha_yield == cfg.cost.alpha_yield
    assert loaded.calibration.tier_distribution == cfg.calibration.tier_distribution
    assert loaded.operational.destinations == cfg.operational.destinations


def test_yield_lognormal_tuple_roundtrip(tmp_path):
    """YAML stores tuples as lists; ensure they're restored correctly."""
    cfg = default_config()
    fpath = tmp_path / "config.yaml"
    cfg.to_yaml(str(fpath))
    loaded = Config.from_yaml(str(fpath))
    assert isinstance(loaded.calibration.yield_lognormal["F"], tuple)
    assert loaded.calibration.yield_lognormal["F"] == cfg.calibration.yield_lognormal["F"]


def test_config_overrides():
    """Custom config should override defaults."""
    cfg = Config(cost=CostCoefficients(alpha_yield=2.0, beta_spill=0.5))
    assert cfg.cost.alpha_yield == 2.0
    assert cfg.cost.beta_spill == 0.5
    # Other components still default
    assert cfg.operational.mct_domestic_min == 35
