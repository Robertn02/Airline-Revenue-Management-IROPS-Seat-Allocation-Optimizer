"""Tests for the synthetic data generator."""
from __future__ import annotations

import numpy as np
import pytest

from reroute import default_config
from reroute.core.types import DisruptionScenario
from reroute.sim.generator import (
    generate_dataset,
    generate_scenario,
    load_dataset,
    make_scarce_dataset,
    save_dataset,
)


def test_scenario_basic_structure():
    rng = np.random.default_rng(42)
    scn = generate_scenario(rng, n_passengers=12, n_recovery_flights=3, force_delay_min=90)
    assert isinstance(scn, DisruptionScenario)
    assert len(scn.passengers) == 12
    assert len(scn.recovery_flights) == 3
    assert scn.metadata["delay_realized_min"] == 90
    assert scn.inbound_flight.actual_arr_min > scn.inbound_flight.sched_arr_min


def test_recovery_flights_after_inbound_arrival():
    """Recovery flights must depart AFTER the delayed inbound arrives."""
    rng = np.random.default_rng(99)
    for _ in range(50):
        scn = generate_scenario(
            rng, n_passengers=20, n_recovery_flights=4,
            force_delay_min=int(rng.integers(60, 240)),
        )
        actual_arr = scn.inbound_flight.actual_arr_min
        for f in scn.recovery_flights:
            assert f.sched_dep_min > actual_arr


def test_realistic_load_factors():
    """Recovery flights should have <25 mean open seats (peak-bank realism)."""
    rng = np.random.default_rng(7)
    open_seats = []
    for _ in range(50):
        scn = generate_scenario(rng, n_passengers=20, n_recovery_flights=3, force_delay_min=90)
        for f in scn.recovery_flights:
            open_seats.append(f.open_seats_total)
    assert np.mean(open_seats) < 25


def test_passenger_initials_no_real_names():
    """No passenger should have real-looking names — only initials."""
    rng = np.random.default_rng(0)
    scn = generate_scenario(rng, n_passengers=30, n_recovery_flights=3)
    for p in scn.passengers:
        parts = p.name_initial.split(". ")
        assert len(parts) == 2
        assert len(parts[0]) == 1
        assert len(parts[1]) == 4


def test_tier_distribution_realistic():
    """Generated tier mix approximates configured distribution."""
    cfg = default_config()
    scns = generate_dataset(n_scenarios=200, seed=42)
    all_tiers = [p.tier for s in scns for p in s.passengers]
    n_total = len(all_tiers)
    for tier, expected in cfg.calibration.tier_distribution.items():
        observed = all_tiers.count(tier) / n_total
        assert abs(observed - expected) < 0.05, f"{tier}: expected {expected:.2f}, got {observed:.2f}"


def test_yield_positive():
    rng = np.random.default_rng(1)
    scn = generate_scenario(rng, n_passengers=30, n_recovery_flights=3)
    for p in scn.passengers:
        assert p.yield_usd > 0


def test_dataset_size():
    scns = generate_dataset(n_scenarios=10, seed=1)
    assert len(scns) == 10


def test_save_and_load_roundtrip(tmp_path):
    scns = generate_dataset(n_scenarios=5, seed=7)
    fpath = tmp_path / "test_data.jsonl"
    save_dataset(scns, str(fpath))
    loaded = load_dataset(str(fpath))
    assert len(loaded) == len(scns)
    assert loaded[0].scenario_id == scns[0].scenario_id
    assert loaded[0].passengers[0].yield_usd == scns[0].passengers[0].yield_usd


def test_seed_reproducibility():
    s1 = generate_dataset(n_scenarios=5, seed=42)
    s2 = generate_dataset(n_scenarios=5, seed=42)
    assert s1[0].scenario_id == s2[0].scenario_id
    assert s1[0].passengers[0].yield_usd == s2[0].passengers[0].yield_usd


def test_scarce_dataset_filter():
    """make_scarce_dataset should yield scenarios within the supply/demand band."""
    scns = make_scarce_dataset(n_scenarios=30, seed=42, ratio_min=0.5, ratio_max=1.5)
    for s in scns:
        assert 0.5 <= s.supply_demand_ratio <= 1.5


def test_scenario_serialization_roundtrip():
    rng = np.random.default_rng(7)
    scn = generate_scenario(rng, n_passengers=10, n_recovery_flights=3)
    d = scn.to_dict()
    reconstructed = DisruptionScenario.from_dict(d)
    assert reconstructed.scenario_id == scn.scenario_id
    assert len(reconstructed.passengers) == len(scn.passengers)
    assert reconstructed.passengers[0].yield_usd == scn.passengers[0].yield_usd
