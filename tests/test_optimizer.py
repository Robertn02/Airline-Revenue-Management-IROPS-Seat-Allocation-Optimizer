"""Tests for the LP allocation optimizer."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from reroute import Allocator, manual_triage
from reroute.core.types import Flight, Passenger
from reroute.model.risk import features_for_scenario, train_from_scenarios
from reroute.sim.generator import generate_dataset, generate_scenario
from reroute.solver.costs import (
    cabin_allowed_for_pax,
    cabin_rank,
    spill_cost,
    yield_dilution,
)


@pytest.fixture(scope="module")
def trained_model():
    scns = generate_dataset(n_scenarios=100, seed=42)
    model, _, _ = train_from_scenarios(scns)
    return model


def test_optimizer_solves(trained_model):
    rng = np.random.default_rng(1)
    scn = generate_scenario(rng, n_passengers=15, n_recovery_flights=3, force_delay_min=90)
    probs = trained_model.predict(scn)
    result = Allocator().solve(scn, probs)
    assert result.feasible
    assert len(result.assignments) == len(scn.passengers)


def test_capacity_respected(trained_model):
    """No flight should be over capacity in any cabin."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        scn = generate_scenario(
            rng, n_passengers=int(rng.integers(15, 40)),
            n_recovery_flights=int(rng.integers(3, 5)),
            force_delay_min=int(rng.integers(80, 200)),
        )
        probs = trained_model.predict(scn)
        result = Allocator().solve(scn, probs)
        if not result.feasible:
            continue
        used = defaultdict(int)
        for a in result.assignments:
            if a.flight_id and a.assigned_cabin:
                used[(a.flight_id, a.assigned_cabin)] += 1
        for f in scn.recovery_flights:
            assert used[(f.flight_id, "F")] <= f.seats_open_F
            assert used[(f.flight_id, "Y+")] <= f.seats_open_Yplus
            assert used[(f.flight_id, "Y")] <= f.seats_open_Y


def test_loyalty_floor_respected(trained_model):
    """Top-tier passengers shouldn't be downgraded more than 1 cabin."""
    rng = np.random.default_rng(4)
    for _ in range(15):
        scn = generate_scenario(rng, n_passengers=25, n_recovery_flights=3, force_delay_min=120)
        probs = trained_model.predict(scn)
        result = Allocator().solve(scn, probs)
        if not result.feasible:
            continue
        for a in result.assignments:
            if a.flight_id is None or a.assigned_cabin is None:
                continue
            p = next(p for p in scn.passengers if p.pax_id == a.pax_id)
            if p.tier in ("EXP", "PLT"):
                drop = cabin_rank(p.cabin) - cabin_rank(a.assigned_cabin)
                assert drop <= 1


def test_lp_dominates_manual_in_aggregate(trained_model):
    """Across many scenarios, LP total cost should be lower."""
    rng = np.random.default_rng(5)
    lp_total = 0.0
    manual_total = 0.0
    n_run = 0
    allocator = Allocator()
    for _ in range(50):
        scn = generate_scenario(
            rng, n_passengers=int(rng.integers(15, 40)),
            n_recovery_flights=3,
            force_delay_min=int(rng.integers(80, 180)),
        )
        probs = trained_model.predict(scn)
        manual = manual_triage(scn, probs)
        lp = allocator.solve(scn, probs)
        if lp.feasible:
            lp_total += lp.expected_loss
            manual_total += manual.expected_loss
            n_run += 1
    assert n_run >= 30
    assert lp_total < manual_total
    pct = 100 * (manual_total - lp_total) / manual_total
    assert pct >= 5, f"LP improvement only {pct:.1f}% — expected ≥5%"


def test_solve_time_reasonable(trained_model):
    rng = np.random.default_rng(6)
    scn = generate_scenario(rng, n_passengers=50, n_recovery_flights=5, force_delay_min=150)
    probs = trained_model.predict(scn)
    result = Allocator().solve(scn, probs)
    assert result.solve_time_ms < 100  # generous bound for slow CI


def test_cost_helpers():
    p = Passenger(
        pax_id="X1", name_initial="A. Bcde", tier="EXP", cabin="F",
        yield_usd=2000.0, inbound_flight_id="AA1", outbound_flight_id="AA2",
        sched_connection_min=120, has_ssr=False, is_unaccompanied_minor=False,
    )
    f = Flight(
        flight_id="AA2", origin="DFW", destination="LAX",
        sched_dep_min=100, sched_arr_min=300,
        actual_dep_min=100, actual_arr_min=300,
        capacity_total=172, capacity_F=12, capacity_Yplus=24,
        seats_open_F=2, seats_open_Yplus=3, seats_open_Y=10,
    )
    assert 200 < yield_dilution(p, f) < 280
    assert spill_cost(p, f, "F") == 0
    assert spill_cost(p, f, "Y+") > 0
    assert cabin_allowed_for_pax(p, "F")
    assert cabin_allowed_for_pax(p, "Y+")
    assert not cabin_allowed_for_pax(p, "Y")  # EXP from F can't drop to Y


def test_allocation_result_serialization(trained_model):
    rng = np.random.default_rng(0)
    scn = generate_scenario(rng, n_passengers=10, n_recovery_flights=3)
    probs = trained_model.predict(scn)
    result = Allocator().solve(scn, probs)
    d = result.to_dict()
    assert d["scenario_id"] == scn.scenario_id
    assert d["strategy"] == "lp"
    assert "assignments" in d
