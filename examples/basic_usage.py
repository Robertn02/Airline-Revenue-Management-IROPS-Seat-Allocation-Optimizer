"""Basic Reroute library usage.

Run from the project root:

    python examples/basic_usage.py

Demonstrates:
    - Generating a single disruption scenario
    - Loading the trained risk model
    - Running both manual triage and LP allocation
    - Inspecting the per-passenger results

Author: Phuc Nguyen
"""
from __future__ import annotations

import numpy as np

from reroute import (
    Allocator,
    RiskModel,
    generate_scenario,
    manual_triage,
)


def main():
    print("=" * 60)
    print("Reroute · basic usage example")
    print("=" * 60)

    # 1. Generate a single scenario with realistic seat scarcity
    rng = np.random.default_rng(seed=11)
    scenario = generate_scenario(
        rng,
        n_passengers=30,
        n_recovery_flights=3,
        force_delay_min=120,
    )
    print(f"\nScenario {scenario.scenario_id}")
    print(f"  Inbound: {scenario.inbound_flight.flight_id} from "
          f"{scenario.inbound_flight.origin}, "
          f"{scenario.metadata['delay_realized_min']}min late")
    print(f"  {len(scenario.passengers)} affected passengers")
    print(f"  {len(scenario.recovery_flights)} recovery options")
    print(f"  {scenario.total_open_seats} total open seats "
          f"(supply/demand = {scenario.supply_demand_ratio:.2f})")

    # 2. Load the risk model (trains a fresh one if none cached)
    print("\nLoading risk model...")
    model = RiskModel.load_default()
    probs = model.predict(scenario)
    print(f"  Mean misconnect probability: {probs.mean():.2%}")
    print(f"  High-confidence cases: "
          f"{sum(1 for p in probs if model.confidence_class(p) == 'H')}/"
          f"{len(probs)}")

    # 3. Run both strategies
    print("\nRunning both strategies...")
    manual = manual_triage(scenario, probs)
    lp = Allocator().solve(scenario, probs)

    print(f"\n  {'Strategy':<12} {'Loss':>12} {'Misses':>8} {'Solve':>10}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 10}")
    print(f"  {'Manual':<12} ${manual.expected_loss:>11.2f} "
          f"{manual.n_misconnects:>8d} {manual.solve_time_ms:>7.1f} ms")
    print(f"  {'Reroute LP':<12} ${lp.expected_loss:>11.2f} "
          f"{lp.n_misconnects:>8d} {lp.solve_time_ms:>7.1f} ms")

    delta = manual.expected_loss - lp.expected_loss
    delta_pct = 100 * delta / manual.expected_loss
    print(f"\n  Saved: ${delta:.2f} ({delta_pct:.1f}% reduction)")

    # 4. Inspect a few specific assignments
    print("\nFirst 5 LP assignments:")
    for i, (p, a) in enumerate(zip(scenario.passengers[:5], lp.assignments[:5])):
        outcome = (
            f"misconnect" if a.is_misconnect
            else f"→ {a.flight_id} {a.assigned_cabin}"
        )
        print(f"  {p.tier:>3} {p.cabin:>2} ${p.yield_usd:>5.0f}  "
              f"p={probs[i]:.2f}  →  {outcome}")


if __name__ == "__main__":
    main()
