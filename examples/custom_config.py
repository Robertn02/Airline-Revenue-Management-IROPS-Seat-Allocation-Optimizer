"""Custom configuration example.

Demonstrates how to override default cost coefficients to change the
optimizer's trade-offs.

Run from the project root:

    python examples/custom_config.py

Author: Phuc Nguyen
"""
from __future__ import annotations

import numpy as np

from reroute import (
    Allocator,
    Config,
    RiskModel,
    default_config,
    generate_scenario,
    manual_triage,
)
from reroute.core.config import CostCoefficients


def solve_with_coeffs(scenario, probs, coefficients: CostCoefficients):
    cfg = Config(cost=coefficients)
    return Allocator(cfg).solve(scenario, probs)


def main():
    # Generate one scenario we'll re-solve with different weights
    rng = np.random.default_rng(seed=7)
    scenario = generate_scenario(
        rng, n_passengers=30, n_recovery_flights=4, force_delay_min=120
    )
    model = RiskModel.load_default()
    probs = model.predict(scenario)

    print("=" * 72)
    print("Effect of changing cost coefficients on the same scenario")
    print("=" * 72)

    presets = [
        ("Default",                  CostCoefficients()),
        ("Aggressive spill avoid",   CostCoefficients(beta_spill=2.5)),
        ("Aggressive miss avoid",    CostCoefficients(lambda_miss=2.5)),
        ("Cheap labor (low harm)",   CostCoefficients(delta_harm=0.3)),
        ("Premium-protective",       CostCoefficients(beta_spill=2.0, delta_harm=2.5)),
    ]

    print(f"\n  {'Strategy':<28} {'Loss':>10} {'Misses':>8} {'Spill':>8} {'Yield':>8}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}")

    manual = manual_triage(scenario, probs)
    print(f"  {'Manual baseline':<28} ${manual.expected_loss:>9.0f} "
          f"{manual.n_misconnects:>8d} "
          f"${manual.objective_breakdown['spill']:>7.0f} "
          f"${manual.objective_breakdown['yield_dilution']:>7.0f}")

    for name, coefs in presets:
        result = solve_with_coeffs(scenario, probs, coefs)
        print(f"  {name:<28} ${result.expected_loss:>9.0f} "
              f"{result.n_misconnects:>8d} "
              f"${result.objective_breakdown['spill']:>7.0f} "
              f"${result.objective_breakdown['yield_dilution']:>7.0f}")

    print("\nNotice how 'aggressive spill avoid' produces near-zero spill cost")
    print("but may accept slightly higher misses or yield dilution as the")
    print("trade-off. The LP makes this trade-off explicit and tunable.")


if __name__ == "__main__":
    main()
