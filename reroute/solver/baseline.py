"""Baseline strategy: serial priority-queue triage.

Models how a gate agent currently handles disruption recovery — process
passengers in tier-then-yield order, assign each to the first feasible
recovery flight to their original destination, fall back to alternatives
or misconnect if all options are full.

This is intentionally not optimized; it represents the realistic baseline
the LP allocator competes against.

Author: Phuc Nguyen
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from reroute.core.config import Config, default_config
from reroute.core.types import (
    AllocationResult,
    Assignment,
    CabinClass,
    DisruptionScenario,
)
from reroute.solver.costs import (
    cabin_allowed_for_pax,
    harm_penalty,
    is_feasible_assignment,
    miss_cost,
    spill_cost,
    yield_dilution,
)


CABINS: list[CabinClass] = ["F", "Y+", "Y"]
TIER_PRIORITY = {"EXP": 0, "PLT": 1, "GLD": 2, "REG": 3}


def manual_triage(
    scenario: DisruptionScenario,
    misconnect_probs: np.ndarray,
    config: Optional[Config] = None,
) -> AllocationResult:
    """Run serial priority-queue triage on a scenario.

    Algorithm:
        1. Sort pax by tier (EXP > PLT > GLD > REG), then by yield desc.
        2. For each pax in order, try to assign to original outbound first.
        3. If full, try alternative recovery flights in departure-time order.
        4. If all full, misconnect.

    Args:
        scenario: The disruption to triage.
        misconnect_probs: Per-passenger misconnect probabilities.
        config: Optional Config.

    Returns:
        AllocationResult with strategy="manual".
    """
    cfg = config or default_config()
    t0 = time.perf_counter()
    pax = scenario.passengers
    flights = scenario.recovery_flights
    inbound = scenario.inbound_flight

    # Priority order
    order = sorted(
        range(len(pax)),
        key=lambda i: (TIER_PRIORITY[pax[i].tier], -pax[i].yield_usd),
    )

    flight_seats = {
        f.flight_id: {
            "F": f.seats_open_F,
            "Y+": f.seats_open_Yplus,
            "Y": f.seats_open_Y,
        } for f in flights
    }

    assignments: list[Assignment] = []
    breakdown = {"yield_dilution": 0.0, "spill": 0.0, "harm": 0.0, "misconnect": 0.0}

    for i in order:
        p = pax[i]
        # Try original outbound first; otherwise sort by earliest departure
        candidates = sorted(
            flights,
            key=lambda f: (
                0 if f.flight_id == p.outbound_flight_id else 1,
                f.sched_dep_min,
            )
        )
        assigned = False
        for f in candidates:
            if not is_feasible_assignment(p, f, inbound, cfg):
                continue
            # Find usable cabin: prefer same cabin as original, fall back
            chosen_cab: Optional[CabinClass] = None
            preferred_order = _preferred_cabin_order(p)
            for cab in preferred_order:
                if not cabin_allowed_for_pax(p, cab):
                    continue
                if flight_seats[f.flight_id][cab] > 0:
                    chosen_cab = cab
                    break
            if chosen_cab is None:
                continue
            # Assign
            flight_seats[f.flight_id][chosen_cab] -= 1
            yld = yield_dilution(p, f)
            spl = spill_cost(p, f, chosen_cab)
            hrm = harm_penalty(p)
            cost_val = (
                cfg.cost.alpha_yield * yld
                + cfg.cost.beta_spill * spl
                + cfg.cost.delta_harm * hrm
            )
            assignments.append(Assignment(
                pax_id=p.pax_id,
                flight_id=f.flight_id,
                assigned_cabin=chosen_cab,
                expected_cost=cost_val,
                binding_constraint="manual_serial",
            ))
            breakdown["yield_dilution"] += yld
            breakdown["spill"] += spl
            breakdown["harm"] += hrm
            assigned = True
            break

        if not assigned:
            mc = misconnect_probs[i] * (
                miss_cost(p, cfg) + cfg.cost.delta_harm * harm_penalty(p)
            )
            assignments.append(Assignment(
                pax_id=p.pax_id,
                flight_id=None,
                assigned_cabin=None,
                expected_cost=mc,
                binding_constraint="capacity_exhausted_manual",
            ))
            breakdown["misconnect"] += mc

    # Re-sort to original passenger order
    pax_order = {p.pax_id: i for i, p in enumerate(pax)}
    assignments.sort(key=lambda a: pax_order[a.pax_id])

    solve_ms = (time.perf_counter() - t0) * 1000
    n_misc = sum(1 for a in assignments if a.is_misconnect)
    total = sum(a.expected_cost for a in assignments)

    return AllocationResult(
        scenario_id=scenario.scenario_id,
        strategy="manual",
        assignments=assignments,
        expected_loss=round(total, 2),
        n_misconnects=n_misc,
        n_assigned=len(pax) - n_misc,
        solve_time_ms=round(solve_ms, 2),
        objective_breakdown={k: round(v, 2) for k, v in breakdown.items()},
        feasible=True,
    )


def _preferred_cabin_order(pax) -> list[CabinClass]:
    """Try same cabin first, then nearest-up, then nearest-down."""
    if pax.cabin == "F":
        return ["F", "Y+", "Y"]
    if pax.cabin == "Y+":
        return ["Y+", "F", "Y"]
    return ["Y", "Y+", "F"]
