"""Cost computation helpers.

Pure functions used by both the LP allocator and the manual baseline so
they share the same accounting model.

Author: Phuc Nguyen
"""
from __future__ import annotations

from typing import Optional

from reroute.core.config import Config, default_config
from reroute.core.types import CabinClass, Flight, Passenger


def cabin_rank(cabin: str) -> int:
    """Higher number = better cabin (F=3, Y+=2, Y=1)."""
    return {"F": 3, "Y+": 2, "Y": 1}.get(cabin, 0)


def yield_dilution(pax: Passenger, flight: Flight) -> float:
    """Yield dilution if passenger reassigned to flight.

    Industry rule of thumb: ~12% of fare for same-day rebook within network.
    A real implementation would compute this from ATPCO fare basis rules.
    """
    return float(pax.yield_usd) * 0.12


def spill_cost(pax: Passenger, flight: Flight, assigned_cabin: CabinClass) -> float:
    """Cost of cabin downgrade (or 0 if same/upgraded)."""
    orig_rank = cabin_rank(pax.cabin)
    new_rank = cabin_rank(assigned_cabin)
    if new_rank >= orig_rank:
        return 0.0
    steps = orig_rank - new_rank
    return float(pax.yield_usd) * 0.25 * steps


def harm_penalty(pax: Passenger) -> float:
    """Penalty for mishandling SSR or unaccompanied minor cases."""
    if pax.is_unaccompanied_minor:
        return 800.0
    if pax.has_ssr:
        return 200.0
    return 0.0


def miss_cost(pax: Passenger, config: Optional[Config] = None) -> float:
    """Full cost of a misconnect (rebooking + service recovery + reputation)."""
    cfg = config or default_config()
    return float(pax.yield_usd) * 0.65 + cfg.cost.miss_fixed_cost_usd


def is_feasible_assignment(
    pax: Passenger,
    flight: Flight,
    inbound: Flight,
    config: Optional[Config] = None,
) -> bool:
    """Hard feasibility check (MCT + extended handling for SSR/UM)."""
    cfg = config or default_config()
    available = flight.sched_dep_min - inbound.actual_arr_min
    if available < cfg.operational.mct_domestic_min:
        return False
    needs_extended = pax.has_ssr or pax.is_unaccompanied_minor
    if needs_extended and available < cfg.operational.ssr_handling_min:
        return False
    return True


def cabin_allowed_for_pax(pax: Passenger, cabin: CabinClass) -> bool:
    """Loyalty floor: top-tier passengers cannot drop more than 1 cabin."""
    orig_rank = cabin_rank(pax.cabin)
    new_rank = cabin_rank(cabin)
    if pax.tier in ("EXP", "PLT") and (orig_rank - new_rank) > 1:
        return False
    return True
