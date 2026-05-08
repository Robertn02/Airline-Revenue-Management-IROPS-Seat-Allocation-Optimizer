"""Synthetic IROPS scenario generator.

Generates passenger itineraries, flight schedules, delay realizations,
and inventory snapshots for a simulated hub-and-spoke network during
disruption windows.

Calibrated to public BTS on-time performance statistics (2024) and
published academic distributions for connection time buffers
(Bratu & Barnhart 2006; Marla et al. 2012).

Why synthetic data:
    Real airline operational data is proprietary. Every published academic
    paper on this problem either uses synthetic data or works under an NDA
    that prevents code release. This generator makes assumptions explicit
    so they can be audited.

Author: Phuc Nguyen
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from reroute.core.config import Config, default_config
from reroute.core.types import (
    CabinClass,
    DisruptionScenario,
    Flight,
    Passenger,
    TierClass,
)


def _weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Sample a key from a {key: weight} dict, normalized to a probability."""
    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys])
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _sample_yield(
    rng: np.random.Generator,
    cabin: str,
    tier: str,
    config: Config,
) -> float:
    """Sample a fare value (USD) from the cabin/tier yield distribution."""
    mu, sigma = config.calibration.yield_lognormal[cabin]
    base = float(rng.lognormal(mean=mu, sigma=sigma))
    multiplier = config.calibration.tier_yield_multiplier[tier]
    return round(base * multiplier, 2)


def _sample_delay(rng: np.random.Generator, config: Config) -> int:
    """Sample a realized delay in minutes for a 'delayed' flight."""
    mean = config.calibration.bts_delay_mean_min
    sigma = config.calibration.bts_delay_lognormal_sigma
    mu = np.log(mean) - 0.5 * sigma**2
    return int(round(rng.lognormal(mean=mu, sigma=sigma)))


def _make_initials(rng: np.random.Generator) -> str:
    """Generate fake initials like 'M. Chen' for display only."""
    first = chr(int(rng.integers(65, 91)))
    last_first = chr(int(rng.integers(65, 91)))
    last_rest = "".join(chr(int(rng.integers(97, 123))) for _ in range(3))
    return f"{first}. {last_first}{last_rest}"


def _make_inbound_flight(
    rng: np.random.Generator,
    base_minute: int,
    delay_min: int,
    config: Config,
) -> Flight:
    """Construct the delayed inbound flight."""
    return Flight(
        flight_id=f"AA{int(rng.integers(1000, 9999))}",
        origin="LAX" if rng.random() < 0.5 else "SFO",
        destination=config.operational.hub_airport,
        sched_dep_min=base_minute - 180,
        sched_arr_min=base_minute,
        actual_dep_min=base_minute - 180 + delay_min,
        actual_arr_min=base_minute + delay_min,
        capacity_total=config.calibration.flight_capacity_total,
        capacity_F=config.calibration.flight_capacity_F,
        capacity_Yplus=config.calibration.flight_capacity_Yplus,
        seats_open_F=0,
        seats_open_Yplus=0,
        seats_open_Y=0,
    )


def _make_recovery_flights(
    rng: np.random.Generator,
    inbound: Flight,
    n_recovery: int,
    config: Config,
) -> list[Flight]:
    """Generate recovery flights departing AFTER inbound's actual arrival."""
    actual_arrival = inbound.actual_arr_min
    chosen_dests = rng.choice(
        config.operational.destinations, size=n_recovery, replace=False
    )
    recovery: list[Flight] = []
    for j, dest in enumerate(chosen_dests):
        offset = 45 + j * (30 + int(rng.integers(0, 25)))
        sched_dep = actual_arrival + offset
        sched_arr = sched_dep + 150 + int(rng.integers(0, 90))

        load_factor = float(rng.uniform(
            config.calibration.recovery_load_factor_low,
            config.calibration.recovery_load_factor_high,
        ))
        capacity = config.calibration.flight_capacity_total
        sold = int(capacity * load_factor)
        open_total = max(2, capacity - sold)
        open_F = int(rng.integers(0, 2))
        open_Yp = int(rng.integers(0, 3))
        open_Y = max(0, open_total - open_F - open_Yp)

        recovery.append(Flight(
            flight_id=f"AA{int(rng.integers(1000, 9999))}",
            origin=config.operational.hub_airport,
            destination=str(dest),
            sched_dep_min=sched_dep,
            sched_arr_min=sched_arr,
            actual_dep_min=sched_dep,
            actual_arr_min=sched_arr,
            capacity_total=capacity,
            capacity_F=config.calibration.flight_capacity_F,
            capacity_Yplus=config.calibration.flight_capacity_Yplus,
            seats_open_F=open_F,
            seats_open_Yplus=open_Yp,
            seats_open_Y=open_Y,
        ))
    return recovery


def _make_passengers(
    rng: np.random.Generator,
    inbound: Flight,
    recovery: list[Flight],
    n_pax: int,
    config: Config,
) -> list[Passenger]:
    """Generate connecting passengers.

    NOTE on label artifact: Tier and cabin are sampled INDEPENDENTLY of
    connection buffer, so misconnect probability has no inherent correlation
    with yield. This is the fix for the label-generator artifact in the
    earlier version.
    """
    passengers: list[Passenger] = []
    for _ in range(n_pax):
        tier: TierClass = _weighted_choice(rng, config.calibration.tier_distribution)  # type: ignore
        cabin: CabinClass = _weighted_choice(rng, config.calibration.cabin_distribution)  # type: ignore

        # Top-tier passengers slightly more often in premium cabins (small effect)
        if tier in ("EXP", "PLT") and rng.random() < 0.20:
            cabin = "F" if rng.random() < 0.40 else "Y+"

        yld = _sample_yield(rng, cabin, tier, config)

        # Connection buffer is sampled INDEPENDENTLY of tier/cabin/yield.
        # Real bookings have buffer driven by schedule, not by passenger value.
        sched_buffer = int(rng.integers(60, 180))

        # Pick outbound destination from recovery set
        outbound = recovery[int(rng.integers(0, len(recovery)))]

        # SSR / UM are rare
        has_ssr = bool(rng.random() < 0.04)
        is_um = bool(rng.random() < 0.005)

        passengers.append(Passenger(
            pax_id=f"PAX{int(rng.integers(100000, 999999))}",
            name_initial=_make_initials(rng),
            tier=tier,
            cabin=cabin,
            yield_usd=yld,
            inbound_flight_id=inbound.flight_id,
            outbound_flight_id=outbound.flight_id,
            sched_connection_min=sched_buffer,
            has_ssr=has_ssr,
            is_unaccompanied_minor=is_um,
        ))
    return passengers


def generate_scenario(
    rng: np.random.Generator,
    n_passengers: int = 12,
    n_recovery_flights: int = 3,
    base_minute: int = 0,
    force_delay_min: Optional[int] = None,
    scenario_id: Optional[str] = None,
    config: Optional[Config] = None,
) -> DisruptionScenario:
    """Generate one disruption scenario suitable for allocation.

    Args:
        rng: numpy random Generator (use `np.random.default_rng(seed)`).
        n_passengers: Number of connecting passengers affected.
        n_recovery_flights: Number of outbound options to model.
        base_minute: Time epoch (minutes) for the scenario.
        force_delay_min: If set, force this exact inbound delay (otherwise
            sampled from the delay distribution). Useful for reproducibility.
        scenario_id: Optional identifier; auto-generated if not provided.
        config: Optional Config; uses defaults if None.

    Returns:
        A DisruptionScenario with passengers + recovery flights ready to solve.
    """
    cfg = config or default_config()

    if scenario_id is None:
        scenario_id = f"SCN-{int(rng.integers(10000, 99999))}"

    delay = (
        force_delay_min if force_delay_min is not None
        else max(_sample_delay(rng, cfg), 60)
    )
    inbound = _make_inbound_flight(rng, base_minute, delay, cfg)
    recovery = _make_recovery_flights(rng, inbound, n_recovery_flights, cfg)
    passengers = _make_passengers(rng, inbound, recovery, n_passengers, cfg)

    return DisruptionScenario(
        scenario_id=scenario_id,
        hub=cfg.operational.hub_airport,
        inbound_flight=inbound,
        passengers=passengers,
        recovery_flights=recovery,
        timestamp_min=base_minute + delay,
        metadata={
            "delay_realized_min": delay,
            "n_affected": len(passengers),
            "n_recovery": len(recovery),
            "total_open_seats": sum(f.open_seats_total for f in recovery),
            "tier_breakdown": {
                t: sum(1 for p in passengers if p.tier == t)
                for t in ["EXP", "PLT", "GLD", "REG"]
            },
        },
    )


def generate_dataset(
    n_scenarios: int = 200,
    seed: int = 42,
    pax_range: tuple[int, int] = (8, 60),
    recovery_range: tuple[int, int] = (3, 5),
    config: Optional[Config] = None,
) -> list[DisruptionScenario]:
    """Generate a multi-scenario dataset for training or evaluation.

    Args:
        n_scenarios: Number of scenarios to generate.
        seed: RNG seed for reproducibility.
        pax_range: (min, max) inclusive bounds on passenger count per scenario.
        recovery_range: (min, max) inclusive bounds on recovery flights.
        config: Optional Config.

    Returns:
        List of DisruptionScenario objects, deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    scenarios: list[DisruptionScenario] = []
    for i in range(n_scenarios):
        n_pax = int(rng.integers(pax_range[0], pax_range[1] + 1))
        n_rec = int(rng.integers(recovery_range[0], recovery_range[1] + 1))
        force_delay = int(rng.integers(60, 240))
        scn = generate_scenario(
            rng,
            scenario_id=f"SCN-{i:05d}",
            n_passengers=n_pax,
            n_recovery_flights=n_rec,
            base_minute=i * 1440,
            force_delay_min=force_delay,
            config=config,
        )
        scenarios.append(scn)
    return scenarios


def make_scarce_dataset(
    n_scenarios: int = 100,
    seed: int = 99,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
    pax_range: tuple[int, int] = (15, 50),
    recovery_range: tuple[int, int] = (3, 6),
    config: Optional[Config] = None,
    max_attempts_multiplier: int = 20,
) -> list[DisruptionScenario]:
    """Generate a dataset filtered to a target supply/demand ratio range.

    The interesting optimization regime is when total open seats are between
    ~50% and ~150% of affected passengers — enough to require trade-offs
    but not so few that everyone misconnects.
    """
    rng = np.random.default_rng(seed)
    scenarios: list[DisruptionScenario] = []
    attempts = 0
    while len(scenarios) < n_scenarios and attempts < n_scenarios * max_attempts_multiplier:
        attempts += 1
        n_pax = int(rng.integers(pax_range[0], pax_range[1] + 1))
        n_rec = int(rng.integers(recovery_range[0], recovery_range[1] + 1))
        force_delay = int(rng.integers(75, 200))
        scn = generate_scenario(
            rng,
            scenario_id=f"SIM-{len(scenarios):05d}",
            n_passengers=n_pax,
            n_recovery_flights=n_rec,
            base_minute=len(scenarios) * 1440,
            force_delay_min=force_delay,
            config=config,
        )
        if ratio_min <= scn.supply_demand_ratio <= ratio_max:
            scenarios.append(scn)
    return scenarios


def save_dataset(scenarios: list[DisruptionScenario], path: str) -> None:
    """Save a dataset as newline-delimited JSON for portability."""
    with open(path, "w") as f:
        for scn in scenarios:
            f.write(json.dumps(scn.to_dict()) + "\n")


def load_dataset(path: str) -> list[DisruptionScenario]:
    """Load a dataset from newline-delimited JSON."""
    scenarios: list[DisruptionScenario] = []
    with open(path) as f:
        for line in f:
            scenarios.append(DisruptionScenario.from_dict(json.loads(line)))
    return scenarios
