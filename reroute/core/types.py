"""Core data types for the Reroute system.

The types here form the API surface that the rest of the package operates on.
All are immutable-by-convention (frozen-style usage), JSON-serializable, and
designed to round-trip through file storage without loss.

Author: Phuc Nguyen
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


CabinClass = Literal["F", "Y+", "Y"]
TierClass = Literal["EXP", "PLT", "GLD", "REG"]
ConfidenceClass = Literal["H", "M", "L"]


@dataclass
class Flight:
    """A single scheduled flight leg with current inventory.

    Attributes:
        flight_id: Identifier (e.g. "AA1247"), unique within a scenario.
        origin: IATA origin airport code.
        destination: IATA destination airport code.
        sched_dep_min: Scheduled departure, minutes from scenario epoch.
        sched_arr_min: Scheduled arrival, minutes from scenario epoch.
        actual_dep_min: Actual / projected departure (for delayed flights).
        actual_arr_min: Actual / projected arrival.
        capacity_total: Total physical seat count (e.g., 172 for narrow-body).
        capacity_F: First/business cabin physical capacity.
        capacity_Yplus: Premium economy physical capacity.
        seats_open_F: Currently unsold F-cabin seats available for assignment.
        seats_open_Yplus: Currently unsold Y+ cabin seats.
        seats_open_Y: Currently unsold main cabin seats.
    """

    flight_id: str
    origin: str
    destination: str
    sched_dep_min: int
    sched_arr_min: int
    actual_dep_min: int
    actual_arr_min: int
    capacity_total: int
    capacity_F: int
    capacity_Yplus: int
    seats_open_F: int
    seats_open_Yplus: int
    seats_open_Y: int

    @property
    def delay_min(self) -> int:
        """Realized arrival delay in minutes (negative if early)."""
        return self.actual_arr_min - self.sched_arr_min

    @property
    def open_seats_total(self) -> int:
        """Total open seats across all cabins."""
        return self.seats_open_F + self.seats_open_Yplus + self.seats_open_Y

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Passenger:
    """A connecting passenger affected by an inbound delay.

    Attributes:
        pax_id: Anonymized identifier (synthetic, not a real PNR).
        name_initial: Initials only (e.g. "M. Chen") for display purposes.
        tier: Loyalty tier (EXP / PLT / GLD / REG).
        cabin: Originally booked cabin class (F / Y+ / Y).
        yield_usd: Fare value attributable to this leg, in USD.
        inbound_flight_id: The delayed flight bringing them to the hub.
        outbound_flight_id: Their originally booked outbound flight.
        sched_connection_min: Originally scheduled connection buffer (minutes).
        has_ssr: True if passenger has any Special Service Request.
        is_unaccompanied_minor: True if UM (extra handling required).
    """

    pax_id: str
    name_initial: str
    tier: TierClass
    cabin: CabinClass
    yield_usd: float
    inbound_flight_id: str
    outbound_flight_id: str
    sched_connection_min: int
    has_ssr: bool
    is_unaccompanied_minor: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DisruptionScenario:
    """One disruption event ready for allocation.

    Encapsulates a delayed inbound flight, the connecting passengers at risk,
    and the recovery flights available for reassignment.

    Attributes:
        scenario_id: Unique identifier for the scenario.
        hub: IATA hub code where the connection occurs.
        inbound_flight: The delayed flight.
        passengers: Connecting passengers from the inbound.
        recovery_flights: Outbound flights available for reaccomodation.
        timestamp_min: Decision-time clock value (typically inbound arrival).
        metadata: Free-form scenario metadata (delay, n_pax, etc.).
    """

    scenario_id: str
    hub: str
    inbound_flight: Flight
    passengers: list[Passenger]
    recovery_flights: list[Flight]
    timestamp_min: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def affected_passengers(self) -> list[Passenger]:
        """Backward-compat alias for `passengers`."""
        return self.passengers

    @property
    def total_open_seats(self) -> int:
        """Total open seats across all recovery flights."""
        return sum(f.open_seats_total for f in self.recovery_flights)

    @property
    def supply_demand_ratio(self) -> float:
        """Open seats divided by passenger count."""
        n = len(self.passengers)
        return self.total_open_seats / n if n > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "hub": self.hub,
            "inbound_flight": self.inbound_flight.to_dict(),
            "passengers": [p.to_dict() for p in self.passengers],
            "recovery_flights": [f.to_dict() for f in self.recovery_flights],
            "timestamp_min": self.timestamp_min,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DisruptionScenario:
        return cls(
            scenario_id=d["scenario_id"],
            hub=d["hub"],
            inbound_flight=Flight(**d["inbound_flight"]),
            passengers=[Passenger(**p) for p in d.get("passengers", d.get("affected_passengers", []))],
            recovery_flights=[Flight(**f) for f in d["recovery_flights"]],
            timestamp_min=d["timestamp_min"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class Assignment:
    """Outcome for a single passenger from an allocation run.

    Attributes:
        pax_id: Passenger this assignment refers to.
        flight_id: Recovery flight ID, or None if unassigned (misconnect).
        assigned_cabin: Cabin actually assigned, or None if unassigned.
        expected_cost: Expected cost contribution to objective (USD).
        binding_constraint: Short reason code explaining the outcome.
    """

    pax_id: str
    flight_id: Optional[str]
    assigned_cabin: Optional[CabinClass]
    expected_cost: float
    binding_constraint: str

    @property
    def is_misconnect(self) -> bool:
        return self.flight_id is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AllocationResult:
    """Output of an allocator (LP or baseline) for one scenario.

    Attributes:
        scenario_id: Scenario this result refers to.
        strategy: "lp" or "manual" — which allocator produced this.
        assignments: One per passenger, in input order.
        expected_loss: Total expected cost across the cohort (USD).
        n_misconnects: Count of unassigned passengers.
        n_assigned: Count of successfully assigned passengers.
        solve_time_ms: Wall-clock solve time in milliseconds.
        objective_breakdown: Cost components (yield/spill/harm/misconnect).
        feasible: True if the optimization completed successfully.
    """

    scenario_id: str
    strategy: str
    assignments: list[Assignment]
    expected_loss: float
    n_misconnects: int
    n_assigned: int
    solve_time_ms: float
    objective_breakdown: dict[str, float]
    feasible: bool = True

    # Backward-compat alias (older code uses total_expected_loss)
    @property
    def total_expected_loss(self) -> float:
        return self.expected_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "strategy": self.strategy,
            "assignments": [a.to_dict() for a in self.assignments],
            "expected_loss": self.expected_loss,
            "n_misconnects": self.n_misconnects,
            "n_assigned": self.n_assigned,
            "solve_time_ms": self.solve_time_ms,
            "objective_breakdown": self.objective_breakdown,
            "feasible": self.feasible,
        }
