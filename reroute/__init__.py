"""
Reroute: cohort-level seat allocation for airline disruption recovery.

A constrained-optimization approach to the irregular operations (IROPS)
seat allocation problem. Combines a calibrated misconnect risk model with
a linear-programming allocator that minimizes expected revenue loss across
the affected passenger cohort while respecting cabin, MCT, and loyalty
constraints.

Quick start:

    >>> from reroute import RiskModel, Allocator, generate_scenario
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> scenario = generate_scenario(rng, n_passengers=20, n_recovery_flights=4)
    >>> model = RiskModel.load_default()
    >>> probs = model.predict(scenario)
    >>> result = Allocator().solve(scenario, probs)
    >>> print(f"Allocated {result.n_assigned}/{len(scenario.passengers)} pax")
    >>> print(f"Expected loss: ${result.expected_loss:.2f}")

See the README and docs/ for detailed usage.
"""

from reroute.core.types import (
    Flight,
    Passenger,
    DisruptionScenario,
    Assignment,
    AllocationResult,
)
from reroute.core.config import Config, default_config
from reroute.model.risk import RiskModel
from reroute.solver.lp import Allocator
from reroute.solver.baseline import manual_triage
from reroute.sim.generator import generate_scenario, generate_dataset
from reroute.sim.harness import SimulationHarness, ScenarioComparison

__version__ = "0.1.0"
__all__ = [
    "Flight",
    "Passenger",
    "DisruptionScenario",
    "Assignment",
    "AllocationResult",
    "Config",
    "default_config",
    "RiskModel",
    "Allocator",
    "manual_triage",
    "generate_scenario",
    "generate_dataset",
    "SimulationHarness",
    "ScenarioComparison",
]
