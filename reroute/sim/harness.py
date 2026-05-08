"""Simulation harness — runs both strategies across scenarios and aggregates.

Author: Phuc Nguyen
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from reroute.core.config import Config, default_config
from reroute.core.logging import get_logger
from reroute.core.types import AllocationResult, DisruptionScenario
from reroute.model.risk import RiskModel
from reroute.solver.baseline import manual_triage
from reroute.solver.lp import Allocator

logger = get_logger(__name__)


@dataclass
class ScenarioComparison:
    """Per-scenario record comparing manual vs LP outcomes."""

    scenario_id: str
    n_pax: int
    n_recovery: int
    total_open_seats: int
    delay_min: int
    seat_demand_ratio: float
    manual_loss: float
    manual_misconnects: int
    manual_solve_ms: float
    lp_loss: float
    lp_misconnects: int
    lp_solve_ms: float
    delta_dollars: float
    delta_pct: float


class SimulationHarness:
    """Orchestrates batch simulation across many scenarios."""

    def __init__(
        self,
        risk_model: RiskModel,
        allocator: Optional[Allocator] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or default_config()
        self.risk_model = risk_model
        self.allocator = allocator or Allocator(self.config)

    def run_one(
        self,
        scenario: DisruptionScenario,
    ) -> tuple[AllocationResult, AllocationResult, np.ndarray]:
        """Run both strategies on a single scenario.

        Returns:
            (manual_result, lp_result, misconnect_probs)
        """
        probs = self.risk_model.predict(scenario)
        manual = manual_triage(scenario, probs, self.config)
        lp = self.allocator.solve(scenario, probs)
        return manual, lp, probs

    def run_batch(
        self,
        scenarios: list[DisruptionScenario],
        verbose: bool = False,
    ) -> list[ScenarioComparison]:
        """Run both strategies across scenarios; return comparisons."""
        results: list[ScenarioComparison] = []
        t0 = time.perf_counter()
        for idx, scn in enumerate(scenarios):
            manual, lp, _ = self.run_one(scn)
            if not lp.feasible:
                logger.warning(f"Skipping infeasible {scn.scenario_id}")
                continue
            delta = manual.expected_loss - lp.expected_loss
            delta_pct = (
                100 * delta / manual.expected_loss
                if manual.expected_loss > 0 else 0.0
            )
            results.append(ScenarioComparison(
                scenario_id=scn.scenario_id,
                n_pax=len(scn.passengers),
                n_recovery=len(scn.recovery_flights),
                total_open_seats=scn.total_open_seats,
                delay_min=scn.metadata.get("delay_realized_min", 0),
                seat_demand_ratio=round(scn.supply_demand_ratio, 3),
                manual_loss=manual.expected_loss,
                manual_misconnects=manual.n_misconnects,
                manual_solve_ms=manual.solve_time_ms,
                lp_loss=lp.expected_loss,
                lp_misconnects=lp.n_misconnects,
                lp_solve_ms=lp.solve_time_ms,
                delta_dollars=round(delta, 2),
                delta_pct=round(delta_pct, 2),
            ))
            if verbose and (idx + 1) % 10 == 0:
                logger.info(f"  ...{idx + 1}/{len(scenarios)} scenarios processed")
        elapsed = time.perf_counter() - t0
        logger.info(f"Batch complete: {len(results)} scenarios in {elapsed:.1f}s")
        return results

    @staticmethod
    def summarize(results: list[ScenarioComparison]) -> dict:
        """Aggregate statistics across a batch."""
        df = pd.DataFrame([asdict(r) for r in results])
        if df.empty:
            return {"n_scenarios": 0}
        return {
            "n_scenarios": len(df),
            "total_passengers": int(df["n_pax"].sum()),
            "manual_total_loss": round(df["manual_loss"].sum(), 2),
            "lp_total_loss": round(df["lp_loss"].sum(), 2),
            "total_delta_dollars": round(df["delta_dollars"].sum(), 2),
            "mean_delta_per_scenario": round(df["delta_dollars"].mean(), 2),
            "median_delta_pct": round(df["delta_pct"].median(), 2),
            "manual_total_misconnects": int(df["manual_misconnects"].sum()),
            "lp_total_misconnects": int(df["lp_misconnects"].sum()),
            "mean_solve_ms_lp": round(df["lp_solve_ms"].mean(), 2),
            "p95_solve_ms_lp": round(df["lp_solve_ms"].quantile(0.95), 2),
        }

    @staticmethod
    def save_results(
        results: list[ScenarioComparison],
        summary: dict,
        out_dir: str | Path,
    ) -> None:
        """Persist comparison and summary to disk."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "simulation_results.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        with open(out / "summary_stats.json", "w") as f:
            json.dump(summary, f, indent=2)
        pd.DataFrame([asdict(r) for r in results]).to_csv(
            out / "comparison.csv", index=False
        )
        logger.info(f"Saved {len(results)} results to {out}/")
