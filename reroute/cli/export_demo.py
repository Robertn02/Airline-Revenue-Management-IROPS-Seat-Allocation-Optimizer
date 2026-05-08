"""Export per-scenario detail to JSON for the web demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from reroute.core.config import Config, default_config
from reroute.core.logging import get_logger
from reroute.model.risk import RiskModel
from reroute.sim.generator import generate_dataset, make_scarce_dataset
from reroute.sim.harness import SimulationHarness
from reroute.solver.lp import Allocator

logger = get_logger(__name__)


def run_export(
    n_scenarios: int = 12,
    model_path: str = "results/model.pkl",
    output_path: str = "results/scenarios_for_demo.json",
    config: Optional[Config] = None,
) -> None:
    """Generate and export demo scenarios to JSON.

    Each scenario in the output includes full passenger detail and both
    manual and LP allocation results, suitable for the web demo to render
    without needing to run a backend.
    """
    cfg = config or default_config()

    if not Path(model_path).exists():
        logger.warning(f"No model at {model_path} — training fresh")
        from reroute.model.risk import train_from_scenarios
        train_scns = generate_dataset(n_scenarios=200, seed=42, config=cfg)
        risk_model, _, _ = train_from_scenarios(train_scns, config=cfg)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        risk_model.save(model_path)
    else:
        risk_model = RiskModel.load(model_path)

    logger.info(f"Generating {n_scenarios} demo scenarios...")
    eval_scns = make_scarce_dataset(n_scenarios=n_scenarios, seed=99, config=cfg)

    harness = SimulationHarness(risk_model, config=cfg)
    output: list[dict] = []

    for scn in eval_scns:
        manual, lp, probs = harness.run_one(scn)
        if not lp.feasible:
            continue

        manual_lookup = {a.pax_id: a for a in manual.assignments}
        lp_lookup = {a.pax_id: a for a in lp.assignments}

        passengers = []
        for i, p in enumerate(scn.passengers):
            m_a = manual_lookup[p.pax_id]
            l_a = lp_lookup[p.pax_id]
            passengers.append({
                "pax_id": p.pax_id,
                "name": p.name_initial,
                "tier": p.tier,
                "cabin": p.cabin,
                "yield_usd": round(p.yield_usd),
                "buffer_min": p.sched_connection_min,
                "has_ssr": p.has_ssr,
                "is_um": p.is_unaccompanied_minor,
                "misconnect_prob": round(float(probs[i]), 3),
                "confidence": risk_model.confidence_class(float(probs[i])),
                "manual": {
                    "flight": m_a.flight_id or "MISCONNECT",
                    "cabin": m_a.assigned_cabin,
                    "cost": round(m_a.expected_cost, 2),
                },
                "lp": {
                    "flight": l_a.flight_id or "MISCONNECT",
                    "cabin": l_a.assigned_cabin,
                    "cost": round(l_a.expected_cost, 2),
                },
            })

        output.append({
            "scenario_id": scn.scenario_id,
            "inbound": {
                "flight": scn.inbound_flight.flight_id,
                "origin": scn.inbound_flight.origin,
                "delay_min": scn.metadata["delay_realized_min"],
            },
            "recovery_flights": [
                {
                    "flight": f.flight_id,
                    "destination": f.destination,
                    "open_F": f.seats_open_F,
                    "open_Yplus": f.seats_open_Yplus,
                    "open_Y": f.seats_open_Y,
                    "open_total": f.open_seats_total,
                    "minutes_after_arrival": f.sched_dep_min - scn.inbound_flight.actual_arr_min,
                }
                for f in scn.recovery_flights
            ],
            "n_passengers": len(scn.passengers),
            "total_open_seats": scn.total_open_seats,
            "supply_demand_ratio": round(scn.supply_demand_ratio, 3),
            "passengers": passengers,
            "results": {
                "manual": {
                    "total_loss": round(manual.expected_loss, 2),
                    "n_misconnects": manual.n_misconnects,
                    "solve_ms": manual.solve_time_ms,
                    "breakdown": manual.objective_breakdown,
                },
                "lp": {
                    "total_loss": round(lp.expected_loss, 2),
                    "n_misconnects": lp.n_misconnects,
                    "solve_ms": lp.solve_time_ms,
                    "breakdown": lp.objective_breakdown,
                },
                "delta_dollars": round(manual.expected_loss - lp.expected_loss, 2),
                "delta_pct": round(
                    100 * (manual.expected_loss - lp.expected_loss) / manual.expected_loss
                    if manual.expected_loss > 0 else 0, 1
                ),
            },
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {len(output)} demo scenarios to {out}")
