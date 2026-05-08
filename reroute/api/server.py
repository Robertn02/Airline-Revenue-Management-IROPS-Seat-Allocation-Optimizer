"""FastAPI server exposing live solving to the web UI.

Endpoints:
    GET  /api/health             — Liveness check
    GET  /api/scenarios           — List pre-computed demo scenarios
    GET  /api/scenarios/{id}     — Get one demo scenario
    POST /api/solve               — Solve a custom scenario with custom coefficients
    POST /api/generate            — Generate a fresh random scenario

The server holds a trained RiskModel in memory and reuses it across requests.

Author: Phuc Nguyen
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from reroute import __version__
from reroute.core.config import Config, CostCoefficients, default_config
from reroute.core.logging import configure_logging, get_logger
from reroute.core.types import DisruptionScenario
from reroute.model.risk import RiskModel
from reroute.sim.generator import generate_scenario, make_scarce_dataset
from reroute.sim.harness import SimulationHarness
from reroute.solver.lp import Allocator

logger = get_logger(__name__)


# ============================================================
# Request / response schemas
# ============================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class CostCoefficientsRequest(BaseModel):
    """Optional cost coefficient overrides for a solve."""
    alpha_yield: float = Field(1.0, ge=0, le=10)
    beta_spill: float = Field(0.85, ge=0, le=10)
    delta_harm: float = Field(1.5, ge=0, le=10)
    lambda_miss: float = Field(1.0, ge=0, le=10)
    miss_fixed_cost_usd: float = Field(250.0, ge=0, le=2000)


class SolveRequest(BaseModel):
    """Request body for /api/solve."""
    scenario: dict[str, Any]
    coefficients: Optional[CostCoefficientsRequest] = None


class GenerateRequest(BaseModel):
    """Request body for /api/generate."""
    n_passengers: int = Field(20, ge=5, le=80)
    n_recovery_flights: int = Field(4, ge=2, le=8)
    delay_min: int = Field(120, ge=30, le=300)
    seed: Optional[int] = None


# ============================================================
# App factory
# ============================================================

class AppState:
    """Singleton holding loaded model + cached demo data."""

    def __init__(self):
        self.model: Optional[RiskModel] = None
        self.config: Config = default_config()
        self.demo_scenarios: list[dict] = []


_state = AppState()


def create_app(model_path: Optional[str] = None) -> FastAPI:
    """Create the FastAPI app with the model preloaded.

    When called as a factory by uvicorn (no args), it defaults to looking for
    a pre-built model at `results/model.pkl`. If not found, it will train one
    fresh — but that is slow and memory-heavy and should not happen in
    production. Build the model into your image with `reroute train` instead.
    """
    if model_path is None:
        # Default for factory invocation
        default_path = Path("results/model.pkl")
        if default_path.exists():
            model_path = str(default_path)
    configure_logging()

    # Load model
    if model_path and Path(model_path).exists():
        _state.model = RiskModel.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.info("Training fresh model on startup...")
        from reroute.model.risk import train_from_scenarios
        from reroute.sim.generator import generate_dataset
        scns = generate_dataset(n_scenarios=200, seed=42)
        _state.model, _, _ = train_from_scenarios(scns)
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            _state.model.save(model_path)

    # Load demo scenarios if available
    demo_path = Path("results/scenarios_for_demo.json")
    if demo_path.exists():
        with open(demo_path) as f:
            _state.demo_scenarios = json.load(f)
        logger.info(f"Loaded {len(_state.demo_scenarios)} demo scenarios")

    app = FastAPI(
        title="Reroute API",
        version=__version__,
        description="Live cohort allocation for airline disruption scenarios",
    )

    # CORS — comma-separated origins via REROUTE_CORS_ORIGINS env var,
    # or "*" by default (fine for a public demo API).
    import os
    cors_env = os.environ.get("REROUTE_CORS_ORIGINS", "*")
    if cors_env == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # ============ ROUTES ============

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            version=__version__,
            model_loaded=_state.model is not None,
        )

    @app.get("/api/info")
    def info():
        """Return model + config metadata."""
        if _state.model is None:
            raise HTTPException(503, "Model not loaded")
        return {
            "version": __version__,
            "model": {
                "auc": _state.model.train_results.auc,
                "brier": _state.model.train_results.brier,
                "n_train": _state.model.train_results.n_train,
                "n_test": _state.model.train_results.n_test,
                "feature_importance": _state.model.train_results.feature_importance,
            },
            "config": _state.config.to_dict(),
        }

    @app.get("/api/scenarios")
    def list_scenarios():
        """List all pre-computed demo scenarios with summary info."""
        return [
            {
                "scenario_id": s["scenario_id"],
                "n_passengers": s["n_passengers"],
                "total_open_seats": s["total_open_seats"],
                "delay_min": s["inbound"]["delay_min"],
                "supply_demand_ratio": s["supply_demand_ratio"],
                "delta_dollars": s["results"]["delta_dollars"],
                "delta_pct": s["results"]["delta_pct"],
            }
            for s in _state.demo_scenarios
        ]

    @app.get("/api/scenarios/{scenario_id}")
    def get_scenario(scenario_id: str):
        """Get full detail for one demo scenario."""
        for s in _state.demo_scenarios:
            if s["scenario_id"] == scenario_id:
                return s
        raise HTTPException(404, f"Scenario {scenario_id} not found")

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        """Generate a fresh random scenario AND solve it with both strategies."""
        if _state.model is None:
            raise HTTPException(503, "Model not loaded")

        seed = req.seed if req.seed is not None else int(np.random.SeedSequence().entropy % (2**32))
        rng = np.random.default_rng(seed)
        scn = generate_scenario(
            rng,
            n_passengers=req.n_passengers,
            n_recovery_flights=req.n_recovery_flights,
            force_delay_min=req.delay_min,
        )

        return _solve_and_format(scn, _state.config)

    @app.post("/api/solve")
    def solve(req: SolveRequest):
        """Solve a scenario with custom coefficients.

        Body:
            scenario: full scenario dict (from generate or scenarios endpoints)
            coefficients: optional override for cost weights

        Returns:
            Same shape as /api/generate output, with both strategies' results.
        """
        if _state.model is None:
            raise HTTPException(503, "Model not loaded")

        try:
            scn = DisruptionScenario.from_dict(req.scenario)
        except Exception as e:
            raise HTTPException(400, f"Invalid scenario: {e}")

        cfg = default_config()
        if req.coefficients:
            cfg.cost = CostCoefficients(**req.coefficients.dict())

        return _solve_and_format(scn, cfg)

    return app


def _solve_and_format(scn: DisruptionScenario, cfg: Config) -> dict[str, Any]:
    """Run both strategies, return full demo-shaped JSON."""
    from reroute.solver.baseline import manual_triage

    probs = _state.model.predict(scn)
    manual = manual_triage(scn, probs, cfg)
    lp = Allocator(cfg).solve(scn, probs)

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
            "confidence": _state.model.confidence_class(float(probs[i])),
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

    return {
        "scenario_id": scn.scenario_id,
        "scenario_full": scn.to_dict(),
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
    }


def run_server(host: str = "127.0.0.1", port: int = 8000, model_path: Optional[str] = None) -> None:
    """Entry point for `reroute serve`."""
    import uvicorn
    app = create_app(model_path)
    # Mount static web UI if present
    web_dir = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    uvicorn.run(app, host=host, port=port, log_level="info")
