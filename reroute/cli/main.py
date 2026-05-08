"""Reroute command-line interface.

Provides commands for the common workflow: generate data, train model,
run simulation, generate plots, export demo data, serve the API.

Usage:
    reroute --help
    reroute train
    reroute simulate --scenarios 100
    reroute analyze
    reroute serve

Author: Phuc Nguyen
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from reroute import __version__
from reroute.core.config import Config, default_config
from reroute.core.logging import configure_logging, get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="reroute")
@click.option("--quiet", is_flag=True, help="Suppress informational output.")
@click.option("--config", type=click.Path(exists=True), help="Path to YAML config.")
@click.pass_context
def cli(ctx: click.Context, quiet: bool, config: str | None) -> None:
    """Reroute — cohort-level seat allocation for airline disruption recovery."""
    configure_logging(quiet=quiet)
    cfg = Config.from_yaml(config) if config else default_config()
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@cli.command()
@click.option("-n", "--scenarios", default=200, help="Number of training scenarios.")
@click.option("-s", "--seed", default=42, help="RNG seed.")
@click.option("-o", "--output", default="results/model.pkl", help="Output model path.")
@click.pass_context
def train(ctx: click.Context, scenarios: int, seed: int, output: str) -> None:
    """Train and save the calibrated risk model."""
    from reroute.model.risk import train_from_scenarios
    from reroute.sim.generator import generate_dataset

    cfg = ctx.obj["config"]
    console.print(f"[bold cyan]Training risk model[/bold cyan] on {scenarios} scenarios...")
    scns = generate_dataset(n_scenarios=scenarios, seed=seed, config=cfg)
    n_pax = sum(len(s.passengers) for s in scns)
    console.print(f"  Generated [bold]{n_pax}[/bold] training samples")

    model, _, _ = train_from_scenarios(scns, seed=seed, config=cfg)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    table = Table(show_header=False, box=None)
    table.add_column("metric", style="dim")
    table.add_column("value")
    table.add_row("AUC", f"{model.train_results.auc:.4f}")
    table.add_row("Brier score", f"{model.train_results.brier:.4f}")
    table.add_row("Log loss", f"{model.train_results.log_loss:.4f}")
    table.add_row("Train samples", str(model.train_results.n_train))
    table.add_row("Test samples", str(model.train_results.n_test))
    console.print(table)
    console.print(f"[green]✓[/green] Model saved to {out}")


@cli.command()
@click.option("-n", "--scenarios", default=100, help="Number of scenarios to simulate.")
@click.option("-s", "--seed", default=99, help="RNG seed.")
@click.option("--model", default="results/model.pkl", help="Path to trained model.")
@click.option("--output-dir", default="results", help="Where to save results.")
@click.pass_context
def simulate(ctx: click.Context, scenarios: int, seed: int, model: str, output_dir: str) -> None:
    """Run both strategies on N scenarios and save comparison."""
    from reroute.model.risk import RiskModel
    from reroute.sim.generator import make_scarce_dataset
    from reroute.sim.harness import SimulationHarness

    cfg = ctx.obj["config"]

    if not Path(model).exists():
        console.print(f"[yellow]No model at {model} — training a fresh one[/yellow]")
        ctx.invoke(train, scenarios=200, seed=42, output=model)

    risk_model = RiskModel.load(model)
    console.print(f"[bold cyan]Generating[/bold cyan] {scenarios} scarcity-filtered scenarios...")
    scns = make_scarce_dataset(n_scenarios=scenarios, seed=seed, config=cfg)
    console.print(f"  Got {len(scns)} valid scenarios")

    harness = SimulationHarness(risk_model, config=cfg)
    console.print("[bold cyan]Running both strategies...[/bold cyan]")
    results = harness.run_batch(scns)
    summary = harness.summarize(results)

    # Display summary
    table = Table(title="Simulation Summary", show_header=False)
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right", style="bold")
    for k, v in summary.items():
        if isinstance(v, float):
            v = f"{v:,.2f}"
        elif isinstance(v, int):
            v = f"{v:,}"
        table.add_row(k, str(v))
    console.print(table)

    harness.save_results(results, summary, out_dir=output_dir)
    console.print(f"[green]✓[/green] Results in {output_dir}/")


@cli.command()
@click.option("--input-dir", default="results", help="Where to find simulation results.")
def analyze(input_dir: str) -> None:
    """Generate analysis figures from simulation results."""
    from reroute.cli.analyze import run_analysis
    run_analysis(Path(input_dir))


@cli.command()
@click.option("-n", "--scenarios", default=12, help="Number of scenarios to export.")
@click.option("--model", default="results/model.pkl", help="Path to trained model.")
@click.option("--output", default="results/scenarios_for_demo.json", help="Output path.")
@click.pass_context
def export_demo(ctx: click.Context, scenarios: int, model: str, output: str) -> None:
    """Export per-scenario detail to JSON for the web demo."""
    from reroute.cli.export_demo import run_export
    cfg = ctx.obj["config"]
    run_export(n_scenarios=scenarios, model_path=model, output_path=output, config=cfg)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host.")
@click.option("--port", default=8000, help="Bind port.")
@click.option("--model", default="results/model.pkl", help="Path to trained model.")
def serve(host: str, port: int, model: str) -> None:
    """Run the FastAPI backend serving live solves to the web UI."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[red]FastAPI dependencies not installed.[/red]")
        console.print("Install with: [cyan]pip install reroute[api][/cyan]")
        sys.exit(1)

    from reroute.api.server import run_server
    run_server(host=host, port=port, model_path=model)


@cli.command()
def info() -> None:
    """Show package info and configuration summary."""
    from reroute.core.config import default_config
    cfg = default_config()
    console.print(f"[bold cyan]Reroute[/bold cyan] v{__version__}")
    console.print()
    console.print("[bold]Default cost coefficients:[/bold]")
    for k, v in cfg.cost.__dict__.items():
        console.print(f"  {k}: {v}")
    console.print()
    console.print("[bold]Operational constants:[/bold]")
    console.print(f"  MCT (domestic): {cfg.operational.mct_domestic_min} min")
    console.print(f"  Hub: {cfg.operational.hub_airport}")
    console.print(f"  Destinations: {', '.join(cfg.operational.destinations)}")


if __name__ == "__main__":
    cli()
