from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import typer

from multi_agent_vlm_orchestrator.agent_system import AgentSystem
from multi_agent_vlm_orchestrator.analysis import compare_baseline_and_two_stage
from multi_agent_vlm_orchestrator.config import (
    load_experiment_config,
    load_models_config,
    load_scripts_config,
)
from multi_agent_vlm_orchestrator.orchestrator import ExperimentRunner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry
from multi_agent_vlm_orchestrator.two_stage import TwoStageExperimentRunner

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def validate(
    models: Path = typer.Option(..., exists=True, dir_okay=False),
    scripts: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    models_config = load_models_config(models)
    scripts_config = load_scripts_config(scripts)
    model_registry = ModelRegistry(models_config)
    model_registry.validate_script_preferences(scripts_config)
    logger.info(
        "Validated %s models and %s scripts",
        len(models_config.models),
        len(scripts_config.scripts),
    )


@app.command()
def run(
    experiment: Path = typer.Option(..., exists=True, dir_okay=False),
    models: Path = typer.Option(..., exists=True, dir_okay=False),
    scripts: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    experiment_config = load_experiment_config(experiment)
    models_config = load_models_config(models)
    scripts_config = load_scripts_config(scripts)

    model_registry = ModelRegistry(models_config)
    model_registry.validate_script_preferences(scripts_config)
    script_registry = ScriptRegistry(scripts_config)
    runner = ExperimentRunner(experiment_config, model_registry, script_registry)
    results = runner.run()

    successes = sum(1 for result in results if result.success)
    failures = len(results) - successes
    logger.info(
        "Completed run with %s successes and %s failures",
        successes,
        failures,
    )


@app.command("run-two-stage")
def run_two_stage(
    experiment: Path = typer.Option(..., exists=True, dir_okay=False),
    models: Path = typer.Option(..., exists=True, dir_okay=False),
    scripts: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    experiment_config = load_experiment_config(experiment)
    models_config = load_models_config(models)
    scripts_config = load_scripts_config(scripts)

    model_registry = ModelRegistry(models_config)
    model_registry.validate_script_preferences(scripts_config)
    script_registry = ScriptRegistry(scripts_config)
    runner = TwoStageExperimentRunner(experiment_config, model_registry, script_registry)
    records = runner.run()

    successes = sum(1 for record in records if record.success)
    failures = len(records) - successes
    logger.info(
        "Completed two-stage run with %s successes and %s failures",
        successes,
        failures,
    )


@app.command("compare")
def compare(
    two_stage: Path = typer.Option(..., exists=True, dir_okay=False),
    baseline: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None, dir_okay=False),
) -> None:
    summary = compare_baseline_and_two_stage(baseline, two_stage)
    payload = json.dumps(summary, ensure_ascii=True, indent=2)
    if output is None:
        typer.echo(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    logger.info("Wrote comparison summary to %s", output)


@app.command("agent")
def agent_command(
    request: str = typer.Option(..., help="Natural-language request for the agent system."),
    models: Path = typer.Option(..., exists=True, dir_okay=False),
    scripts: Path = typer.Option(..., exists=True, dir_okay=False),
    output_path: Path = typer.Option(
        Path("results/agent_latest.jsonl"),
        dir_okay=False,
        help="Default results file used by run and summarize tools.",
    ),
) -> None:
    system = AgentSystem.from_paths(models, scripts, output_path)
    response = system.handle(request)
    logger.info("Intent: %s", response.intent)
    logger.info(response.final_text)


if __name__ == "__main__":
    app()
