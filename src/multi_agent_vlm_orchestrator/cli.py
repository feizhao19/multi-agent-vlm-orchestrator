from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from multi_agent_vlm_orchestrator.agent_system import AgentSystem
from multi_agent_vlm_orchestrator.config import (
    load_experiment_config,
    load_models_config,
    load_scripts_config,
)
from multi_agent_vlm_orchestrator.models import RunRequest, SupervisorConfig
from multi_agent_vlm_orchestrator.orchestrator import ExperimentRunner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry

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


@app.command("agent")
def agent_command(
    request: str = typer.Option(..., help="Natural-language request for the agent system."),
    models: Path = typer.Option(..., exists=True, dir_okay=False),
    scripts: Path = typer.Option(..., exists=True, dir_okay=False),
    supervisor_model: Optional[str] = typer.Option(
        None,
        help="Optional text-only supervisor model, for example qwen3-8b.",
    ),
    output_path: Path = typer.Option(
        Path("results/agent_latest.jsonl"),
        dir_okay=False,
        help="Default results file used by run and summarize tools.",
    ),
) -> None:
    supervisor = None
    if supervisor_model is not None:
        supervisor = SupervisorConfig(planner_type="llm", model_name=supervisor_model)
    system = AgentSystem.from_paths(models, scripts, output_path, supervisor)
    response = system.handle(request)
    logger.info("Intent: %s", response.intent)
    logger.info(response.final_text)


@app.command("supervisor-run")
def supervisor_run(
    script_id: str = typer.Option(..., help="Script identifier, for example script_001."),
    prompt: str = typer.Option(..., help="User prompt to send through the script template."),
    models: Path = typer.Option(..., exists=True, dir_okay=False),
    scripts: Path = typer.Option(..., exists=True, dir_okay=False),
    model_name: Optional[str] = typer.Option(
        None,
        help="Requested worker model. If omitted, the script preferred model is used.",
    ),
    output_path: Path = typer.Option(
        Path("results/supervisor_run.jsonl"),
        dir_okay=False,
        help="Results file written by the supervisor dispatch.",
    ),
) -> None:
    system = AgentSystem.from_paths(models, scripts, output_path)
    response = system.handle_run_request(
        RunRequest(
            script_id=script_id,
            prompt=prompt,
            model_name=model_name,
            output_path=str(output_path),
        )
    )
    logger.info("Intent: %s", response.intent)
    logger.info(response.final_text)


if __name__ == "__main__":
    app()
