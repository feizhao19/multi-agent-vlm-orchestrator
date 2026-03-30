from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from multi_agent_vlm_orchestrator.models import (
    ExperimentConfig,
    ExperimentInput,
    ToolCall,
    ToolOutput,
)
from multi_agent_vlm_orchestrator.orchestrator import ExperimentRunner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry


@dataclass
class ToolContext:
    model_registry: ModelRegistry
    script_registry: ScriptRegistry
    default_output_path: Path


ToolHandler = Callable[[ToolContext, dict], ToolOutput]


def _tool_list_models(context: ToolContext, arguments: dict) -> ToolOutput:
    models = [
        {
            "name": name,
            "model_id": profile.model_id,
            "backend": profile.backend.value,
            "device": profile.device,
            "conda_env": profile.conda_env,
        }
        for name, profile in sorted(context.model_registry.items())
    ]
    return ToolOutput(tool_name="list_models", success=True, content={"models": models})


def _tool_list_scripts(context: ToolContext, arguments: dict) -> ToolOutput:
    selected = arguments.get("script_ids")
    scripts = context.script_registry.select(selected)
    payload = [
        {
            "script_id": script_id,
            "description": script.description,
            "preferred_model": script.preferred_model,
            "tags": script.tags,
        }
        for script_id, script in scripts.items()
    ]
    return ToolOutput(tool_name="list_scripts", success=True, content={"scripts": payload})


def _tool_run_experiment(context: ToolContext, arguments: dict) -> ToolOutput:
    prompt = arguments["prompt"]
    script_ids = arguments.get("script_ids")
    model_name = arguments.get("model_name")
    output_path = Path(arguments.get("output_path", context.default_output_path))
    if model_name is not None:
        try:
            context.model_registry.get(model_name)
        except KeyError as exc:
            return ToolOutput(tool_name="run_experiment", success=False, error=str(exc))
    experiment = ExperimentConfig(
        experiment=ExperimentInput(
            name=arguments.get("name", "agent_run"),
            prompt=prompt,
            script_ids=script_ids,
            model_name=model_name,
            output_path=str(output_path),
            metadata={"request_source": "agent_system"},
        )
    )
    runner = ExperimentRunner(
        experiment_config=experiment,
        model_registry=context.model_registry,
        script_registry=context.script_registry,
    )
    results = runner.run()
    successes = sum(1 for result in results if result.success)
    failures = len(results) - successes
    return ToolOutput(
        tool_name="run_experiment",
        success=True,
        content={
            "output_path": str(output_path),
            "requested_script_ids": script_ids,
            "requested_model_name": model_name,
            "total_results": len(results),
            "successes": successes,
            "failures": failures,
        },
    )


def _tool_summarize_results(context: ToolContext, arguments: dict) -> ToolOutput:
    output_path = Path(arguments.get("output_path", context.default_output_path))
    if not output_path.exists():
        return ToolOutput(
            tool_name="summarize_results",
            success=False,
            error=f"Results file not found: {output_path}",
        )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successes = sum(1 for record in records if record.get("success"))
    failures = len(records) - successes
    by_model: dict[str, int] = {}
    for record in records:
        model_name = record.get("model_name", "unknown")
        by_model[model_name] = by_model.get(model_name, 0) + 1

    return ToolOutput(
        tool_name="summarize_results",
        success=True,
        content={
            "output_path": str(output_path),
            "total_results": len(records),
            "successes": successes,
            "failures": failures,
            "results_per_model": by_model,
        },
    )


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {
            "list_models": _tool_list_models,
            "list_scripts": _tool_list_scripts,
            "run_experiment": _tool_run_experiment,
            "summarize_results": _tool_summarize_results,
        }

    def execute(self, context: ToolContext, call: ToolCall) -> ToolOutput:
        try:
            handler = self._handlers[call.tool_name]
        except KeyError as exc:
            return ToolOutput(
                tool_name=call.tool_name,
                success=False,
                error=f"Unknown tool '{call.tool_name}'",
            )
        return handler(context, call.arguments)

    def names(self) -> list[str]:
        return sorted(self._handlers)
