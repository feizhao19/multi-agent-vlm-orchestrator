from __future__ import annotations

import json
import logging
from pathlib import Path

from multi_agent_vlm_orchestrator.agents import SubAgent
from multi_agent_vlm_orchestrator.clients import build_client
from multi_agent_vlm_orchestrator.models import AgentResult, AgentTask, ExperimentConfig
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry

logger = logging.getLogger(__name__)


def _render_prompt(template: str, user_prompt: str) -> str:
    return template.format(user_prompt=user_prompt)


class ExperimentRunner:
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_registry: ModelRegistry,
        script_registry: ScriptRegistry,
    ) -> None:
        self.experiment_config = experiment_config
        self.model_registry = model_registry
        self.script_registry = script_registry

    def build_tasks(self) -> list[AgentTask]:
        experiment = self.experiment_config.experiment
        scripts = self.script_registry.select(experiment.script_ids)
        tasks: list[AgentTask] = []
        for script_id, script in scripts.items():
            task = AgentTask(
                script_id=script_id,
                model_name=script.preferred_model,
                prompt=_render_prompt(script.prompt_template, experiment.prompt),
                image_path=Path(script.image_path) if script.image_path else None,
                description=script.description,
            )
            tasks.append(task)
        return tasks

    def run(self) -> list[AgentResult]:
        results: list[AgentResult] = []
        for task in self.build_tasks():
            profile = self.model_registry.get(task.model_name)
            agent = SubAgent(
                model_name=task.model_name,
                profile=profile,
                client=build_client(profile),
            )
            logger.info("Running script %s with model %s", task.script_id, task.model_name)
            result = agent.run(task)
            results.append(result)
        self._write_results(results)
        return results

    def _write_results(self, results: list[AgentResult]) -> None:
        output_path = Path(self.experiment_config.experiment.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.model_dump(), ensure_ascii=True) + "\n")
        logger.info("Wrote %s results to %s", len(results), output_path)
