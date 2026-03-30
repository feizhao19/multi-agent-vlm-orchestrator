from __future__ import annotations

import json
from pathlib import Path

from multi_agent_vlm_orchestrator.config import load_models_config, load_scripts_config
from multi_agent_vlm_orchestrator.models import AgentResponse, PlannerDecision, ToolOutput
from multi_agent_vlm_orchestrator.planner import RuleBasedPlanner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry
from multi_agent_vlm_orchestrator.tools import ToolContext, ToolRegistry


class IntentRouterAgent:
    def __init__(self, planner: RuleBasedPlanner) -> None:
        self.planner = planner

    def route(self, request: str) -> PlannerDecision:
        return self.planner.plan(request)


class ExecutionAgent:
    def __init__(self, tools: ToolRegistry, context: ToolContext) -> None:
        self.tools = tools
        self.context = context

    def run(self, decision: PlannerDecision) -> list[ToolOutput]:
        return [self.tools.execute(self.context, call) for call in decision.tool_calls]


class ResponseAgent:
    def render(self, request: str, decision: PlannerDecision, outputs: list[ToolOutput]) -> AgentResponse:
        if decision.intent == "help":
            text = (
                "I can list models, list scripts, run an experiment, or summarize results. "
                "Example: 'run script 1 and 3 on this prompt: describe the image'."
            )
            return AgentResponse(
                request=request,
                intent=decision.intent,
                planner_summary=decision.summary,
                tool_outputs=outputs,
                final_text=text,
            )

        success_outputs = [output for output in outputs if output.success]
        failed_outputs = [output for output in outputs if not output.success]
        fragments = [decision.summary]
        if success_outputs:
            fragments.extend(self._render_output(output) for output in success_outputs)
        if failed_outputs:
            fragments.extend(
                f"Tool {output.tool_name} failed: {output.error}" for output in failed_outputs
            )
        return AgentResponse(
            request=request,
            intent=decision.intent,
            planner_summary=decision.summary,
            tool_outputs=outputs,
            final_text=" ".join(fragments),
        )

    def _render_output(self, output: ToolOutput) -> str:
        if output.tool_name == "list_models":
            names = [item["name"] for item in output.content["models"]]
            return f"Configured models: {', '.join(names)}."
        if output.tool_name == "list_scripts":
            scripts = [item["script_id"] for item in output.content["scripts"]]
            return f"Available scripts: {', '.join(scripts)}."
        if output.tool_name == "run_experiment":
            return (
                "Run complete with "
                f"{output.content['successes']} successes, {output.content['failures']} failures, "
                f"results saved to {output.content['output_path']}."
            )
        if output.tool_name == "summarize_results":
            return (
                "Result summary: "
                f"{output.content['total_results']} results, "
                f"{output.content['successes']} successes, "
                f"{output.content['failures']} failures."
            )
        return json.dumps(output.content, ensure_ascii=True)


class AgentSystem:
    def __init__(
        self,
        router: IntentRouterAgent,
        execution_agent: ExecutionAgent,
        response_agent: ResponseAgent,
    ) -> None:
        self.router = router
        self.execution_agent = execution_agent
        self.response_agent = response_agent

    @classmethod
    def from_paths(
        cls,
        models_path: Path,
        scripts_path: Path,
        default_output_path: Path = Path("results/agent_latest.jsonl"),
    ) -> "AgentSystem":
        models_config = load_models_config(models_path)
        scripts_config = load_scripts_config(scripts_path)
        model_registry = ModelRegistry(models_config)
        script_registry = ScriptRegistry(scripts_config)
        model_registry.validate_script_preferences(scripts_config)
        context = ToolContext(
            model_registry=model_registry,
            script_registry=script_registry,
            default_output_path=default_output_path,
        )
        return cls(
            router=IntentRouterAgent(RuleBasedPlanner()),
            execution_agent=ExecutionAgent(ToolRegistry(), context),
            response_agent=ResponseAgent(),
        )

    def handle(self, request: str) -> AgentResponse:
        decision = self.router.route(request)
        outputs = self.execution_agent.run(decision)
        return self.response_agent.render(request, decision, outputs)
