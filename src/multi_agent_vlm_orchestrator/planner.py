from __future__ import annotations

import re

from multi_agent_vlm_orchestrator.models import PlannerDecision, ToolCall


SCRIPT_ID_PATTERN = re.compile(r"script[_\s-]?(\d+)", re.IGNORECASE)
MODEL_PATTERN = re.compile(r"(?:with|use|using)\s+([a-zA-Z0-9._/-]+)")


class RuleBasedPlanner:
    def plan(self, request: str) -> PlannerDecision:
        normalized = request.strip()
        lowered = normalized.lower()

        if any(token in lowered for token in ["list model", "show model", "available model"]):
            return PlannerDecision(
                intent="list_models",
                summary="List the configured model profiles.",
                tool_calls=[ToolCall(tool_name="list_models")],
            )

        if any(token in lowered for token in ["list script", "show script", "available script"]):
            return PlannerDecision(
                intent="list_scripts",
                summary="List the configured experiment scripts.",
                tool_calls=[
                    ToolCall(
                        tool_name="list_scripts",
                        arguments={"script_ids": self._extract_script_ids(normalized)},
                    )
                ],
            )

        if any(token in lowered for token in ["summarize", "summary", "report", "result"]):
            return PlannerDecision(
                intent="summarize_results",
                summary="Summarize the most recent experiment results.",
                tool_calls=[ToolCall(tool_name="summarize_results")],
            )

        if any(token in lowered for token in ["run", "test", "execute", "benchmark", "experiment"]):
            return PlannerDecision(
                intent="run_experiment",
                summary="Run the experiment against the selected scripts and preferred models.",
                tool_calls=[
                    ToolCall(
                        tool_name="run_experiment",
                        arguments={
                            "prompt": self._extract_prompt(normalized),
                            "script_ids": self._extract_script_ids(normalized),
                            "model_name": self._extract_model_name(normalized),
                        },
                    )
                ],
            )

        return PlannerDecision(
            intent="help",
            summary="The request did not map to an executable action.",
            tool_calls=[],
        )

    def _extract_script_ids(self, request: str) -> list[str] | None:
        matches = [f"script_{int(match):03d}" for match in SCRIPT_ID_PATTERN.findall(request)]
        return matches or None

    def _extract_model_name(self, request: str) -> str | None:
        match = MODEL_PATTERN.search(request)
        return match.group(1).strip(" .,:;") if match else None

    def _extract_prompt(self, request: str) -> str:
        lowered = request.lower()
        marker = "on this prompt:"
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            return request[start:].strip()
        return request
