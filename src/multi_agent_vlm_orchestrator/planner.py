from __future__ import annotations

import json
import re
from textwrap import dedent

from multi_agent_vlm_orchestrator.clients import ModelClient
from multi_agent_vlm_orchestrator.models import AgentTask, PlannerDecision, TaskMode, ToolCall


SCRIPT_ID_PATTERN = re.compile(r"script[_\s-]?(\d+)", re.IGNORECASE)
MODEL_PATTERN = re.compile(r"(?:with|use|using)\s+([a-zA-Z0-9._/-]+)")


class BasePlanner:
    def plan(self, request: str) -> PlannerDecision:
        raise NotImplementedError


class RuleBasedPlanner(BasePlanner):
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
                            "task_mode": self._extract_task_mode(normalized).value,
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

    def _extract_task_mode(self, request: str) -> TaskMode:
        lowered = request.lower()
        if any(token in lowered for token in ["generate image", "create image", "draw image"]):
            return TaskMode.TEXT_TO_IMAGE
        if "image" in lowered or "picture" in lowered or "photo" in lowered:
            return TaskMode.VISION_TO_TEXT
        return TaskMode.TEXT_ONLY


class LLMSupervisorPlanner(BasePlanner):
    def __init__(self, client: ModelClient, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def plan(self, request: str) -> PlannerDecision:
        supervisor_prompt = dedent(
            f"""
            You are a routing supervisor.
            Reply with exactly one JSON object.
            Do not explain. Do not think aloud. Do not use markdown or code fences.

            Schema:
            {{
              "intent": "run_experiment" | "list_models" | "list_scripts" | "summarize_results" | "help",
              "summary": "short sentence",
              "tool_calls": [
                {{
                  "tool_name": "list_models" | "list_scripts" | "run_experiment" | "summarize_results",
                  "arguments": {{
                    "prompt": "string or omitted",
                    "script_ids": ["script_001"] or null,
                    "model_name": "string or null",
                    "task_mode": "text_only" | "vision_to_text" | "text_to_image" | "image_to_image" | "multimodal_chat"
                  }},
                  "rationale": "short sentence"
                }}
              ]
            }}

            If a script number appears like 1, convert it to script_001.
            If the task is image description, use task_mode "vision_to_text".
            Keep the JSON short.

            Request: {request}
            """
        ).strip()
        task = AgentTask(
            script_id="supervisor_request",
            model_name=self.model_name,
            prompt=supervisor_prompt,
            task_mode=TaskMode.TEXT_ONLY,
            description="Supervisor planning request",
        )
        raw_text, _ = self.client.generate(task)
        return self._parse_decision(raw_text)

    def _parse_decision(self, raw_text: str) -> PlannerDecision:
        payload_text = self._extract_json_object(raw_text)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Supervisor model returned invalid JSON: {raw_text}") from exc
        payload = self._normalize_payload(payload)
        return PlannerDecision.model_validate(payload)

    def _extract_json_object(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                _, end = decoder.raw_decode(text[index:])
                return text[index:index + end]
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Supervisor model returned invalid JSON: {raw_text}")

    def _normalize_payload(self, payload: dict) -> dict:
        for tool_call in payload.get("tool_calls", []):
            arguments = tool_call.get("arguments", {})
            script_ids = arguments.get("script_ids")
            if isinstance(script_ids, list):
                arguments["script_ids"] = [self._normalize_script_id(script_id) for script_id in script_ids]
        return payload

    def _normalize_script_id(self, script_id: str) -> str:
        if re.fullmatch(r"\d+", str(script_id)):
            return f"script_{int(script_id):03d}"
        return str(script_id)
