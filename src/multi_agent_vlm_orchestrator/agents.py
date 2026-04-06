from __future__ import annotations

from multi_agent_vlm_orchestrator.clients import ModelClient
from multi_agent_vlm_orchestrator.models import AgentResult, AgentTask, ModelProfile


class SubAgent:
    def __init__(self, model_name: str, profile: ModelProfile, client: ModelClient) -> None:
        self.model_name = model_name
        self.profile = profile
        self.client = client

    def run(self, task: AgentTask) -> AgentResult:
        try:
            response_text, metadata = self.client.generate(task)
        except Exception as exc:
            return AgentResult(
                script_id=task.script_id,
                model_name=self.model_name,
                model_id=self.profile.model_id,
                backend=self.profile.backend,
                task_mode=task.task_mode,
                response_text="",
                response_image_path=None,
                success=False,
                error=str(exc),
                metadata={},
            )

        return AgentResult(
            script_id=task.script_id,
            model_name=self.model_name,
            model_id=self.profile.model_id,
            backend=self.profile.backend,
            response_text=response_text,
            response_image_path=metadata.get("response_image_path"),
            task_mode=task.task_mode,
            success=True,
            error=None,
            metadata=metadata,
        )
