from __future__ import annotations

import time

from multi_agent_vlm_orchestrator.clients import VLMClient
from multi_agent_vlm_orchestrator.models import AgentResult, AgentTask, ModelProfile


class SubAgent:
    def __init__(self, model_name: str, profile: ModelProfile, client: VLMClient) -> None:
        self.model_name = model_name
        self.profile = profile
        self.client = client

    def run(self, task: AgentTask) -> AgentResult:
        start = time.perf_counter()
        try:
            response_text, metadata = self.client.generate(task)
        except Exception as exc:
            return AgentResult(
                script_id=task.script_id,
                model_name=self.model_name,
                model_id=self.profile.model_id,
                backend=self.profile.backend,
                response_text="",
                success=False,
                error=str(exc),
                metadata={"elapsed_seconds": time.perf_counter() - start},
            )

        metadata = dict(metadata)
        metadata["elapsed_seconds"] = time.perf_counter() - start
        return AgentResult(
            script_id=task.script_id,
            model_name=self.model_name,
            model_id=self.profile.model_id,
            backend=self.profile.backend,
            response_text=response_text,
            success=True,
            error=None,
            metadata=metadata,
        )
