from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import logging

from multi_agent_vlm_orchestrator.models import AgentTask, BackendType, ModelProfile

logger = logging.getLogger(__name__)


class VLMClient(ABC):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @abstractmethod
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class MockVLMClient(VLMClient):
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        image_note = f" image={task.image_path}" if task.image_path else ""
        text = (
            f"[mock:{self.profile.model_id}] script={task.script_id}{image_note} "
            f"prompt={task.prompt[:120]}"
        )
        return text, {"mode": "mock"}


class TransformersLocalVLMClient(VLMClient):
    def __init__(self, profile: ModelProfile) -> None:
        super().__init__(profile)
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers_local backend requires optional dependencies. "
                "Install with: uv sync --extra local"
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.profile.dtype, torch.float16)
        logger.info("Loading model %s", self.profile.model_id)
        self._processor = AutoProcessor.from_pretrained(
            self.profile.model_id,
            revision=self.profile.revision,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.profile.model_id,
            revision=self.profile.revision,
            torch_dtype=torch_dtype,
        ).to(self.profile.device)

    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        self._lazy_load()
        if task.image_path is None:
            raise ValueError("transformers_local backend requires an image_path per task")

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for image loading. Install with: uv sync --extra local"
            ) from exc

        image = Image.open(task.image_path).convert("RGB")
        prompt_text = task.prompt
        if self.profile.system_prompt:
            prompt_text = f"{self.profile.system_prompt}\n\n{prompt_text}"

        inputs = self._processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.profile.device) for key, value in inputs.items()}
        output = self._model.generate(
            **inputs,
            max_new_tokens=self.profile.max_new_tokens,
        )
        text = self._processor.batch_decode(output, skip_special_tokens=True)[0]
        return text, {"mode": "transformers_local", "image_path": str(task.image_path)}


def build_client(profile: ModelProfile) -> VLMClient:
    if profile.backend == BackendType.MOCK:
        return MockVLMClient(profile)
    if profile.backend == BackendType.TRANSFORMERS_LOCAL:
        return TransformersLocalVLMClient(profile)
    raise ValueError(f"Unsupported backend '{profile.backend}'")
