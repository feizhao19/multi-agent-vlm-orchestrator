from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import subprocess
from pathlib import Path
from typing import Any

from multi_agent_vlm_orchestrator.models import AgentTask, BackendType, ModelProfile

logger = logging.getLogger(__name__)


class VLMClient(ABC):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @abstractmethod
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def generate_image(self, prompt: str, output_path: Path) -> tuple[Path, dict[str, Any]]:
        raise NotImplementedError(
            f"Backend '{self.profile.backend}' does not support text_to_image"
        )


class MockVLMClient(VLMClient):
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        image_note = f" image={task.image_path}" if task.image_path else ""
        text = (
            f"[mock:{self.profile.model_id}] script={task.script_id}{image_note} "
            f"prompt={task.prompt[:120]}"
        )
        return text, {"mode": "mock"}

    def generate_image(self, prompt: str, output_path: Path) -> tuple[Path, dict[str, Any]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Portable pixmap keeps the mock path dependency-free while still producing an image.
        width = 64
        height = 64
        seed = sum(prompt.encode("utf-8")) % 255
        with output_path.open("w", encoding="ascii") as handle:
            handle.write(f"P3\n{width} {height}\n255\n")
            for y in range(height):
                row: list[str] = []
                for x in range(width):
                    row.append(str((x * 4 + seed) % 255))
                    row.append(str((y * 4 + seed // 2) % 255))
                    row.append(str((x + y + seed // 3) % 255))
                handle.write(" ".join(row) + "\n")
        return output_path, {"mode": "mock", "format": "ppm"}


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
        prompt_text = task.prompt
        if self.profile.system_prompt:
            prompt_text = f"{self.profile.system_prompt}\n\n{prompt_text}"

        if task.image_path is None:
            inputs = self._processor(text=prompt_text, return_tensors="pt")
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "Pillow is required for image loading. Install with: uv sync --extra local"
                ) from exc
            image = Image.open(task.image_path).convert("RGB")
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


class DiffusersLocalImageClient(VLMClient):
    def __init__(self, profile: ModelProfile) -> None:
        super().__init__(profile)
        self._pipeline = None

    def _lazy_load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers_local backend requires optional dependencies. "
                "Install diffusers and a compatible torch build in this environment."
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.profile.dtype, torch.float16)
        logger.info("Loading image generation model %s", self.profile.model_id)
        self._pipeline = DiffusionPipeline.from_pretrained(
            self.profile.model_id,
            revision=self.profile.revision,
            torch_dtype=torch_dtype,
            **self.profile.extra.get("from_pretrained_kwargs", {}),
        ).to(self.profile.device)

    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError("diffusers_local backend only supports text_to_image")

    def generate_image(self, prompt: str, output_path: Path) -> tuple[Path, dict[str, Any]]:
        self._lazy_load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        call_kwargs = dict(self.profile.extra.get("call_kwargs", {}))
        image = self._pipeline(prompt, **call_kwargs).images[0]
        image.save(output_path)
        return output_path, {"mode": "diffusers_local", "call_kwargs": call_kwargs}


class ExternalCommandClient(VLMClient):
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        command = self._format_command(
            "generate_command",
            prompt=task.prompt,
            image_path=str(task.image_path or ""),
            output_path="",
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds(),
        )
        return completed.stdout.strip(), {
            "mode": "external_command",
            "stderr": completed.stderr.strip(),
        }

    def generate_image(self, prompt: str, output_path: Path) -> tuple[Path, dict[str, Any]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._format_command(
            "generate_image_command",
            prompt=prompt,
            image_path="",
            output_path=str(output_path),
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds(),
        )
        if not output_path.exists():
            raise RuntimeError(
                f"External image command completed but did not create {output_path}"
            )
        return output_path, {
            "mode": "external_command",
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    def _format_command(
        self,
        key: str,
        prompt: str,
        image_path: str,
        output_path: str,
    ) -> list[str]:
        template = self.profile.extra.get(key)
        if not isinstance(template, list) or not all(isinstance(item, str) for item in template):
            raise ValueError(f"Model '{self.profile.model_id}' requires extra.{key} as a list")
        values = {
            "model_id": self.profile.model_id,
            "prompt": prompt,
            "image_path": image_path,
            "output_path": output_path,
        }
        return [item.format(**values) for item in template]

    def _timeout_seconds(self) -> float | None:
        timeout = self.profile.extra.get("timeout_seconds")
        if timeout is None:
            return None
        return float(timeout)


def build_client(profile: ModelProfile) -> VLMClient:
    if profile.backend == BackendType.MOCK:
        return MockVLMClient(profile)
    if profile.backend == BackendType.TRANSFORMERS_LOCAL:
        return TransformersLocalVLMClient(profile)
    if profile.backend == BackendType.DIFFUSERS_LOCAL:
        return DiffusersLocalImageClient(profile)
    if profile.backend == BackendType.EXTERNAL_COMMAND:
        return ExternalCommandClient(profile)
    raise ValueError(f"Unsupported backend '{profile.backend}'")
