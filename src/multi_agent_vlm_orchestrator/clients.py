from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import logging

from multi_agent_vlm_orchestrator.models import AgentTask, BackendType, ModelProfile, TaskMode

logger = logging.getLogger(__name__)


class ModelClient(ABC):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @abstractmethod
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class MockModelClient(ModelClient):
    def generate(self, task: AgentTask) -> tuple[str, dict[str, Any]]:
        image_note = f" image={task.image_path}" if task.image_path else ""
        text = (
            f"[mock:{self.profile.model_id}] script={task.script_id} mode={task.task_mode.value}"
            f"{image_note} prompt={task.prompt[:120]}"
        )
        metadata: dict[str, Any] = {
            "mode": "mock",
            "model_kind": self.profile.model_kind.value,
            "task_mode": task.task_mode.value,
        }
        if task.task_mode in {TaskMode.TEXT_TO_IMAGE, TaskMode.IMAGE_TO_IMAGE}:
            metadata["response_image_path"] = f"mock_outputs/{task.script_id}.png"
            text = f"{text} generated_image={metadata['response_image_path']}"
        return text, metadata


class TransformersLocalModelClient(ModelClient):
    def __init__(self, profile: ModelProfile) -> None:
        super().__init__(profile)
        self._processor = None
        self._model = None
        self._tokenizer = None

    def _is_qwen2_vl(self) -> bool:
        return "qwen2-vl" in self.profile.model_id.lower()

    def _lazy_load(self) -> None:
        if self._model is not None and (
            self.profile.model_kind.value == "llm"
            or (self._processor is not None)
        ):
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers_local backend requires optional dependencies. "
                "Install the local model requirements for this project first."
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.profile.dtype, torch.float16)
        logger.info("Loading model %s", self.profile.model_id)
        if self.profile.model_kind.value == "llm":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "LLM supervisor requires transformers text-generation dependencies."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.profile.model_id,
                revision=self.profile.revision,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.profile.model_id,
                revision=self.profile.revision,
                torch_dtype=torch_dtype,
            ).to(self.profile.device)
        else:
            if self._is_qwen2_vl():
                try:
                    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
                except ImportError as exc:
                    raise RuntimeError(
                        "Qwen2-VL workers require transformers with Qwen2VLForConditionalGeneration."
                    ) from exc
                self._processor = AutoProcessor.from_pretrained(
                    self.profile.model_id,
                    revision=self.profile.revision,
                )
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.profile.model_id,
                    revision=self.profile.revision,
                    torch_dtype=torch_dtype,
                ).to(self.profile.device)
                return
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Vision/unified workers require a transformers build with AutoModelForVision2Seq."
                ) from exc
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

        if self.profile.model_kind.value == "llm":
            assert self._tokenizer is not None
            inputs = self._tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(self.profile.device) for key, value in inputs.items()}
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.profile.max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            generated = output[:, prompt_length:]
            text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return text, {
                "mode": "transformers_local",
                "task_mode": task.task_mode.value,
                "model_kind": self.profile.model_kind.value,
            }

        if self._is_qwen2_vl():
            metadata: dict[str, Any] = {
                "mode": "transformers_local",
                "task_mode": task.task_mode.value,
                "model_kind": self.profile.model_kind.value,
            }
            if task.image_path is None:
                raise ValueError(f"{task.task_mode.value} requires an image_path")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": str(task.image_path)},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.profile.device) for key, value in inputs.items()}
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.profile.max_new_tokens,
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], output)
            ]
            text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            metadata["image_path"] = str(task.image_path)
            return text, metadata

        processor_kwargs: dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
        metadata: dict[str, Any] = {
            "mode": "transformers_local",
            "task_mode": task.task_mode.value,
            "model_kind": self.profile.model_kind.value,
        }

        if task.image_path is not None:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "Pillow is required for image loading. Install local model dependencies first."
                ) from exc
            image = Image.open(task.image_path).convert("RGB")
            processor_kwargs["images"] = image
            metadata["image_path"] = str(task.image_path)
        elif task.task_mode in {TaskMode.VISION_TO_TEXT, TaskMode.IMAGE_TO_IMAGE, TaskMode.MULTIMODAL_CHAT}:
            raise ValueError(f"{task.task_mode.value} requires an image_path")

        if task.task_mode in {TaskMode.TEXT_TO_IMAGE, TaskMode.IMAGE_TO_IMAGE}:
            raise RuntimeError(
                "transformers_local text/image generation is not implemented yet for unified models"
            )

        inputs = self._processor(**processor_kwargs)
        inputs = {key: value.to(self.profile.device) for key, value in inputs.items()}
        output = self._model.generate(
            **inputs,
            max_new_tokens=self.profile.max_new_tokens,
        )
        text = self._processor.batch_decode(output, skip_special_tokens=True)[0]
        return text, metadata


def build_client(profile: ModelProfile) -> ModelClient:
    if profile.backend == BackendType.MOCK:
        return MockModelClient(profile)
    if profile.backend == BackendType.TRANSFORMERS_LOCAL:
        return TransformersLocalModelClient(profile)
    raise ValueError(f"Unsupported backend '{profile.backend}'")
