from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BackendType(str, Enum):
    MOCK = "mock"
    TRANSFORMERS_LOCAL = "transformers_local"


class ModelKind(str, Enum):
    LLM = "llm"
    VLM = "vlm"
    UNIFIED = "unified"


class TaskMode(str, Enum):
    TEXT_ONLY = "text_only"
    VISION_TO_TEXT = "vision_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    MULTIMODAL_CHAT = "multimodal_chat"


class ModelCapabilities(BaseModel):
    supports_text_input: bool = True
    supports_image_input: bool = False
    supports_text_output: bool = True
    supports_image_output: bool = False
    supports_tool_calling: bool = False

    def supports_mode(self, mode: "TaskMode") -> bool:
        if mode == TaskMode.TEXT_ONLY:
            return self.supports_text_input and self.supports_text_output
        if mode == TaskMode.VISION_TO_TEXT:
            return self.supports_image_input and self.supports_text_output
        if mode == TaskMode.TEXT_TO_IMAGE:
            return self.supports_text_input and self.supports_image_output
        if mode == TaskMode.IMAGE_TO_IMAGE:
            return self.supports_image_input and self.supports_image_output
        if mode == TaskMode.MULTIMODAL_CHAT:
            return (
                self.supports_text_input
                and self.supports_image_input
                and self.supports_text_output
            )
        return False


class ModelProfile(BaseModel):
    provider: str = "huggingface"
    backend: BackendType
    model_kind: ModelKind = ModelKind.VLM
    model_id: str
    device: str = "cpu"
    conda_env: str | None = None
    dtype: str = "float16"
    revision: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 256
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    extra: dict[str, Any] = Field(default_factory=dict)


class ScriptDefinition(BaseModel):
    description: str
    preferred_model: str
    prompt_template: str
    image_path: str | None = None
    tags: list[str] = Field(default_factory=list)


class ExperimentInput(BaseModel):
    name: str
    prompt: str
    script_ids: list[str] | None = None
    model_name: str | None = None
    task_mode: TaskMode | None = None
    output_path: str = "results/latest_run.jsonl"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    models: dict[str, ModelProfile]


class ScriptsConfig(BaseModel):
    scripts: dict[str, ScriptDefinition]


class ExperimentConfig(BaseModel):
    experiment: ExperimentInput


class AgentTask(BaseModel):
    script_id: str
    model_name: str
    prompt: str
    task_mode: TaskMode = TaskMode.TEXT_ONLY
    image_path: Path | None = None
    description: str

    @model_validator(mode="after")
    def check_image_path(self) -> "AgentTask":
        if self.image_path is not None:
            self.image_path = self.image_path.expanduser()
        return self


class AgentResult(BaseModel):
    script_id: str
    model_name: str
    model_id: str
    backend: BackendType
    task_mode: TaskMode
    response_text: str
    response_image_path: str | None = None
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class ToolOutput(BaseModel):
    tool_name: str
    success: bool
    content: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PlannerDecision(BaseModel):
    intent: Literal["run_experiment", "list_models", "list_scripts", "summarize_results", "help"]
    summary: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class SupervisorConfig(BaseModel):
    planner_type: Literal["rule_based", "llm"] = "rule_based"
    model_name: str | None = None


class AgentResponse(BaseModel):
    request: str
    intent: str
    planner_summary: str
    tool_outputs: list[ToolOutput] = Field(default_factory=list)
    final_text: str


class RunRequest(BaseModel):
    script_id: str
    prompt: str
    model_name: str | None = None
    image_path: str | None = None
    task_mode: TaskMode | None = None
    run_name: str = "supervisor_run"
    output_path: str = "results/supervisor_run.jsonl"
    metadata: dict[str, Any] = Field(default_factory=dict)
