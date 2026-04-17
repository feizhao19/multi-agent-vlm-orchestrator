from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BackendType(str, Enum):
    MOCK = "mock"
    TRANSFORMERS_LOCAL = "transformers_local"
    DIFFUSERS_LOCAL = "diffusers_local"
    EXTERNAL_COMMAND = "external_command"


class ModelProfile(BaseModel):
    provider: str = "huggingface"
    backend: BackendType
    model_id: str
    capabilities: list[str] = Field(default_factory=lambda: ["vision_to_text", "text_to_text"])
    device: str = "cpu"
    conda_env: str | None = None
    dtype: str = "float16"
    revision: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 256
    extra: dict[str, Any] = Field(default_factory=dict)


class ScriptDefinition(BaseModel):
    description: str
    preferred_model: str
    prompt_template: str
    image_path: str | None = None
    tags: list[str] = Field(default_factory=list)


class ExperimentSample(BaseModel):
    sample_id: str
    prompt: str
    persona_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentInput(BaseModel):
    name: str
    prompt: str
    samples: list[ExperimentSample] = Field(default_factory=list)
    script_ids: list[str] | None = None
    output_path: str = "results/latest_run.jsonl"
    image_output_dir: str = "results/generated_images"
    generator_model: str | None = None
    evaluator_model: str | None = None
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
    response_text: str
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageGenerationResult(BaseModel):
    model_name: str
    model_id: str
    backend: BackendType
    image_path: str | None = None
    prompt: str
    success: bool
    error: str | None = None
    elapsed_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TwoStageExperimentResult(BaseModel):
    experiment_name: str
    script_id: str
    sample_id: str
    persona_id: str | None = None
    prompt: str
    baseline: AgentResult | None = None
    image_generation: ImageGenerationResult
    avatar_answer: AgentResult | None = None
    success: bool
    error: str | None = None
    elapsed_seconds: float
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


class AgentResponse(BaseModel):
    request: str
    intent: str
    planner_summary: str
    tool_outputs: list[ToolOutput] = Field(default_factory=list)
    final_text: str
