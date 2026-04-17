from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from multi_agent_vlm_orchestrator.agents import SubAgent
from multi_agent_vlm_orchestrator.clients import build_client
from multi_agent_vlm_orchestrator.models import (
    AgentResult,
    AgentTask,
    BackendType,
    ExperimentConfig,
    ExperimentSample,
    ImageGenerationResult,
    TwoStageExperimentResult,
)
from multi_agent_vlm_orchestrator.orchestrator import _render_prompt
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry

logger = logging.getLogger(__name__)


def _slug(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-") or "item"


def _image_suffix(backend: BackendType) -> str:
    if backend == BackendType.MOCK:
        return ".ppm"
    return ".png"


class TwoStageExperimentRunner:
    """Runs baseline text answering and the text -> image -> text avatar path."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_registry: ModelRegistry,
        script_registry: ScriptRegistry,
    ) -> None:
        self.experiment_config = experiment_config
        self.model_registry = model_registry
        self.script_registry = script_registry

    def run(self) -> list[TwoStageExperimentResult]:
        experiment = self.experiment_config.experiment
        if experiment.generator_model is None:
            raise ValueError("Two-stage experiments require experiment.generator_model")

        generator_profile = self.model_registry.get(experiment.generator_model)
        if "text_to_image" not in generator_profile.capabilities:
            raise ValueError(
                f"Model '{experiment.generator_model}' does not declare text_to_image support"
            )
        generator_client = build_client(generator_profile)

        scripts = self.script_registry.select(experiment.script_ids)
        samples = self._samples()
        records: list[TwoStageExperimentResult] = []
        for sample in samples:
            for script_id, script in scripts.items():
                start = time.perf_counter()
                prompt = _render_prompt(script.prompt_template, sample.prompt)
                sample_id = sample.sample_id
                persona_id = sample.persona_id
                evaluator_model = experiment.evaluator_model or script.preferred_model
                self._validate_answer_models(script.preferred_model, evaluator_model)
                metadata = {**experiment.metadata, **sample.metadata}

                baseline = self._run_answer(
                    script_id=script_id,
                    model_name=script.preferred_model,
                    prompt=prompt,
                    image_path=None,
                    description=f"{script.description} baseline",
                )
                image_result = self._run_image_generation(
                    generator_model=experiment.generator_model,
                    generator_profile=generator_profile,
                    generator_client=generator_client,
                    prompt=prompt,
                    sample_id=sample_id,
                    persona_id=persona_id,
                )

                avatar_answer: AgentResult | None = None
                if image_result.success and image_result.image_path is not None:
                    avatar_answer = self._run_answer(
                        script_id=script_id,
                        model_name=evaluator_model,
                        prompt=prompt,
                        image_path=Path(image_result.image_path),
                        description=f"{script.description} generated-avatar answer",
                    )

                success = bool(
                    baseline.success
                    and image_result.success
                    and avatar_answer is not None
                    and avatar_answer.success
                )
                error = None
                if not success:
                    error = self._first_error(baseline, image_result, avatar_answer)

                records.append(
                    TwoStageExperimentResult(
                        experiment_name=experiment.name,
                        script_id=script_id,
                        sample_id=sample_id,
                        persona_id=str(persona_id) if persona_id is not None else None,
                        prompt=prompt,
                        baseline=baseline,
                        image_generation=image_result,
                        avatar_answer=avatar_answer,
                        success=success,
                        error=error,
                        elapsed_seconds=time.perf_counter() - start,
                        metadata=metadata,
                    )
                )

        self._write_results(records)
        return records

    def _run_answer(
        self,
        script_id: str,
        model_name: str,
        prompt: str,
        image_path: Path | None,
        description: str,
    ) -> AgentResult:
        profile = self.model_registry.get(model_name)
        task = AgentTask(
            script_id=script_id,
            model_name=model_name,
            prompt=prompt,
            image_path=image_path,
            description=description,
        )
        agent = SubAgent(model_name=model_name, profile=profile, client=build_client(profile))
        return agent.run(task)

    def _validate_answer_models(self, baseline_model: str, evaluator_model: str) -> None:
        baseline_profile = self.model_registry.get(baseline_model)
        if "text_to_text" not in baseline_profile.capabilities:
            raise ValueError(f"Model '{baseline_model}' does not declare text_to_text support")

        evaluator_profile = self.model_registry.get(evaluator_model)
        if "vision_to_text" not in evaluator_profile.capabilities:
            raise ValueError(f"Model '{evaluator_model}' does not declare vision_to_text support")

    def _samples(self) -> list[ExperimentSample]:
        experiment = self.experiment_config.experiment
        if experiment.samples:
            return experiment.samples
        sample_id = str(experiment.metadata.get("sample_id", "sample_001"))
        persona_id = experiment.metadata.get("persona_id")
        return [
            ExperimentSample(
                sample_id=sample_id,
                prompt=experiment.prompt,
                persona_id=str(persona_id) if persona_id is not None else None,
                metadata={},
            )
        ]

    def _run_image_generation(
        self,
        generator_model: str,
        generator_profile,
        generator_client,
        prompt: str,
        sample_id: str,
        persona_id: object | None,
    ) -> ImageGenerationResult:
        experiment = self.experiment_config.experiment
        start = time.perf_counter()
        persona_part = _slug(str(persona_id)) if persona_id is not None else "default"
        image_name = f"avatar{_image_suffix(generator_profile.backend)}"
        output_path = (
            Path(experiment.image_output_dir)
            / _slug(sample_id)
            / _slug(generator_model)
            / persona_part
            / image_name
        )
        try:
            image_path, metadata = generator_client.generate_image(prompt, output_path)
        except Exception as exc:
            return ImageGenerationResult(
                model_name=generator_model,
                model_id=generator_profile.model_id,
                backend=generator_profile.backend,
                image_path=None,
                prompt=prompt,
                success=False,
                error=str(exc),
                elapsed_seconds=time.perf_counter() - start,
                metadata={},
            )
        return ImageGenerationResult(
            model_name=generator_model,
            model_id=generator_profile.model_id,
            backend=generator_profile.backend,
            image_path=str(image_path),
            prompt=prompt,
            success=True,
            error=None,
            elapsed_seconds=time.perf_counter() - start,
            metadata=metadata,
        )

    def _write_results(self, records: list[TwoStageExperimentResult]) -> None:
        output_path = Path(self.experiment_config.experiment.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=True) + "\n")
        logger.info("Wrote %s two-stage records to %s", len(records), output_path)

    @staticmethod
    def _first_error(
        baseline: AgentResult,
        image_result: ImageGenerationResult,
        avatar_answer: AgentResult | None,
    ) -> str | None:
        if not baseline.success:
            return baseline.error
        if not image_result.success:
            return image_result.error
        if avatar_answer is None:
            return "avatar_answer was not run"
        if not avatar_answer.success:
            return avatar_answer.error
        return None
