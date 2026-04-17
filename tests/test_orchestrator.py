import json
import unittest
from pathlib import Path

from multi_agent_vlm_orchestrator.models import (
    ExperimentConfig,
    ExperimentInput,
    ExperimentSample,
    ModelProfile,
    ModelsConfig,
    ScriptDefinition,
    ScriptsConfig,
)
from multi_agent_vlm_orchestrator.orchestrator import ExperimentRunner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry
from multi_agent_vlm_orchestrator.two_stage import TwoStageExperimentRunner


class OrchestratorTests(unittest.TestCase):
    def test_runner_writes_results(self) -> None:
        tmp_path = Path("results/test_tmp")
        tmp_path.mkdir(parents=True, exist_ok=True)
        result_path = tmp_path / "results.jsonl"

        experiment = ExperimentConfig(
            experiment=ExperimentInput(
                name="test",
                prompt="what is in the image?",
                output_path=str(result_path),
            )
        )
        models = ModelsConfig(
            models={"mock": ModelProfile(backend="mock", model_id="mock/model")}
        )
        scripts = ScriptsConfig(
            scripts={
                "script_001": ScriptDefinition(
                    description="demo",
                    preferred_model="mock",
                    prompt_template="Answer this: {user_prompt}",
                )
            }
        )
        runner = ExperimentRunner(
            experiment_config=experiment,
            model_registry=ModelRegistry(models),
            script_registry=ScriptRegistry(scripts),
        )

        results = runner.run()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        payload = json.loads(result_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["script_id"], "script_001")

    def test_two_stage_runner_writes_image_and_results(self) -> None:
        tmp_path = Path("results/test_tmp/two_stage")
        tmp_path.mkdir(parents=True, exist_ok=True)
        result_path = tmp_path / "results.jsonl"
        image_dir = tmp_path / "images"

        experiment = ExperimentConfig(
            experiment=ExperimentInput(
                name="two-stage-test",
                prompt="fallback prompt",
                samples=[
                    ExperimentSample(
                        sample_id="sample-a",
                        persona_id="persona-a",
                        prompt="face description plus questions",
                    )
                ],
                output_path=str(result_path),
                image_output_dir=str(image_dir),
                generator_model="mock-image",
                evaluator_model="mock",
            )
        )
        models = ModelsConfig(
            models={
                "mock": ModelProfile(backend="mock", model_id="mock/model"),
                "mock-image": ModelProfile(
                    backend="mock",
                    model_id="mock/image",
                    capabilities=["text_to_image"],
                ),
            }
        )
        scripts = ScriptsConfig(
            scripts={
                "script_001": ScriptDefinition(
                    description="demo",
                    preferred_model="mock",
                    prompt_template="Answer this as JSON: {user_prompt}",
                )
            }
        )
        runner = TwoStageExperimentRunner(
            experiment_config=experiment,
            model_registry=ModelRegistry(models),
            script_registry=ScriptRegistry(scripts),
        )

        results = runner.run()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertTrue(result_path.exists())
        self.assertIsNotNone(results[0].image_generation.image_path)
        self.assertTrue(Path(results[0].image_generation.image_path or "").exists())
        self.assertIn(
            "sample-a/mock-image/persona-a/avatar.ppm",
            results[0].image_generation.image_path,
        )
        payload = json.loads(result_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["script_id"], "script_001")
        self.assertEqual(payload["sample_id"], "sample-a")


if __name__ == "__main__":
    unittest.main()
