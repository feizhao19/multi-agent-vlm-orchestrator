import json
import unittest
from pathlib import Path

from multi_agent_vlm_orchestrator.models import (
    ExperimentConfig,
    ExperimentInput,
    ModelProfile,
    ModelsConfig,
    ScriptDefinition,
    ScriptsConfig,
)
from multi_agent_vlm_orchestrator.orchestrator import ExperimentRunner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry


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


if __name__ == "__main__":
    unittest.main()
