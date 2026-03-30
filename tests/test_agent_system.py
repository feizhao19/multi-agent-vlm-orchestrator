import json
import unittest
from pathlib import Path

from multi_agent_vlm_orchestrator.agent_system import AgentSystem
from multi_agent_vlm_orchestrator.models import (
    ModelProfile,
    ModelsConfig,
    RunRequest,
    ScriptDefinition,
    ScriptsConfig,
)
from multi_agent_vlm_orchestrator.planner import RuleBasedPlanner
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry
from multi_agent_vlm_orchestrator.tools import ToolContext, ToolRegistry


class PlannerTests(unittest.TestCase):
    def test_extract_script_ids_from_run_request(self) -> None:
        decision = RuleBasedPlanner().plan("run script 1 and script-3 on this prompt")
        self.assertEqual(decision.intent, "run_experiment")
        self.assertEqual(decision.tool_calls[0].arguments["script_ids"], ["script_001", "script_003"])

    def test_extract_model_name_and_prompt(self) -> None:
        decision = RuleBasedPlanner().plan(
            "run script 1 with qwen2-vl-2b on this prompt: describe the image"
        )
        self.assertEqual(decision.tool_calls[0].arguments["model_name"], "qwen2-vl-2b")
        self.assertEqual(decision.tool_calls[0].arguments["prompt"], "describe the image")


class AgentSystemTests(unittest.TestCase):
    def test_tool_registry_runs_experiment(self) -> None:
        context = ToolContext(
            model_registry=ModelRegistry(
                ModelsConfig(
                    models={"mock": ModelProfile(backend="mock", model_id="mock/model", conda_env="vlm")}
                )
            ),
            script_registry=ScriptRegistry(
                ScriptsConfig(
                    scripts={
                        "script_001": ScriptDefinition(
                            description="demo",
                            preferred_model="mock",
                            prompt_template="Answer this: {user_prompt}",
                        )
                    }
                )
            ),
            default_output_path=Path("results/test_agent/results.jsonl"),
        )
        output = ToolRegistry().execute(
            context,
            RuleBasedPlanner().plan("run script 1 on this prompt").tool_calls[0],
        )

        self.assertTrue(output.success)
        self.assertEqual(output.content["successes"], 1)
        self.assertTrue(Path(output.content["output_path"]).exists())

    def test_agent_system_lists_models(self) -> None:
        base = Path("results/test_agent_system")
        base.mkdir(parents=True, exist_ok=True)
        models_path = base / "models.json"
        scripts_path = base / "scripts.json"
        models_path.write_text(
            json.dumps(
                {
                    "models": {
                        "mock": {
                            "backend": "mock",
                            "model_id": "mock/model",
                            "conda_env": "vlm-mock",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scripts_path.write_text(
            json.dumps(
                {
                    "scripts": {
                        "script_001": {
                            "description": "demo",
                            "preferred_model": "mock",
                            "prompt_template": "{user_prompt}",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        response = AgentSystem.from_paths(models_path, scripts_path).handle("list models")

        self.assertEqual(response.intent, "list_models")
        self.assertIn("Configured models: mock.", response.final_text)

    def test_structured_run_request_prefers_user_model(self) -> None:
        base = Path("results/test_structured_run")
        base.mkdir(parents=True, exist_ok=True)
        models_path = base / "models.json"
        scripts_path = base / "scripts.json"
        output_path = base / "results.jsonl"
        models_path.write_text(
            json.dumps(
                {
                    "models": {
                        "mock-a": {"backend": "mock", "model_id": "mock/a"},
                        "mock-b": {"backend": "mock", "model_id": "mock/b"},
                    }
                }
            ),
            encoding="utf-8",
        )
        scripts_path.write_text(
            json.dumps(
                {
                    "scripts": {
                        "script_001": {
                            "description": "demo",
                            "preferred_model": "mock-a",
                            "prompt_template": "{user_prompt}",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        system = AgentSystem.from_paths(models_path, scripts_path, output_path)

        response = system.handle_run_request(
            RunRequest(
                script_id="script_001",
                prompt="describe the image",
                model_name="mock-b",
                output_path=str(output_path),
            )
        )

        self.assertEqual(response.intent, "run_experiment")
        payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["model_name"], "mock-b")


if __name__ == "__main__":
    unittest.main()
