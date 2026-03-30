import unittest

from multi_agent_vlm_orchestrator.models import (
    ModelProfile,
    ModelsConfig,
    ScriptDefinition,
    ScriptsConfig,
)
from multi_agent_vlm_orchestrator.registry import ModelRegistry, ScriptRegistry


class RegistryTests(unittest.TestCase):
    def test_validate_script_preferences(self) -> None:
        models = ModelsConfig(
            models={
                "mock": ModelProfile(backend="mock", model_id="mock/model"),
            }
        )
        scripts = ScriptsConfig(
            scripts={
                "script_001": ScriptDefinition(
                    description="test",
                    preferred_model="mock",
                    prompt_template="{user_prompt}",
                )
            }
        )
        registry = ModelRegistry(models)
        registry.validate_script_preferences(scripts)

    def test_script_selection(self) -> None:
        scripts = ScriptsConfig(
            scripts={
                "script_001": ScriptDefinition(
                    description="a",
                    preferred_model="mock",
                    prompt_template="{user_prompt}",
                ),
                "script_002": ScriptDefinition(
                    description="b",
                    preferred_model="mock",
                    prompt_template="{user_prompt}",
                ),
            }
        )
        registry = ScriptRegistry(scripts)
        selected = registry.select(["script_002"])
        self.assertEqual(list(selected), ["script_002"])


if __name__ == "__main__":
    unittest.main()
