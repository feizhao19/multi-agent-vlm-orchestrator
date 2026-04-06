from __future__ import annotations

from multi_agent_vlm_orchestrator.models import (
    ModelProfile,
    ModelsConfig,
    ScriptDefinition,
    ScriptsConfig,
    TaskMode,
)


class ModelRegistry:
    def __init__(self, config: ModelsConfig) -> None:
        self._models = config.models

    def items(self) -> list[tuple[str, ModelProfile]]:
        return list(self._models.items())

    def get(self, name: str) -> ModelProfile:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model profile '{name}'") from exc

    def validate_script_preferences(self, scripts: ScriptsConfig) -> None:
        missing = [
            script_id
            for script_id, script in scripts.scripts.items()
            if script.preferred_model not in self._models
        ]
        if missing:
            details = ", ".join(sorted(missing))
            raise ValueError(f"Scripts reference undefined preferred models: {details}")

    def validate_task_mode(self, model_name: str, task_mode: TaskMode) -> None:
        profile = self.get(model_name)
        if not profile.capabilities.supports_mode(task_mode):
            raise ValueError(
                f"Model '{model_name}' does not support task mode '{task_mode.value}'"
            )


class ScriptRegistry:
    def __init__(self, config: ScriptsConfig) -> None:
        self._scripts = config.scripts

    def items(self) -> list[tuple[str, ScriptDefinition]]:
        return list(self._scripts.items())

    def get(self, script_id: str) -> ScriptDefinition:
        try:
            return self._scripts[script_id]
        except KeyError as exc:
            raise KeyError(f"Unknown script '{script_id}'") from exc

    def select(self, script_ids: list[str] | None) -> dict[str, ScriptDefinition]:
        if script_ids is None:
            return dict(self._scripts)
        return {script_id: self.get(script_id) for script_id in script_ids}
