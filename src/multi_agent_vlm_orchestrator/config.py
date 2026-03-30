from __future__ import annotations

import json
from pathlib import Path

from multi_agent_vlm_orchestrator.models import ExperimentConfig, ModelsConfig, ScriptsConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            data = json.load(handle)
        elif path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config requires PyYAML. Install with: uv sync --extra yaml"
                ) from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported config format '{path.suffix}'")
    return data or {}


def load_models_config(path: str | Path) -> ModelsConfig:
    return ModelsConfig.model_validate(_load_yaml(Path(path)))


def load_scripts_config(path: str | Path) -> ScriptsConfig:
    return ScriptsConfig.model_validate(_load_yaml(Path(path)))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_yaml(Path(path)))
