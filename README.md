# Multi-Agent VLM Orchestrator

`multi-agent-vlm-orchestrator` is an agent-oriented benchmark runner for Hugging Face vision-language
models. It supports both direct batch execution and a higher-level agent entrypoint that
interprets a natural-language request, selects tools, and triggers experiment runs.

## Agent structure

- `IntentRouterAgent`: maps a user request to an intent such as `run_experiment` or `list_models`.
- `ExecutionAgent`: executes tool calls against the script/model registry.
- `ResponseAgent`: turns tool outputs into a final user-facing response.
- `ExperimentRunner`: low-level runner used by the `run_experiment` tool.
- `SubAgent`: binds one script to one preferred VLM profile.
- `VLMClient`: backend adapter for Hugging Face model execution.

## Tools exposed to the agent

- `list_models`
- `list_scripts`
- `run_experiment`
- `summarize_results`

The current planner is rule-based so the system works locally without an external LLM.
The agent boundaries are explicit, so you can replace the planner later with LangGraph
or a tool-calling LLM without changing the tool layer.

## Quick start

```bash
cd /home/larry5/project/multi-agent-vlm-orchestrator
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli validate \
  --models configs/models.json \
  --scripts configs/scripts.json
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli agent \
  --request "list models" \
  --models configs/models.json \
  --scripts configs/scripts.json
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli agent \
  --request "run script 1 and script 3 on this prompt: describe the image" \
  --models configs/models.json \
  --scripts configs/scripts.json
```

## Direct batch mode

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli run \
  --experiment configs/experiment.json \
  --models configs/models.json \
  --scripts configs/scripts.json
```

## Backends

- `mock`: dry-run backend for pipeline testing
- `transformers_local`: local Hugging Face execution through `transformers`

Install local inference dependencies when you need real VLM execution:

```bash
uv sync --extra dev --extra local
```

## Config model

`models.json` stores your preferred VLM profiles. `conda_env` is optional metadata for
future per-model environment routing.

```json
{
  "models": {
    "qwen2-vl-2b": {
      "provider": "huggingface",
      "backend": "transformers_local",
      "model_id": "Qwen/Qwen2-VL-2B-Instruct",
      "conda_env": "vlm-qwen",
      "device": "cuda",
      "dtype": "bfloat16",
      "max_new_tokens": 256
    }
  }
}
```

`scripts.json` maps each script to its preferred model.

```json
{
  "scripts": {
    "script_001": {
      "description": "OCR and chart understanding",
      "preferred_model": "qwen2-vl-2b",
      "prompt_template": "Analyze the image and answer: {user_prompt}"
    }
  }
}
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```
