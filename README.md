# Multi-Agent VLM Orchestrator

`multi-agent-vlm-orchestrator` is a multi-agent system for multimodal model orchestration. A supervisor agent accepts a request, routes it to the requested worker model, and returns the result through a consistent tool-based execution layer.

## Overview

- `1 Supervisor Agent`
  Accepts `script_id`, `prompt`, and optional `model_name`, validates the request, and routes work to the correct worker. The supervisor can be rule-based or a text-only LLM such as `qwen3`.
- `N Worker Agents`
  Each worker is bound to one unique model and runs inference through a backend adapter.
- `Tool Layer`
  Provides reusable actions such as `list_models`, `list_scripts`, `run_experiment`, and `summarize_results`.
- `Execution Layer`
  Handles script rendering, model loading, inference, and result writing.

This is the current routing contract:

1. user sends `script_id`
2. user sends `prompt`
3. user may send `model_name`
4. if `model_name` is present, the supervisor routes to that worker
5. otherwise the supervisor falls back to the script's `preferred_model`

## Architecture

```text
User Request
  -> Supervisor Agent
  -> Worker Selection
  -> Model Worker Agent
  -> Hugging Face Backend
  -> Result Store
```

Current implementation components:

- `IntentRouterAgent`
- `ExecutionAgent`
- `ResponseAgent`
- `ExperimentRunner`
- `SubAgent`
- `VLMClient`

The planner is currently rule-based. The system is intentionally structured so the planner can later be replaced by LangGraph or another tool-calling LLM layer without rewriting the worker execution path.

## Tech Stack

- `Python`
- `Pydantic` for configs and structured request models
- `Typer` for CLI entrypoints
- `Hugging Face transformers` for local LLM/VLM inference
- `JSON` and `JSONL` for configs and run outputs
- `Conda` metadata in model profiles for future per-model environment routing

## Request Modes

Natural-language agent request:

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli agent \
  --request "run script 1 with qwen2-vl-2b on this prompt: describe the image" \
  --models configs/models.json \
  --scripts configs/scripts.json
```

Natural-language request with `qwen3` as supervisor:

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli agent \
  --request "run script 1 with qwen2-vl-2b on this prompt: describe the image" \
  --supervisor-model qwen3-8b \
  --models configs/models.json \
  --scripts configs/scripts.json
```

Structured supervisor request:

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli supervisor-run \
  --script-id script_001 \
  --model-name qwen2-vl-2b \
  --prompt "describe the image" \
  --models configs/models.json \
  --scripts configs/scripts.json
```

Structured routing priority:

1. `model_name` provided by the user
2. `preferred_model` from `scripts.json`

## Quick Start

```bash
cd /home/larry5/project/multi-agent-vlm-orchestrator
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli validate \
  --models configs/models.json \
  --scripts configs/scripts.json
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli agent \
  --request "list models" \
  --models configs/models.json \
  --scripts configs/scripts.json
```

Direct batch mode is also available:

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli run \
  --experiment configs/experiment.json \
  --models configs/models.json \
  --scripts configs/scripts.json
```

## Model Backends

- `mock`
  Dry-run backend for pipeline validation
- `transformers_local`
  Local Hugging Face VLM execution through `transformers`

Install local model dependencies when you need real inference:

```bash
uv sync --extra dev --extra local
```

## Config Files

`configs/models.json` defines the available worker and supervisor model profiles:

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

`configs/scripts.json` maps scripts to their default model preferences:

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

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```
