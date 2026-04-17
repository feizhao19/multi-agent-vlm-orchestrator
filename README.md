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

## Two-stage avatar baseline

The single-stage `run` command is the Stage A baseline: render the configured prompt and
send it directly to the selected model.

Use `run-two-stage` for Stage B: render the same prompt, generate an avatar image, then
send that generated image plus the same prompt to an evaluator VLM.

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli run-two-stage \
  --experiment configs/experiment_two_stage.json \
  --models configs/models.json \
  --scripts configs/scripts.json
```

Two-stage records are JSONL objects containing the baseline answer, image path and
metadata, avatar-conditioned answer, failure reason, and elapsed time.

`experiment.samples` lets you batch multiple face/persona prompts in one run. Generated
avatars are written under:

```text
<image_output_dir>/<sample_id>/<generator_model>/<persona_id>/avatar.<ext>
```

Compare a Stage A result file with a Stage B result file:

```bash
PYTHONPATH=src python3 -m multi_agent_vlm_orchestrator.cli compare \
  --baseline results/demo_run.jsonl \
  --two-stage results/two_stage_face_avatar_smoke.jsonl \
  --output results/two_stage_comparison.json
```

`configs/models.json` includes a `bagel-7b-mot` profile for
`ByteDance-Seed/BAGEL-7B-MoT`. BAGEL is wired as `external_command` because its
any-to-any inference path needs a local wrapper around the official code. Fill
`extra.generate_image_command` and `extra.generate_command` before running it.
The included `scripts/run_bagel.py` wrapper expects:

```bash
export BAGEL_REPO_PATH=/path/to/official/BAGEL
export BAGEL_MODEL_PATH=/path/to/downloaded/BAGEL-7B-MoT
```

## Backends

- `mock`: dry-run backend for pipeline testing
- `transformers_local`: local Hugging Face execution through `transformers`
- `diffusers_local`: local text-to-image execution through `diffusers`
- `external_command`: adapter for model-specific scripts such as BAGEL wrappers

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
