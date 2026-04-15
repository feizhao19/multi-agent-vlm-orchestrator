#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from multi_agent_vlm_orchestrator.agents import SubAgent
from multi_agent_vlm_orchestrator.clients import build_client
from multi_agent_vlm_orchestrator.config import load_models_config
from multi_agent_vlm_orchestrator.models import AgentTask, TaskMode
from multi_agent_vlm_orchestrator.registry import ModelRegistry


ROOT = Path("/home/larry5/project/multi-agent-vlm-orchestrator")
DEFAULT_MODELS = ROOT / "configs" / "models_face_persona.json"
DEFAULT_DATASETS = [
    ROOT / "data" / "older-adult_unique_face_prompts_json_response_young_adult_persona.json",
    ROOT / "data" / "older-adult_unique_face_prompts_json_response_older_adult_persona.json",
]
DEFAULT_OUTPUT = ROOT / "results" / "face_persona_experiment_results.jsonl"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text.strip():
        return None
    candidates = [text.strip()]
    fenced = text.strip().replace("```json", "```")
    if "```" in fenced:
        parts = fenced.split("```")
        candidates.extend(part.strip() for part in parts if part.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _normalize_answers(payload: dict[str, Any]) -> dict[str, Any]:
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in answers.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                normalized[key] = int(stripped)
            else:
                normalized[key] = stripped
        else:
            normalized[key] = value
    return normalized


def _load_existing_keys(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        keys.add((record["model_name"], record["persona_id"], record["sample_id"]))
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run face persona comparison experiments.")
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--print-response", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    models_config = load_models_config(args.models_config)
    model_registry = ModelRegistry(models_config)
    dataset_paths = args.dataset or DEFAULT_DATASETS
    model_names = args.model_name or list(models_config.models.keys())
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = _load_existing_keys(output_path)

    with output_path.open("a", encoding="utf-8") as handle:
        for model_name in model_names:
            model_registry.validate_task_mode(model_name, TaskMode.TEXT_ONLY)
            profile = model_registry.get(model_name)
            agent = SubAgent(
                model_name=model_name,
                profile=profile,
                client=build_client(profile),
            )
            for dataset_path in dataset_paths:
                payload = json.loads(dataset_path.read_text(encoding="utf-8"))
                persona_id = payload.get("persona_id", dataset_path.stem)
                persona_label = payload.get("persona_label", persona_id)
                samples = payload["samples"]
                if args.sample_limit is not None:
                    samples = samples[: args.sample_limit]
                for index, sample in enumerate(samples, start=1):
                    sample_id = sample["unique_face_id"]
                    record_key = (model_name, persona_id, sample_id)
                    if record_key in completed:
                        continue

                    print(
                        f"[start] loading model={model_name} persona={persona_id} sample={sample_id}",
                        flush=True,
                    )
                    if args.print_prompt:
                        print("[prompt]")
                        print(sample["prompt_json"], flush=True)

                    task = AgentTask(
                        script_id=f"face_persona_{persona_id}",
                        model_name=model_name,
                        prompt=sample["prompt_json"],
                        task_mode=TaskMode.TEXT_ONLY,
                        description=f"Face persona evaluation for {persona_label}",
                    )
                    started_at = time.time()
                    result = agent.run(task)
                    elapsed_seconds = round(time.time() - started_at, 3)
                    if args.print_response:
                        print("[response]")
                        print(result.response_text, flush=True)
                    parsed_response = _extract_json_object(result.response_text)
                    normalized_answers = _normalize_answers(parsed_response or {})
                    expected_questions = [question["id"] for question in sample["questions"]]
                    missing_answers = [
                        question_id for question_id in expected_questions if question_id not in normalized_answers
                    ]
                    extra_answers = [
                        question_id for question_id in normalized_answers if question_id not in expected_questions
                    ]
                    record = {
                        "run_group": "face_persona_experiment",
                        "dataset_file": str(dataset_path),
                        "persona_id": persona_id,
                        "persona_label": persona_label,
                        "sample_id": sample_id,
                        "prompt_text": sample["prompt_json"],
                        "prompt_sha256": hashlib.sha256(
                            sample["prompt_json"].encode("utf-8")
                        ).hexdigest(),
                        "question_ids": expected_questions,
                        "result_index": index,
                        "elapsed_seconds": elapsed_seconds,
                        "json_valid": parsed_response is not None,
                        "parsed_response": parsed_response,
                        "normalized_answers": normalized_answers,
                        "missing_answers": missing_answers,
                        "extra_answers": extra_answers,
                        **result.model_dump(),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.add(record_key)
                    status = "ok" if result.success else "error"
                    print(
                        f"[{status}] model={model_name} persona={persona_id} sample={sample_id} "
                        f"json_valid={record['json_valid']} elapsed={elapsed_seconds}s"
                    )

    print(f"Wrote experiment results to {output_path}")


if __name__ == "__main__":
    main()
