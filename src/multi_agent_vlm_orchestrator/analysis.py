from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _json_answer_stats(records: list[dict[str, Any]], field_path: list[str]) -> dict[str, int]:
    parsed = 0
    missing = 0
    for record in records:
        value: Any = record
        for key in field_path:
            value = value.get(key) if isinstance(value, dict) else None
        if not isinstance(value, str) or not value.strip():
            missing += 1
            continue
        if _try_parse_json(value) is not None:
            parsed += 1
    return {"json_parseable": parsed, "missing_or_empty": missing}


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _stage_model_counts(records: list[dict[str, Any]], stage: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        stage_record = record.get(stage)
        if not isinstance(stage_record, dict):
            continue
        model_name = str(stage_record.get("model_name") or "unknown")
        counts[model_name] = counts.get(model_name, 0) + 1
    return dict(sorted(counts.items()))


def compare_baseline_and_two_stage(
    baseline_path: Path | None,
    two_stage_path: Path,
) -> dict[str, Any]:
    two_stage_records = _read_jsonl(two_stage_path)
    baseline_records = _read_jsonl(baseline_path) if baseline_path is not None else []

    two_stage_successes = sum(1 for record in two_stage_records if record.get("success"))
    baseline_successes = sum(1 for record in baseline_records if record.get("success"))
    embedded_baseline_successes = sum(
        1
        for record in two_stage_records
        if record.get("baseline", {}).get("success")
    )
    avatar_successes = sum(
        1
        for record in two_stage_records
        if record.get("avatar_answer", {}).get("success")
    )
    image_successes = sum(
        1
        for record in two_stage_records
        if record.get("image_generation", {}).get("success")
    )

    summary: dict[str, Any] = {
        "two_stage_path": str(two_stage_path),
        "two_stage_total": len(two_stage_records),
        "two_stage_successes": two_stage_successes,
        "two_stage_failures": len(two_stage_records) - two_stage_successes,
        "embedded_baseline_successes": embedded_baseline_successes,
        "image_generation_successes": image_successes,
        "avatar_answer_successes": avatar_successes,
        "records_by_sample": _count_by(two_stage_records, "sample_id"),
        "records_by_persona": _count_by(two_stage_records, "persona_id"),
        "baseline_models": _stage_model_counts(two_stage_records, "baseline"),
        "avatar_answer_models": _stage_model_counts(two_stage_records, "avatar_answer"),
        "embedded_baseline_json": _json_answer_stats(
            two_stage_records,
            ["baseline", "response_text"],
        ),
        "avatar_answer_json": _json_answer_stats(
            two_stage_records,
            ["avatar_answer", "response_text"],
        ),
    }
    if baseline_path is not None:
        summary.update(
            {
                "baseline_path": str(baseline_path),
                "baseline_total": len(baseline_records),
                "baseline_successes": baseline_successes,
                "baseline_failures": len(baseline_records) - baseline_successes,
                "baseline_json": _json_answer_stats(baseline_records, ["response_text"]),
            }
        )
    return summary
