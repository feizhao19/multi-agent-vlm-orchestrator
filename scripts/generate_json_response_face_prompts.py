#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("/home/larry5/project/multi-agent-vlm-orchestrator")
INPUT_PATH = ROOT / "data" / "older-adult_unique_face_prompts.json"
OUTPUT_PATH = ROOT / "data" / "older-adult_unique_face_prompts_json_response.json"


def build_prompt(sample: dict) -> str:
    face_id = sample["unique_face_id"]
    description = sample["description"]
    questions = sample["questions"]

    schema_lines = [
        "{",
        f'  "unique_face_id": "{face_id}",',
        '  "answers": {',
    ]
    for index, question in enumerate(questions):
        comma = "," if index < len(questions) - 1 else ""
        schema_lines.append(f'    "{question["id"]}": "<answer>"{comma}')
    schema_lines.extend(["  }", "}"])

    prompt_lines = [
        "请根据以下面孔描述回答所有问题。",
        "你必须只输出严格的 JSON 对象，不要输出任何解释、分析、Markdown、代码块或额外文字。",
        "回答规则:",
        "1. 所有题目都必须回答，不能遗漏。",
        '2. 对于带有 A/B/C 等选项的题目，答案只输出选项字母，例如 "A"。',
        "3. 对于 1-9 评分题，答案只输出 1 到 9 的整数。",
        "4. 输出 JSON 的 key 必须与下面给定的格式完全一致。",
        f"面孔编号: {face_id}",
        f"描述: {description}",
        "问题:",
    ]
    prompt_lines.extend(f'{question["id"]}: {question["text"]}' for question in questions)
    prompt_lines.append("输出 JSON 格式:")
    prompt_lines.extend(schema_lines)
    return "\n".join(prompt_lines)


def main() -> None:
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    samples = payload["samples"]

    response_schema = {
        "unique_face_id": "string",
        "answers": {question["id"]: "string_or_integer" for question in samples[0]["questions"]},
    }

    output_samples = []
    for sample in samples:
        output_samples.append(
            {
                "unique_face_id": sample["unique_face_id"],
                "description": sample["description"],
                "questions": sample["questions"],
                "response_schema": {
                    "unique_face_id": sample["unique_face_id"],
                    "answers": {question["id"]: "<answer>" for question in sample["questions"]},
                },
                "prompt_json": build_prompt(sample),
            }
        )

    output_payload = {
        "source_file": payload["source_file"],
        "derived_from": INPUT_PATH.name,
        "sheet_name": payload["sheet_name"],
        "sample_count": payload["sample_count"],
        "question_count_per_sample": payload["question_count_per_sample"],
        "response_format_note": "Multiple-choice questions should return option letters only; rating questions should return integers from 1 to 9.",
        "response_schema": response_schema,
        "samples": output_samples,
    }

    OUTPUT_PATH.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Sample count: {len(output_samples)}")


if __name__ == "__main__":
    main()
