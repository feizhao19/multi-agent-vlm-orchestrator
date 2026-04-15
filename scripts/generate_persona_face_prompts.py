#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path("/home/larry5/project/multi-agent-vlm-orchestrator")
INPUT_PATH = ROOT / "data" / "older-adult_unique_face_prompts_json_response.json"

PERSONAS = [
    {
        "persona_id": "young_adult",
        "persona_label": "健康的青壮年人",
        "instruction": "请你模拟一位健康的青壮年人，从这个身份视角完成下面的面孔判断任务。",
        "output_path": ROOT / "data" / "older-adult_unique_face_prompts_json_response_young_adult_persona.json",
    },
    {
        "persona_id": "older_adult",
        "persona_label": "健康的老年人",
        "instruction": "请你模拟一位健康的老年人，从这个身份视角完成下面的面孔判断任务。",
        "output_path": ROOT / "data" / "older-adult_unique_face_prompts_json_response_older_adult_persona.json",
    },
]


def add_persona_to_prompt(prompt_text: str, persona_instruction: str) -> str:
    lines = prompt_text.splitlines()
    if not lines:
        return persona_instruction
    return "\n".join([persona_instruction, lines[0], *lines[1:]])


def main() -> None:
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    for persona in PERSONAS:
        output_samples = []
        for sample in payload["samples"]:
            output_samples.append(
                {
                    **sample,
                    "persona_id": persona["persona_id"],
                    "persona_label": persona["persona_label"],
                    "prompt_json": add_persona_to_prompt(sample["prompt_json"], persona["instruction"]),
                }
            )

        output_payload = {
            **payload,
            "derived_from": INPUT_PATH.name,
            "persona_id": persona["persona_id"],
            "persona_label": persona["persona_label"],
            "persona_instruction": persona["instruction"],
            "samples": output_samples,
        }

        persona["output_path"].write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f'Wrote {persona["output_path"]}')
        print(f'Sample count: {len(output_samples)}')


if __name__ == "__main__":
    main()
