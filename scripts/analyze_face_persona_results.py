#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/larry5/project/multi-agent-vlm-orchestrator")
DEFAULT_INPUT = ROOT / "results" / "face_persona_experiment_results.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "face_persona_analysis"


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_numeric_answer(value: Any) -> bool:
    return isinstance(value, int) or isinstance(value, float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze face persona experiment results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Results file not found: {args.input}")
    records = _read_records(args.input)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_name"], record["persona_id"])].append(record)

    for (model_name, persona_id), group in sorted(grouped.items()):
        success_count = sum(1 for record in group if record["success"])
        json_valid_count = sum(1 for record in group if record["json_valid"])
        fully_answered_count = sum(
            1
            for record in group
            if record["json_valid"] and not record["missing_answers"] and not record["extra_answers"]
        )
        overview_rows.append(
            {
                "model_name": model_name,
                "persona_id": persona_id,
                "persona_label": group[0]["persona_label"],
                "total_records": len(group),
                "success_count": success_count,
                "success_rate": round(success_count / len(group), 4),
                "json_valid_count": json_valid_count,
                "json_valid_rate": round(json_valid_count / len(group), 4),
                "fully_answered_count": fully_answered_count,
                "fully_answered_rate": round(fully_answered_count / len(group), 4),
                "mean_elapsed_seconds": round(
                    sum(record["elapsed_seconds"] for record in group) / len(group),
                    4,
                ),
            }
        )

        by_question: dict[str, list[Any]] = defaultdict(list)
        for record in group:
            for question_id, answer in record.get("normalized_answers", {}).items():
                by_question[question_id].append(answer)

        for question_id, answers in sorted(by_question.items()):
            numeric_answers = [answer for answer in answers if _is_numeric_answer(answer)]
            categorical_counts = Counter(
                str(answer) for answer in answers if not _is_numeric_answer(answer)
            )
            question_rows.append(
                {
                    "model_name": model_name,
                    "persona_id": persona_id,
                    "question_id": question_id,
                    "answer_count": len(answers),
                    "numeric_answer_count": len(numeric_answers),
                    "numeric_mean": round(sum(numeric_answers) / len(numeric_answers), 4)
                    if numeric_answers
                    else "",
                    "categorical_distribution": json.dumps(
                        dict(sorted(categorical_counts.items())),
                        ensure_ascii=False,
                    ),
                }
            )

    paired_records: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        paired_records[(record["model_name"], record["sample_id"])][record["persona_id"]] = record

    for (model_name, sample_id), persona_map in sorted(paired_records.items()):
        if "young_adult" not in persona_map or "older_adult" not in persona_map:
            continue
        young = persona_map["young_adult"]
        older = persona_map["older_adult"]
        all_question_ids = sorted(set(young["question_ids"]) | set(older["question_ids"]))
        for question_id in all_question_ids:
            young_answer = young.get("normalized_answers", {}).get(question_id)
            older_answer = older.get("normalized_answers", {}).get(question_id)
            if young_answer is None or older_answer is None:
                continue
            diff_numeric = ""
            if _is_numeric_answer(young_answer) and _is_numeric_answer(older_answer):
                diff_numeric = older_answer - young_answer
            comparison_rows.append(
                {
                    "model_name": model_name,
                    "sample_id": sample_id,
                    "question_id": question_id,
                    "young_answer": young_answer,
                    "older_answer": older_answer,
                    "answers_match": young_answer == older_answer,
                    "numeric_difference_older_minus_young": diff_numeric,
                }
            )

    comparison_summary_rows: list[dict[str, Any]] = []
    comparison_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        comparison_grouped[(row["model_name"], row["question_id"])].append(row)

    for (model_name, question_id), rows in sorted(comparison_grouped.items()):
        numeric_differences = [
            row["numeric_difference_older_minus_young"]
            for row in rows
            if isinstance(row["numeric_difference_older_minus_young"], (int, float))
        ]
        comparison_summary_rows.append(
            {
                "model_name": model_name,
                "question_id": question_id,
                "pair_count": len(rows),
                "match_rate": round(
                    sum(1 for row in rows if row["answers_match"]) / len(rows),
                    4,
                ),
                "mean_numeric_difference_older_minus_young": round(
                    sum(numeric_differences) / len(numeric_differences),
                    4,
                )
                if numeric_differences
                else "",
            }
        )

    _write_csv(
        output_dir / "model_persona_overview.csv",
        overview_rows,
        [
            "model_name",
            "persona_id",
            "persona_label",
            "total_records",
            "success_count",
            "success_rate",
            "json_valid_count",
            "json_valid_rate",
            "fully_answered_count",
            "fully_answered_rate",
            "mean_elapsed_seconds",
        ],
    )
    _write_csv(
        output_dir / "question_summary.csv",
        question_rows,
        [
            "model_name",
            "persona_id",
            "question_id",
            "answer_count",
            "numeric_answer_count",
            "numeric_mean",
            "categorical_distribution",
        ],
    )
    _write_csv(
        output_dir / "persona_pairwise_comparison.csv",
        comparison_rows,
        [
            "model_name",
            "sample_id",
            "question_id",
            "young_answer",
            "older_answer",
            "answers_match",
            "numeric_difference_older_minus_young",
        ],
    )
    _write_csv(
        output_dir / "persona_comparison_summary.csv",
        comparison_summary_rows,
        [
            "model_name",
            "question_id",
            "pair_count",
            "match_rate",
            "mean_numeric_difference_older_minus_young",
        ],
    )

    summary_payload = {
        "input_path": str(args.input),
        "record_count": len(records),
        "models": sorted({record["model_name"] for record in records}),
        "personas": sorted({record["persona_id"] for record in records}),
        "output_files": [
            "model_persona_overview.csv",
            "question_summary.csv",
            "persona_pairwise_comparison.csv",
            "persona_comparison_summary.csv",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
