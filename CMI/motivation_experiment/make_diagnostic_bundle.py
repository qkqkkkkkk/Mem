from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_bundle(input_dir: Path, output_dir: Path) -> None:
    rows_path = input_dir / "memory_interventions.jsonl"
    rows = [json.loads(line) for line in rows_path.open(encoding="utf-8") if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_rows: list[dict[str, object]] = []
    for row in rows:
        no_outputs = row.get("no_memory_outputs", [])
        with_outputs = row.get("with_memory_outputs", [])
        for rollout_id, judgment in enumerate(row.get("llm_judgments", [])):
            warnings = judgment.get("normalization_warnings", [])
            if not warnings:
                continue
            warning_rows.append(
                {
                    "example_id": row["example_id"],
                    "memory_id": row["memory_id"],
                    "rollout_id": rollout_id,
                    "task": row["task"],
                    "memory_content": row["memory_content"],
                    "no_memory_output": no_outputs[rollout_id] if rollout_id < len(no_outputs) else "",
                    "with_memory_output": with_outputs[rollout_id] if rollout_id < len(with_outputs) else "",
                    "decision_without": judgment.get("decision_without", ""),
                    "decision_with": judgment.get("decision_with", ""),
                    "reported_same_decision": judgment.get("reported_same_decision", ""),
                    "normalized_same_decision": judgment.get("same_decision", ""),
                    "decision_change_score": judgment.get("decision_change_score", ""),
                    "confidence": judgment.get("confidence", ""),
                    "explanation": judgment.get("explanation", ""),
                    "warnings": "; ".join(warnings),
                }
            )

    warning_fields = [
        "example_id", "memory_id", "rollout_id", "task", "memory_content",
        "no_memory_output", "with_memory_output", "decision_without", "decision_with",
        "reported_same_decision", "normalized_same_decision", "decision_change_score", "confidence",
        "explanation", "warnings",
        "human_conclusion_change_score", "human_factual_change_score", "human_action_applicable",
        "human_action_change_score", "human_notes", "reviewer", "review_status",
    ]
    _write_csv(output_dir / "judge_warning_review.csv", warning_rows, warning_fields)

    # A single question-memory pair has up to five stochastic rollouts. This compact
    # file is the unit a reviewer should inspect, rather than treating repeats as
    # independent manual-review items.
    warning_pairs: dict[tuple[str, str], list[dict[str, object]]] = {}
    for warning in warning_rows:
        warning_pairs.setdefault((str(warning["example_id"]), str(warning["memory_id"])), []).append(warning)
    pair_rows: list[dict[str, object]] = []
    for pair_warnings in warning_pairs.values():
        first = pair_warnings[0]
        pair_rows.append(
            {
                "example_id": first["example_id"],
                "memory_id": first["memory_id"],
                "warning_rollouts": ";".join(str(item["rollout_id"]) for item in pair_warnings),
                "task": first["task"],
                "memory_content": first["memory_content"],
                "paired_outputs": json.dumps(
                    [
                        {
                            "rollout_id": item["rollout_id"],
                            "without": item["no_memory_output"],
                            "with": item["with_memory_output"],
                            "old_reported_same": item["reported_same_decision"],
                            "old_normalized_same": item["normalized_same_decision"],
                            "old_score": item["decision_change_score"],
                            "old_explanation": item["explanation"],
                        }
                        for item in pair_warnings
                    ],
                    ensure_ascii=False,
                ),
                "human_conclusion_change_score": "",
                "human_factual_change_score": "",
                "human_action_applicable": "",
                "human_action_change_score": "",
                "human_notes": "",
                "reviewer": "",
                "review_status": "pending",
            }
        )
    pair_fields = [
        "example_id", "memory_id", "warning_rollouts", "task", "memory_content",
        "paired_outputs", "human_conclusion_change_score", "human_factual_change_score",
        "human_action_applicable", "human_action_change_score", "human_notes", "reviewer", "review_status",
    ]
    _write_csv(output_dir / "judge_warning_pair_review.csv", pair_rows, pair_fields)

    anomaly_rows: list[dict[str, object]] = []
    for row in rows:
        tags: list[str] = []
        if row.get("label") == "harmful":
            tags.append("all_harmful")
        if row.get("label") == "harmful" and float(row.get("utility", 0.0)) > 0.0:
            tags.append("harmful_positive_utility")
        if row.get("label") == "harmful" and float(row.get("utility", 0.0)) >= 0.25:
            tags.append("harmful_positive_outlier")
        if row.get("utility_negative_with_95ci"):
            tags.append("negative_utility_95ci")
        if not tags:
            continue
        anomaly_rows.append(
            {
                "review_tags": ";".join(tags),
                "example_id": row["example_id"],
                "memory_id": row["memory_id"],
                "label_for_internal_review": row.get("label", ""),
                "task": row["task"],
                "gold_behavior": row.get("gold_behavior", ""),
                "memory_content": row["memory_content"],
                "relevance_score": row.get("relevance_score", ""),
                "old_B": row.get("behavioral_reliance", ""),
                "utility": row.get("utility", ""),
                "utility_ci_lower": row.get("utility_ci_lower", ""),
                "utility_ci_upper": row.get("utility_ci_upper", ""),
                "deterministic_utility": row.get("deterministic_utility", ""),
                "llm_utility": row.get("llm_utility", ""),
                "no_memory_outputs": json.dumps(row.get("no_memory_outputs", []), ensure_ascii=False),
                "with_memory_outputs": json.dumps(row.get("with_memory_outputs", []), ensure_ascii=False),
                "llm_judgments": json.dumps(row.get("llm_judgments", []), ensure_ascii=False),
                "human_memory_role": "",
                "human_utility_interpretation": "",
                "human_notes": "",
                "reviewer": "",
                "review_status": "pending",
            }
        )
    anomaly_fields = [
        "review_tags", "example_id", "memory_id", "label_for_internal_review", "task", "gold_behavior",
        "memory_content", "relevance_score", "old_B", "utility", "utility_ci_lower", "utility_ci_upper",
        "deterministic_utility", "llm_utility", "no_memory_outputs", "with_memory_outputs", "llm_judgments",
        "human_memory_role", "human_utility_interpretation", "human_notes", "reviewer", "review_status",
    ]
    _write_csv(output_dir / "anomaly_review.csv", anomaly_rows, anomaly_fields)

    (output_dir / "README.md").write_text(
        "# 人工审查说明\n\n"
        "- `judge_warning_review.csv`：54 条 rollout 级 warning 的完整明细，用来追溯旧 judge 的原始矛盾；不含 construction label。\n"
        "- `judge_warning_pair_review.csv`：将重复 rollout 合并到 question-memory pair；这是建议的人工审查单位，且不含 construction label。填写四个 `human_*` 分数（0、0.5、1），并标记 `review_status`。\n"
        "- `anomaly_review.csv`：全部 harmful memory、utility 置信区间完全为负的样本，以及 harmful 但正 utility 的离群点。`harmful_positive_outlier` 是最优先核查对象。\n\n"
        "人工审查只判断两份模型输出实际是否发生结论、事实或行动变化；不要根据 construction label 判断。`human_memory_role` 仅在看完 task、memory、输出后填写（例如 label error、judge error、model resisted harmful memory、unclear）。\n",
        encoding="utf-8",
    )
    print(json.dumps({"warnings": len(warning_rows), "warning_pairs": len(pair_rows), "anomaly_rows": len(anomaly_rows), "output_dir": str(output_dir)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export judge-warning and utility-anomaly review files from an existing run.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_bundle(args.input_dir, args.output_dir or args.input_dir / "diagnostics")
