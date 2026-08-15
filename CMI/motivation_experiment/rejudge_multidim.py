from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motivation_experiment.run_pilot import _bounded_score, _parse_bool, _plot, _quantile, summarize, write_jsonl, write_summary_csv
from src.api.openai_client import OpenAIClient
from src.utils.io import ensure_dir, load_config, write_json


def _multidim_judge(
    client: OpenAIClient,
    config: dict[str, Any],
    row: dict[str, Any],
    no_output: str,
    with_output: str,
    rollout_id: int,
) -> dict[str, Any]:
    prompt = f"""You are an impartial evaluator comparing two answers to the same task.

Task:
{row['task']}

Answer WITHOUT the candidate memory:
{no_output}

Answer WITH the candidate memory:
{with_output}

Assess how the answer changed, independently of whether either answer is correct.
Do not use a memory role label; no role label is provided.
Do not assess correctness and do not infer the expected answer. This is a behavioral
change measurement only. Extract what each answer actually claims, even when both
claims are wrong.

Use these dimensions:
- conclusion_change_score: did the final answer/conclusion change? 0=same conclusion, 0.5=partially changed, 1=different conclusion.
- factual_change_score: did the factual claims relevant to the task change? 0=same relevant facts, 0.5=some changed/added facts, 1=materially different facts. Do not count harmless wording changes.
- action_applicable: true only if the task involves a recommendation, decision, tool choice, or action; otherwise false.
- action_change_score: if action_applicable is true, use the same 0/0.5/1 scale for the action; otherwise return 0.
- conclusion_confidence, factual_confidence, action_confidence: confidence in each corresponding score, in [0, 1]. For a non-applicable action, action_confidence must be 0.

A refusal, "not enough information", or NO_ANSWER is a substantive conclusion. Two answers that are both NO_ANSWER have conclusion_change_score=0 unless one contains a different substantive claim. Do not copy field descriptions into the values.

Return JSON only:
{{
  "conclusion_change_score": 0.0,
  "conclusion_confidence": 0.0,
  "factual_change_score": 0.0,
  "factual_confidence": 0.0,
  "action_applicable": false,
  "action_change_score": 0.0,
  "action_confidence": 0.0,
  "conclusion_without": "short conclusion or NO_ANSWER",
  "conclusion_with": "short conclusion or NO_ANSWER",
  "explanation": "brief dimension-by-dimension comparison"
}}

All scores and confidence must be in [0,1].
"""
    result = client.complete(
        prompt,
        model=config.get("openai", {}).get("judge_model", "llama3:8b"),
        temperature=0.0,
        max_output_tokens=600,
        json_mode=True,
        metadata={
            "purpose": "motivation_multidim_rejudge_v1",
            "example_id": row["example_id"],
            "memory_id": row["memory_id"],
            "rollout_id": rollout_id,
        },
    )
    data = result.get("json") or {}
    action_applicable = _parse_bool(data.get("action_applicable"), False)
    conclusion = _bounded_score(data.get("conclusion_change_score"))
    factual = _bounded_score(data.get("factual_change_score"))
    action = _bounded_score(data.get("action_change_score"), 0.0)
    conclusion_confidence = _bounded_score(data.get("conclusion_confidence"), 0.0)
    factual_confidence = _bounded_score(data.get("factual_confidence"), 0.0)
    action_confidence = _bounded_score(data.get("action_confidence"), 0.0)
    errors: list[str] = []
    if conclusion is None or factual is None:
        errors.append("missing conclusion_change_score or factual_change_score")
    if action_applicable and action is None:
        errors.append("action_applicable=true but action_change_score is missing")
    if conclusion is not None and factual is not None:
        applicable_scores = [conclusion, factual] + ([action or 0.0] if action_applicable else [])
        computed_overall = statistics.fmean(applicable_scores)
    else:
        computed_overall = None
    applicable_confidences = [conclusion_confidence or 0.0, factual_confidence or 0.0]
    if action_applicable:
        applicable_confidences.append(action_confidence or 0.0)
    warnings: list[str] = []
    return {
        "conclusion_change_score": conclusion,
        "factual_change_score": factual,
        "action_applicable": action_applicable,
        "action_change_score": action if action is not None else 0.0,
        "overall_change_score": computed_overall,
        "conclusion_confidence": conclusion_confidence,
        "factual_confidence": factual_confidence,
        "action_confidence": action_confidence if action_applicable else 0.0,
        "overall_confidence": statistics.fmean(applicable_confidences),
        "conclusion_without": str(data.get("conclusion_without", "")),
        "conclusion_with": str(data.get("conclusion_with", "")),
        "explanation": str(data.get("explanation", "")),
        "valid": not errors,
        "validation_errors": errors,
        "normalization_warnings": warnings,
    }


def _b_distribution(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["behavioral_reliance"]) for row in rows]
    return {
        "min": min(values),
        "p20": _quantile(values, 0.2),
        "median": _quantile(values, 0.5),
        "p80": _quantile(values, 0.8),
        "max": max(values),
        "n_unique": len(set(values)),
        "exact_zero_count": sum(value == 0.0 for value in values),
        "exact_one_count": sum(value == 1.0 for value in values),
    }


def rejudge(input_dir: Path, output_dir: Path, config_path: Path, judge_model: str | None = None) -> dict[str, Any]:
    rows_path = input_dir / "memory_interventions.jsonl"
    rows = [json.loads(line) for line in rows_path.open(encoding="utf-8") if line.strip()]
    config = load_config(str(config_path))
    if judge_model:
        config.setdefault("openai", {})["judge_model"] = judge_model
    output_dir = ensure_dir(output_dir)
    client = OpenAIClient(
        use_cache=True,
        use_api=bool(config.get("openai", {}).get("use_api", True)),
        provider=config.get("openai", {}).get("provider"),
        base_url=config.get("openai", {}).get("base_url"),
        cache_dir=str(output_dir / "cache"),
    )
    invalid: list[dict[str, Any]] = []
    rejudged_rows: list[dict[str, Any]] = []
    for row in rows:
        no_outputs = row.get("no_memory_outputs", [])
        with_outputs = row.get("with_memory_outputs", [])
        judgments: list[dict[str, Any]] = []
        for rollout_id, (no_output, with_output) in enumerate(zip(no_outputs, with_outputs)):
            judgment = _multidim_judge(client, config, row, no_output, with_output, rollout_id)
            judgments.append(judgment)
            if not judgment["valid"]:
                invalid.append({"example_id": row["example_id"], "memory_id": row["memory_id"], "rollout_id": rollout_id, "errors": judgment["validation_errors"]})
        if invalid and any(item["example_id"] == row["example_id"] and item["memory_id"] == row["memory_id"] for item in invalid):
            continue
        b_values = [float(j["overall_change_score"]) for j in judgments if j.get("overall_change_score") is not None]
        if len(b_values) != len(no_outputs):
            invalid.append({"example_id": row["example_id"], "memory_id": row["memory_id"], "errors": ["not enough valid multidimensional scores"]})
            continue
        revised = dict(row)
        revised["old_behavioral_reliance"] = row.get("behavioral_reliance")
        revised["old_behavior_metric"] = row.get("behavior_metric")
        revised["behavior_metric"] = "multidim_decision"
        revised["behavioral_reliance"] = statistics.fmean(b_values)
        revised["behavioral_reliance_sd"] = statistics.pstdev(b_values) if len(b_values) > 1 else 0.0
        revised["behavior_changed"] = int(any(value > 0.05 for value in b_values))
        revised["multidim_B_values"] = b_values
        revised["multidim_judgments"] = judgments
        revised["multidim_B_conclusion"] = statistics.fmean(float(j["conclusion_change_score"]) for j in judgments)
        revised["multidim_B_factual"] = statistics.fmean(float(j["factual_change_score"]) for j in judgments)
        applicable_actions = [float(j["action_change_score"]) for j in judgments if j["action_applicable"]]
        revised["multidim_B_action"] = statistics.fmean(applicable_actions) if applicable_actions else None
        revised["multidim_action_applicable_rate"] = statistics.fmean(
            1.0 if j["action_applicable"] else 0.0 for j in judgments
        )
        revised["multidim_overall_confidence"] = statistics.fmean(float(j["overall_confidence"]) for j in judgments)
        revised["multidim_warning_count"] = sum(len(j.get("normalization_warnings", [])) for j in judgments)
        rejudged_rows.append(revised)
    if invalid:
        raise RuntimeError(f"{len(invalid)} multidimensional judge outputs were invalid. See first errors: {invalid[:5]}")

    old_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    old_thresholds = old_summary.get("thresholds", {})
    summary = summarize(
        rejudged_rows,
        float(old_thresholds.get("top_relevance_quantile", 0.8)),
        float(old_thresholds.get("high_B_quantile", 0.8)),
        float(old_thresholds.get("utility_epsilon", 0.0)),
        42,
    )
    summary["run"] = dict(old_summary.get("run", {}))
    summary["run"].update({
        "judge_model": config.get("openai", {}).get("judge_model"),
        "behavior_scorer": "multidim_decision",
        "rejudge_source": str(input_dir),
        "rejudge_rollouts": 5,
        "invalid_multidim_judgments": 0,
    })
    summary["diagnostic_comparison"] = {
        "old_behavior_metric": old_summary.get("overall", {}).get("behavior_metric"),
        "old_pearson_B_U": old_summary.get("correlations", {}).get("pearson_B_U"),
        "new_pearson_B_U": summary.get("correlations", {}).get("pearson_B_U"),
        "old_mean_B": old_summary.get("overall", {}).get("mean_B"),
        "new_mean_B": summary.get("overall", {}).get("mean_B"),
        "old_B_distribution": _b_distribution(rows),
        "new_B_distribution": _b_distribution(rejudged_rows),
        "new_warning_count": sum(row.get("multidim_warning_count", 0) for row in rejudged_rows),
        "new_mean_conclusion_B": statistics.fmean(float(row["multidim_B_conclusion"]) for row in rejudged_rows),
        "new_mean_factual_B": statistics.fmean(float(row["multidim_B_factual"]) for row in rejudged_rows),
        "new_mean_action_B": statistics.fmean(
            float(row["multidim_B_action"])
            for row in rejudged_rows
            if row["multidim_B_action"] is not None
        ) if any(row["multidim_B_action"] is not None for row in rejudged_rows) else None,
    }
    write_jsonl(rejudged_rows, output_dir / "memory_interventions.jsonl")
    write_summary_csv(summary, output_dir / "summary.csv")
    summary["figures"] = _plot(rejudged_rows, summary, ensure_dir(output_dir / "figures"))
    write_json(summary, output_dir / "summary.json")
    print(json.dumps({"n_rows": len(rejudged_rows), "output_dir": str(output_dir), "summary": summary["diagnostic_comparison"]}, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rejudge existing intervention outputs with a multidimensional behavior rubric.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--judge-model", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rejudge(args.input_dir, args.output_dir, args.config, args.judge_model)
