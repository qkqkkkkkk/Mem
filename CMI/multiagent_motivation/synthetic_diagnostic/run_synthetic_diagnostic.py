from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagent_motivation.run_team_pilot import _call_team, _score_team_answer
from src.api.openai_client import OpenAIClient
from src.benchmark.schema import ScoringCriteria
from src.utils.io import ensure_dir, load_config, write_json, write_jsonl


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _example(case: dict[str, Any]) -> Any:
    return SimpleNamespace(
        example_id=case["case_id"],
        task_family=case["task_family"],
        current_task=SimpleNamespace(instruction=case["task"]),
        gold_behavior=case["gold_behavior"],
        scoring_criteria=ScoringCriteria(**case["scoring_criteria"]),
    )


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    required = {
        "case_id", "task", "worker_no_outputs", "worker_with_outputs",
        "oracle_team_no_outputs", "oracle_team_with_outputs",
        "expected_standard_mismatch", "expected_structural_mismatch",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"{case.get('case_id', '<unknown>')} missing fields: {sorted(missing)}")
        if len(case["worker_no_outputs"]) != len(case["worker_with_outputs"]):
            raise ValueError(f"{case['case_id']} has mismatched worker rollout counts")
        if len(case["oracle_team_no_outputs"]) != len(case["oracle_team_with_outputs"]):
            raise ValueError(f"{case['case_id']} has mismatched oracle team rollout counts")
    return cases


def run(args: argparse.Namespace) -> Path:
    cases = _load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    config = load_config(str(args.config))
    output_dir = ensure_dir(args.output_dir)
    client = None
    if args.mode == "ollama":
        openai_config = config.get("openai", {})
        client = OpenAIClient(
            use_cache=not args.no_cache,
            use_api=bool(openai_config.get("use_api", True)),
            provider=openai_config.get("provider"),
            base_url=openai_config.get("base_url"),
            cache_dir=str(output_dir / "cache"),
        )
        if openai_config.get("provider") == "ollama" and not client.use_api:
            raise RuntimeError("Ollama is configured but API use is disabled")

    rows: list[dict[str, Any]] = []
    for case in cases:
        example = _example(case)
        worker_no = case["worker_no_outputs"]
        worker_with = case["worker_with_outputs"]
        if args.mode == "oracle":
            team_no = case["oracle_team_no_outputs"]
            team_with = case["oracle_team_with_outputs"]
        else:
            team_no = []
            team_with = []
            for rollout_id, (no_report, with_report) in enumerate(zip(worker_no, worker_with)):
                common = {"case_id": case["case_id"], "rollout_id": rollout_id}
                team_no.append(_call_team(client, config, case["task"], no_report, {"purpose": "synthetic_team_no", **common})["text"])
                team_with.append(_call_team(client, config, case["task"], with_report, {"purpose": "synthetic_team_with", **common})["text"])

        local_no_scores: list[float] = []
        local_with_scores: list[float] = []
        team_no_scores: list[float] = []
        team_with_scores: list[float] = []
        for rollout_id, (local_no, local_with, team_no_text, team_with_text) in enumerate(zip(worker_no, worker_with, team_no, team_with)):
            if args.mode == "oracle" and args.utility_scorer == "deterministic":
                from src.scoring.deterministic_scorers import score_response
                local_no_score = float(score_response(local_no, example).final_score)
                local_with_score = float(score_response(local_with, example).final_score)
                team_no_score = float(score_response(team_no_text, example).final_score)
                team_with_score = float(score_response(team_with_text, example).final_score)
            else:
                common = {"case_id": case["case_id"], "rollout_id": rollout_id}
                local_no_score, _ = _score_team_answer(client, config, example, local_no, {"purpose": "synthetic_local_no", **common}, args.utility_scorer)
                local_with_score, _ = _score_team_answer(client, config, example, local_with, {"purpose": "synthetic_local_with", **common}, args.utility_scorer)
                team_no_score, _ = _score_team_answer(client, config, example, team_no_text, {"purpose": "synthetic_team_score_no", **common}, args.utility_scorer)
                team_with_score, _ = _score_team_answer(client, config, example, team_with_text, {"purpose": "synthetic_team_score_with", **common}, args.utility_scorer)
            local_no_scores.append(local_no_score)
            local_with_scores.append(local_with_score)
            team_no_scores.append(team_no_score)
            team_with_scores.append(team_with_score)

        local_rollout = [with_score - no_score for no_score, with_score in zip(local_no_scores, local_with_scores)]
        team_rollout = [with_score - no_score for no_score, with_score in zip(team_no_scores, team_with_scores)]
        local_utility = _mean(local_rollout)
        team_utility = _mean(team_rollout)
        resource_cost_delta = float(case.get("resource_cost_with", 0.0)) - float(case.get("resource_cost_no", 0.0))
        resource_weight = float(case.get("resource_cost_weight", 0.5))
        adjusted_team_utility = team_utility - resource_weight * resource_cost_delta
        standard_mismatch = (local_utility > args.utility_epsilon) != (team_utility > args.utility_epsilon)
        structural_mismatch = (local_utility > args.utility_epsilon) != (adjusted_team_utility > args.utility_epsilon)
        rows.append(
            {
                **case,
                "mode": args.mode,
                "local_no_outputs": worker_no,
                "local_with_outputs": worker_with,
                "team_no_outputs": team_no,
                "team_with_outputs": team_with,
                "local_rollout_utilities": local_rollout,
                "team_rollout_utilities": team_rollout,
                "local_utility": local_utility,
                "team_utility": team_utility,
                "resource_cost_delta": resource_cost_delta,
                "resource_adjusted_team_utility": adjusted_team_utility,
                "standard_mismatch_detected": standard_mismatch,
                "structural_mismatch_detected": structural_mismatch,
            }
        )

    expected_structural = [row for row in rows if row["expected_structural_mismatch"]]
    expected_standard = [row for row in rows if row["expected_standard_mismatch"]]
    resource_only = [row for row in rows if not row["expected_standard_mismatch"] and row["expected_structural_mismatch"]]
    summary = {
        "mode": args.mode,
        "utility_scorer": args.utility_scorer,
        "n_cases": len(rows),
        "expected_structural_mismatch_cases": len(expected_structural),
        "expected_standard_answer_mismatch_cases": len(expected_standard),
        "expected_resource_only_cases": len(resource_only),
        "standard_detected_cases": sum(row["standard_mismatch_detected"] for row in rows),
        "structural_detected_cases": sum(row["structural_mismatch_detected"] for row in rows),
        "standard_recall_on_answer_cases": (
            sum(row["standard_mismatch_detected"] for row in expected_standard) / len(expected_standard)
            if expected_standard else None
        ),
        "structural_recall_on_all_cases": (
            sum(row["structural_mismatch_detected"] for row in expected_structural) / len(expected_structural)
            if expected_structural else None
        ),
        "resource_case_standard_detection": sum(row["standard_mismatch_detected"] for row in resource_only),
        "resource_case_adjusted_detection": sum(row["structural_mismatch_detected"] for row in resource_only),
        "cases": [
            {
                "case_id": row["case_id"],
                "case_type": row["case_type"],
                "expected_standard_mismatch": row["expected_standard_mismatch"],
                "expected_structural_mismatch": row["expected_structural_mismatch"],
                "local_utility": row["local_utility"],
                "team_utility": row["team_utility"],
                "resource_adjusted_team_utility": row["resource_adjusted_team_utility"],
                "standard_mismatch_detected": row["standard_mismatch_detected"],
                "structural_mismatch_detected": row["structural_mismatch_detected"],
            }
            for row in rows
        ],
    }
    write_jsonl(rows, output_dir / "synthetic_results.jsonl")
    write_json(summary, output_dir / "summary.json")
    print(json.dumps({key: summary[key] for key in ("n_cases", "standard_detected_cases", "structural_detected_cases", "output_dir") if key in summary} | {"output_dir": str(output_dir)}, ensure_ascii=False))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated synthetic local-team mismatch diagnostic.")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config_local.yaml")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["oracle", "ollama"], default="oracle", help="Oracle uses hand-written team outputs; ollama tests the actual Synthesizer.")
    parser.add_argument("--utility-scorer", choices=["deterministic", "llm", "hybrid"], default="deterministic")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--utility-epsilon", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
