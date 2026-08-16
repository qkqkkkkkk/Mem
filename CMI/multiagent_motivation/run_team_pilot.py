from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.openai_client import OpenAIClient
from src.benchmark.load_dataset import load_examples
from src.scoring.deterministic_scorers import score_response
from src.utils.io import ensure_dir, load_config, write_json, write_jsonl


def _bounded_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score or score in {float("inf"), float("-inf")}:
        return None
    return max(0.0, min(1.0, score))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _combine(deterministic: float, judged: float, deterministic_weight: float, judge_weight: float) -> float:
    total = deterministic_weight + judge_weight
    if total <= 0:
        return deterministic
    return (deterministic_weight * deterministic + judge_weight * judged) / total


def _synth_prompt(task: str, worker_report: str) -> str:
    return f"""You are the synthesizer in a two-agent question-answering team.

You are responsible for the final answer shown to the user. A worker investigated
the question and sent you a report. You do not have access to the worker's source
memory or any other hidden context. Use the report as evidence, but check whether
it is specific, internally coherent, and responsive to the question. If it is
uncertain or contradictory, do not invent details. Return only the final answer.

Question:
{task}

Worker report:
{worker_report}

Final answer:
"""


def _team_judge_prompt(task: str, gold_behavior: str, criteria: dict[str, Any], answer: str) -> str:
    return f"""You are an impartial task-answer evaluator.

Task:
{task}

Expected behavior:
{gold_behavior}

Scoring criteria:
{json.dumps(criteria, ensure_ascii=False)}

Team answer:
{answer}

Score the answer for task correctness and adherence to the criteria. Do not score
style unless the criteria require it. Return JSON only:
{{
  "score": 0.0,
  "confidence": 0.0,
  "explanation": "brief reason"
}}

Both score and confidence must be numbers in [0, 1].
"""


def _call_team(
    client: OpenAIClient,
    config: dict[str, Any],
    task: str,
    worker_report: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    openai_config = config.get("openai", {})
    result = client.complete(
        _synth_prompt(task, worker_report),
        model=str(openai_config.get("team_model", "llama3:8b")),
        temperature=float(openai_config.get("team_temperature", 0.3)),
        max_output_tokens=int(openai_config.get("team_max_output_tokens", 400)),
        metadata=metadata,
    )
    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("Synthesizer returned an empty answer")
    return {"text": text, "model": result.get("model"), "cached": result.get("cached", False)}


def _judge_team(
    client: OpenAIClient,
    config: dict[str, Any],
    task: str,
    gold_behavior: str,
    criteria: dict[str, Any],
    answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    openai_config = config.get("openai", {})
    result = client.complete(
        _team_judge_prompt(task, gold_behavior, criteria, answer),
        model=str(openai_config.get("team_judge_model", "qwen2.5:14b")),
        temperature=0.0,
        max_output_tokens=int(openai_config.get("team_judge_max_output_tokens", 300)),
        json_mode=True,
        metadata=metadata,
    )
    data = result.get("json") or {}
    score = _bounded_score(data.get("score"))
    confidence = _bounded_score(data.get("confidence"))
    if score is None:
        raise RuntimeError(f"Invalid team judge score: {data!r}")
    return {
        "score": score,
        "confidence": confidence if confidence is not None else 0.0,
        "explanation": str(data.get("explanation", "")),
        "model": result.get("model"),
        "cached": result.get("cached", False),
    }


def _score_team_answer(
    client: OpenAIClient,
    config: dict[str, Any],
    example: Any,
    answer: str,
    metadata: dict[str, Any],
    utility_scorer: str,
) -> tuple[float, dict[str, Any]]:
    deterministic = float(score_response(answer, example).final_score)
    if utility_scorer == "deterministic":
        return deterministic, {"deterministic_score": deterministic, "judge": None}
    judge = _judge_team(
        client,
        config,
        example.current_task.instruction,
        example.gold_behavior,
        example.scoring_criteria.model_dump() if hasattr(example.scoring_criteria, "model_dump") else example.scoring_criteria.dict(),
        answer,
        metadata,
    )
    if utility_scorer == "llm":
        return judge["score"], {"deterministic_score": deterministic, "judge": judge}
    scoring_config = config.get("scoring", {})
    combined = _combine(
        deterministic,
        judge["score"],
        float(scoring_config.get("deterministic_weight", 0.5)),
        float(scoring_config.get("judge_weight", 0.5)),
    )
    return combined, {"deterministic_score": deterministic, "judge": judge}


def run(args: argparse.Namespace) -> Path:
    config = load_config(str(args.config))
    random.seed(args.seed if args.seed is not None else int(config.get("seed", 42)))
    source_dir = Path(args.source_dir)
    source_path = source_dir / "memory_interventions.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Source intervention file not found: {source_path}")
    source_rows = [json.loads(line) for line in source_path.open(encoding="utf-8") if line.strip()]
    if args.max_interventions is not None:
        source_rows = source_rows[: args.max_interventions]

    examples = load_examples(str(args.dataset), max_examples=args.max_examples)
    examples_by_id = {example.example_id: example for example in examples}
    missing = sorted({row["example_id"] for row in source_rows} - set(examples_by_id))
    if missing:
        raise ValueError(f"Dataset does not contain source example IDs: {missing[:5]}")

    output_dir = ensure_dir(args.output_dir)
    openai_config = config.get("openai", {})
    client = OpenAIClient(
        use_cache=not args.no_cache,
        use_api=bool(openai_config.get("use_api", True)),
        provider=openai_config.get("provider"),
        base_url=openai_config.get("base_url"),
        cache_dir=str(output_dir / "cache"),
    )
    if openai_config.get("provider") == "ollama" and not client.use_api:
        raise RuntimeError("Ollama is configured but API use is disabled.")
    team_model = str(openai_config.get("team_model", "llama3:8b"))
    team_judge_model = str(openai_config.get("team_judge_model", "qwen2.5:14b"))
    if args.require_independent_roles and (team_model == team_judge_model or team_model == args.worker_model):
        raise ValueError("team_model, team_judge_model, and worker_model must be distinct when independent roles are required")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source in source_rows:
        example = examples_by_id[source["example_id"]]
        no_reports = list(source.get("no_memory_outputs", []))
        with_reports = list(source.get("with_memory_outputs", []))
        if len(no_reports) != len(with_reports) or not no_reports:
            failures.append({"example_id": source["example_id"], "memory_id": source.get("memory_id", ""), "error": "with/no worker rollout counts differ or are empty"})
            continue
        if args.rollouts is not None:
            no_reports = no_reports[: args.rollouts]
            with_reports = with_reports[: args.rollouts]
        team_no: list[str] = []
        team_with: list[str] = []
        no_scores: list[float] = []
        with_scores: list[float] = []
        no_score_details: list[dict[str, Any]] = []
        with_score_details: list[dict[str, Any]] = []
        try:
            for rollout_id, (no_report, with_report) in enumerate(zip(no_reports, with_reports)):
                common = {
                    "example_id": source["example_id"],
                    "memory_id": source.get("memory_id", ""),
                    "rollout_id": rollout_id,
                }
                no_result = _call_team(client, config, example.current_task.instruction, no_report, {"purpose": "multiagent_team_no_memory", "condition": "no_memory", **common})
                with_result = _call_team(client, config, example.current_task.instruction, with_report, {"purpose": "multiagent_team_with_memory", "condition": "with_memory", **common})
                team_no.append(no_result["text"])
                team_with.append(with_result["text"])
                no_score, no_details = _score_team_answer(client, config, example, no_result["text"], {"purpose": "multiagent_team_judge", "condition": "no_memory", **common}, args.utility_scorer)
                with_score, with_details = _score_team_answer(client, config, example, with_result["text"], {"purpose": "multiagent_team_judge", "condition": "with_memory", **common}, args.utility_scorer)
                no_scores.append(no_score)
                with_scores.append(with_score)
                no_score_details.append(no_details)
                with_score_details.append(with_details)
        except Exception as exc:  # noqa: BLE001
            failures.append({"example_id": source["example_id"], "memory_id": source.get("memory_id", ""), "error": str(exc)})
            continue

        team_utilities = [with_score - no_score for no_score, with_score in zip(no_scores, with_scores)]
        revised = dict(source)
        revised.update(
            {
                "local_utility": float(source.get("utility", 0.0)),
                "local_rollout_utilities": source.get("rollout_utilities", []),
                "team_no_memory_outputs": team_no,
                "team_with_memory_outputs": team_with,
                "team_no_scores": no_scores,
                "team_with_scores": with_scores,
                "team_no_score_details": no_score_details,
                "team_with_score_details": with_score_details,
                "team_rollout_utilities": team_utilities,
                "team_utility": _mean(team_utilities),
                "team_utility_sd": statistics.pstdev(team_utilities) if len(team_utilities) > 1 else 0.0,
                "local_team_sign_mismatch": (float(source.get("utility", 0.0)) > args.utility_epsilon) != (_mean(team_utilities) > args.utility_epsilon),
                "team_model": team_model,
                "team_judge_model": team_judge_model if args.utility_scorer != "deterministic" else None,
                "team_utility_scorer": args.utility_scorer,
            }
        )
        rows.append(revised)

    summary = {
        "n_source_interventions": len(source_rows),
        "n_completed_interventions": len(rows),
        "n_completed_examples": len({row["example_id"] for row in rows}),
        "run": {
            "source_dir": str(source_dir),
            "dataset": str(args.dataset),
            "config": str(args.config),
            "team_model": team_model,
            "team_judge_model": team_judge_model if args.utility_scorer != "deterministic" else None,
            "utility_scorer": args.utility_scorer,
            "rollouts": args.rollouts,
            "worker_model": args.worker_model,
            "failures": failures,
        },
    }
    write_jsonl(rows, output_dir / "team_interventions.jsonl")
    write_json(summary, output_dir / "run_summary.json")
    print(json.dumps({"completed": len(rows), "failures": len(failures), "output_dir": str(output_dir)}, ensure_ascii=False))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a worker-to-synthesizer team layer to an existing motivation run.")
    parser.add_argument("--source-dir", required=True, type=Path, help="Existing motivation result directory with memory_interventions.jsonl")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_local.yaml")), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-interventions", type=int, default=None, help="Use a small prefix for a smoke test")
    parser.add_argument("--rollouts", type=int, default=None, help="Use this many existing worker rollouts; default uses all")
    parser.add_argument("--utility-scorer", choices=["deterministic", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--worker-model", default="qwen2.5:7b")
    parser.add_argument("--utility-epsilon", type=float, default=0.0)
    parser.add_argument("--require-independent-roles", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
