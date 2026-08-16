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

from multiagent_motivation.run_team_pilot import _score_team_answer
from src.api.openai_client import OpenAIClient
from src.benchmark.load_dataset import load_examples
from src.utils.io import ensure_dir, load_config, write_json, write_jsonl


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stored_team_utility(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    no_scores = [float(value) for value in row.get("team_no_scores", [])]
    with_scores = [float(value) for value in row.get("team_with_scores", [])]
    if not no_scores or len(no_scores) != len(with_scores):
        raise ValueError("Missing or mismatched stored team scores")
    return no_scores, with_scores


def run(args: argparse.Namespace) -> Path:
    config = load_config(str(args.config))
    source_dir = Path(args.input_dir)
    source_path = source_dir / "team_interventions.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Input team result not found: {source_path}")
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
    judge_model = str(openai_config.get("team_judge_model", "qwen2.5:14b"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source in source_rows:
        example = examples_by_id[source["example_id"]]
        no_outputs = list(source.get("no_memory_outputs", []))
        with_outputs = list(source.get("with_memory_outputs", []))
        if args.rollouts is not None:
            no_outputs = no_outputs[: args.rollouts]
            with_outputs = with_outputs[: args.rollouts]
        if len(no_outputs) != len(with_outputs) or not no_outputs:
            failures.append({"example_id": source["example_id"], "memory_id": source.get("memory_id", ""), "error": "with/no worker rollout counts differ or are empty"})
            continue
        try:
            stored_no, stored_with = _stored_team_utility(source)
            if args.rollouts is not None:
                stored_no = stored_no[: args.rollouts]
                stored_with = stored_with[: args.rollouts]
            if len(stored_no) != len(no_outputs):
                raise ValueError("Stored team score count does not match requested rollouts")

            local_no_scores: list[float] = []
            local_with_scores: list[float] = []
            local_no_details: list[dict[str, Any]] = []
            local_with_details: list[dict[str, Any]] = []
            for rollout_id, (no_output, with_output) in enumerate(zip(no_outputs, with_outputs)):
                common = {
                    "example_id": source["example_id"],
                    "memory_id": source.get("memory_id", ""),
                    "rollout_id": rollout_id,
                }
                no_score, no_detail = _score_team_answer(
                    client,
                    config,
                    example,
                    no_output,
                    {"purpose": "multiagent_same_judge_local_score", "condition": "local_no_memory", **common},
                    args.utility_scorer,
                )
                with_score, with_detail = _score_team_answer(
                    client,
                    config,
                    example,
                    with_output,
                    {"purpose": "multiagent_same_judge_local_score", "condition": "local_with_memory", **common},
                    args.utility_scorer,
                )
                local_no_scores.append(no_score)
                local_with_scores.append(with_score)
                local_no_details.append(no_detail)
                local_with_details.append(with_detail)

            local_rollout = [with_score - no_score for no_score, with_score in zip(local_no_scores, local_with_scores)]
            team_rollout = [with_score - no_score for no_score, with_score in zip(stored_no, stored_with)]
            revised = dict(source)
            revised.update(
                {
                    "original_local_utility": source.get("local_utility"),
                    "original_team_utility": source.get("team_utility"),
                    "original_local_team_sign_mismatch": source.get("local_team_sign_mismatch"),
                    "same_judge_model": judge_model,
                    "same_judge_utility_scorer": args.utility_scorer,
                    "same_judge_local_no_scores": local_no_scores,
                    "same_judge_local_with_scores": local_with_scores,
                    "same_judge_local_no_details": local_no_details,
                    "same_judge_local_with_details": local_with_details,
                    "same_judge_local_rollout_utilities": local_rollout,
                    "same_judge_team_rollout_utilities": team_rollout,
                    "local_utility": _mean(local_rollout),
                    "team_utility": _mean(team_rollout),
                    "team_utility_sd": statistics.pstdev(team_rollout) if len(team_rollout) > 1 else 0.0,
                    "local_team_sign_mismatch": (_mean(local_rollout) > args.utility_epsilon) != (_mean(team_rollout) > args.utility_epsilon),
                }
            )
            rows.append(revised)
        except Exception as exc:  # noqa: BLE001
            failures.append({"example_id": source["example_id"], "memory_id": source.get("memory_id", ""), "error": str(exc)})

    summary = {
        "n_source_interventions": len(source_rows),
        "n_completed_interventions": len(rows),
        "n_completed_examples": len({row["example_id"] for row in rows}),
        "run": {
            "input_dir": str(source_dir),
            "dataset": str(args.dataset),
            "config": str(args.config),
            "same_judge_model": judge_model,
            "utility_scorer": args.utility_scorer,
            "rollouts": args.rollouts,
            "rejudged_conditions": ["local_no_memory", "local_with_memory"],
            "team_scores_reused_from_input": True,
            "failures": failures,
        },
    }
    write_jsonl(rows, output_dir / "team_interventions.jsonl")
    write_json(summary, output_dir / "run_summary.json")
    print(json.dumps({"completed": len(rows), "failures": len(failures), "output_dir": str(output_dir)}, ensure_ascii=False))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore local worker outputs with the same judge used for team utility.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Existing multiagent result directory")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-interventions", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--utility-scorer", choices=["deterministic", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--utility-epsilon", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
