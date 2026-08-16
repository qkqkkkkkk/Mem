from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multiagent_motivation.analyze_team_results import analyze


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _real_row(row: dict[str, Any]) -> dict[str, Any]:
    # Keep real rows untouched except for an explicit provenance marker.
    normalized = dict(row)
    normalized["origin"] = "real"
    normalized["case_type"] = "observed_benchmark_intervention"
    normalized["expected_standard_mismatch"] = None
    normalized["expected_structural_mismatch"] = None
    normalized["resource_cost_delta"] = 0.0
    normalized["resource_cost_weight"] = 0.0
    normalized["resource_adjusted_team_utility"] = float(normalized["team_utility"])
    return normalized


def _synthetic_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["origin"] = "synthetic"
    normalized["example_id"] = normalized.get("case_id", "synthetic")
    normalized["memory_id"] = normalized.get("case_id", "synthetic")
    normalized["label"] = "synthetic"
    normalized["memory_content"] = normalized.get("notes", "")
    normalized["relevance_score"] = None
    normalized["resource_adjusted_team_utility"] = float(normalized["resource_adjusted_team_utility"])
    return normalized


def combine(real_dir: Path, synthetic_dir: Path, output_dir: Path) -> dict[str, Any]:
    real_path = real_dir / "team_interventions.jsonl"
    synthetic_path = synthetic_dir / "synthetic_results.jsonl"
    if not real_path.exists():
        raise FileNotFoundError(f"Missing real results: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Missing synthetic results: {synthetic_path}")
    real_rows = [_real_row(row) for row in _load_jsonl(real_path)]
    synthetic_rows = [_synthetic_row(row) for row in _load_jsonl(synthetic_path)]
    rows = real_rows + synthetic_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "team_interventions.jsonl"
    with combined_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    answer_dir = output_dir / "analysis_answer_only"
    structural_dir = output_dir / "analysis_structural"
    answer_summary = analyze(output_dir, answer_dir, epsilon=0.0, seed=42, n_bootstrap=2000)

    # Replace the team utility with the resource-adjusted value only for the
    # structural diagnostic. Preserve the answer-only rows and summary above.
    structural_rows = []
    for row in rows:
        adjusted = dict(row)
        adjusted["team_utility"] = float(row["resource_adjusted_team_utility"])
        structural_rows.append(adjusted)
    structural_path = structural_dir.parent / "team_interventions_structural.jsonl"
    with structural_path.open("w", encoding="utf-8") as handle:
        for row in structural_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    structural_summary = analyze(
        output_dir,
        structural_dir,
        epsilon=0.0,
        seed=42,
        n_bootstrap=2000,
        input_filename="team_interventions_structural.jsonl",
    )
    (structural_dir / "summary.json").write_text(
        json.dumps(structural_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (structural_dir / "README.txt").write_text(
        "This summary uses resource_adjusted_team_utility for synthetic cases.\n",
        encoding="utf-8",
    )
    summary = {
        "n_real": len(real_rows),
        "n_synthetic": len(synthetic_rows),
        "n_total": len(rows),
        "real_source": str(real_dir),
        "synthetic_source": str(synthetic_dir),
        "answer_only_summary": answer_summary,
        "structural_summary": structural_summary,
    }
    (output_dir / "combined_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine real team interventions with isolated synthetic diagnostic cases.")
    parser.add_argument("--real-dir", required=True, type=Path)
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = combine(args.real_dir, args.synthetic_dir, args.output_dir)
    print(json.dumps({"n_real": result["n_real"], "n_synthetic": result["n_synthetic"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
