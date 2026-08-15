from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

# Permit `python motivation_experiment/analyze_results.py` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motivation_experiment.run_pilot import _plot, summarize, write_summary_csv
from src.utils.io import ensure_dir, write_json


def analyze(output_dir: Path, seed: int = 42) -> dict:
    rows_path = output_dir / "memory_interventions.jsonl"
    old_summary_path = output_dir / "summary.json"
    rows = [json.loads(line) for line in rows_path.open(encoding="utf-8") if line.strip()]
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8")) if old_summary_path.exists() else {}
    thresholds = old_summary.get("thresholds", {})
    summary = summarize(
        rows,
        float(thresholds.get("top_relevance_quantile", 0.8)),
        float(thresholds.get("high_B_quantile", 0.8)),
        float(thresholds.get("utility_epsilon", 0.0)),
        seed,
    )
    for key in ("run", "manual_overrides"):
        if key in old_summary:
            summary[key] = old_summary[key]
    summary["analysis"] = {
        "primary_correlation_ci": "question_cluster_bootstrap",
        "prediction_cv": "leave_one_question_out",
        "seed": seed,
    }
    write_summary_csv(summary, output_dir / "summary.csv")
    summary["figures"] = _plot(rows, summary, ensure_dir(output_dir / "figures"))
    write_json(summary, old_summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reanalyze an existing motivation pilot without calling an LLM.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = analyze(args.output_dir, args.seed)
    print(json.dumps({
        "n_interventions": result.get("n_interventions"),
        "n_examples": result.get("n_examples"),
        "summary": str(args.output_dir / "summary.json"),
    }, ensure_ascii=False))
