from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else None


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    clusters = sorted({str(row["example_id"]) for row in rows})
    estimate = statistic(rows)
    if estimate is None or len(clusters) < 2:
        return {
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "n": len(rows),
            "n_clusters": len(clusters),
            "method": "question_cluster_bootstrap",
        }
    rng = random.Random(seed)
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[str(row["example_id"])].append(row)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sampled = []
        for cluster in [rng.choice(clusters) for _ in clusters]:
            sampled.extend(by_cluster[cluster])
        value = statistic(sampled)
        if value is not None and math.isfinite(value):
            samples.append(value)
    samples.sort()
    lower_index = max(0, int(0.025 * len(samples)) - 1)
    upper_index = min(len(samples) - 1, int(0.975 * len(samples)))
    return {
        "estimate": estimate,
        "lower": samples[lower_index] if samples else None,
        "upper": samples[upper_index] if samples else None,
        "n": len(rows),
        "n_clusters": len(clusters),
        "method": "question_cluster_bootstrap",
    }


def _rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    return _mean([1.0 if predicate(row) else 0.0 for row in rows])


def _rate_statistic(predicate: Callable[[dict[str, Any]], bool]) -> Callable[[list[dict[str, Any]]], float | None]:
    return lambda rows: _rate(rows, predicate)


def _correlation_statistic(left: str, right: str) -> Callable[[list[dict[str, Any]]], float | None]:
    return lambda rows: _pearson([float(row[left]) for row in rows], [float(row[right]) for row in rows])


def _mismatch_direction(row: dict[str, Any], epsilon: float) -> str | None:
    local_positive = float(row["local_utility"]) > epsilon
    team_positive = float(row["team_utility"]) > epsilon
    if local_positive == team_positive:
        return None
    if local_positive and not team_positive:
        return "local_positive_team_nonpositive"
    return "local_nonpositive_team_positive"


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "mean": _mean(values),
        "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "n_positive": sum(value > 0 for value in values),
        "n_nonpositive": sum(value <= 0 for value in values),
    }


def analyze(input_dir: Path, output_dir: Path, epsilon: float, seed: int, n_bootstrap: int) -> dict[str, Any]:
    path = input_dir / "team_interventions.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    for row in rows:
        direction = _mismatch_direction(row, epsilon)
        row["local_team_sign_mismatch"] = direction is not None
        row["mismatch_direction"] = direction or "same_sign"
        completed_rows.append(row)
        if direction:
            mismatch_rows.append(row)

    mismatch_rate = _cluster_bootstrap(
        completed_rows,
        _rate_statistic(lambda row: _mismatch_direction(row, epsilon) is not None),
        seed,
        n_bootstrap,
    )
    direction_rates = {
        direction: _cluster_bootstrap(
            completed_rows,
            _rate_statistic(lambda row, direction=direction: _mismatch_direction(row, epsilon) == direction),
            seed + index + 1,
            n_bootstrap,
        )
        for index, direction in enumerate(("local_positive_team_nonpositive", "local_nonpositive_team_positive"))
    }
    contingency = Counter()
    for row in completed_rows:
        local_bucket = "positive" if float(row["local_utility"]) > epsilon else "nonpositive"
        team_bucket = "positive" if float(row["team_utility"]) > epsilon else "nonpositive"
        contingency[(local_bucket, team_bucket)] += 1

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed_rows:
        by_family[str(row.get("task_family", "unknown"))].append(row)
    family_summary = {}
    for family, family_rows in sorted(by_family.items()):
        family_summary[family] = {
            "n": len(family_rows),
            "n_questions": len({row["example_id"] for row in family_rows}),
            "mismatch_rate": _cluster_bootstrap(
                family_rows,
                _rate_statistic(lambda row: _mismatch_direction(row, epsilon) is not None),
                seed + 100 + len(family_summary),
                n_bootstrap,
            ),
            "local_utility": _distribution(family_rows, "local_utility"),
            "team_utility": _distribution(family_rows, "team_utility"),
        }

    summary = {
        "n_interventions": len(completed_rows),
        "n_examples": len({row["example_id"] for row in completed_rows}),
        "epsilon": epsilon,
        "local_utility": _distribution(completed_rows, "local_utility"),
        "team_utility": _distribution(completed_rows, "team_utility"),
        "mismatch_rate": mismatch_rate,
        "mismatch_directions": direction_rates,
        "contingency_table": {
            "local_positive_team_positive": contingency[("positive", "positive")],
            "local_positive_team_nonpositive": contingency[("positive", "nonpositive")],
            "local_nonpositive_team_positive": contingency[("nonpositive", "positive")],
            "local_nonpositive_team_nonpositive": contingency[("nonpositive", "nonpositive")],
        },
        "correlations": {
            "pearson_local_team_utility": _cluster_bootstrap(
                completed_rows,
                _correlation_statistic("local_utility", "team_utility"),
                seed + 20,
                n_bootstrap,
            )
        },
        "task_family": family_summary,
        "case_counts": {
            "all_mismatches": len(mismatch_rows),
            "local_positive_team_nonpositive": sum(
                row["mismatch_direction"] == "local_positive_team_nonpositive" for row in mismatch_rows
            ),
            "local_nonpositive_team_positive": sum(
                row["mismatch_direction"] == "local_nonpositive_team_positive" for row in mismatch_rows
            ),
        },
    }

    with (output_dir / "mismatch_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in mismatch_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary_csv(summary, output_dir / "summary.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(completed_rows), "mismatches": len(mismatch_rows), "output_dir": str(output_dir)}, ensure_ascii=False))
    return summary


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    records = []

    def add(metric: str, value: dict[str, Any], family: str = ""):
        records.append({
            "metric": metric,
            "estimate": value.get("estimate"),
            "lower": value.get("lower"),
            "upper": value.get("upper"),
            "n": value.get("n"),
            "n_clusters": value.get("n_clusters"),
            "task_family": family,
        })

    add("mismatch_rate", summary["mismatch_rate"])
    for name, value in summary["mismatch_directions"].items():
        add(name, value)
    add("pearson_local_team_utility", summary["correlations"]["pearson_local_team_utility"])
    for family, family_summary in summary["task_family"].items():
        add("mismatch_rate", family_summary["mismatch_rate"], family)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "estimate", "lower", "upper", "n", "n_clusters", "task_family"])
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local-versus-team utility mismatch.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--utility-epsilon", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input_dir, args.output_dir or args.input_dir, args.utility_epsilon, args.seed, args.bootstrap)
