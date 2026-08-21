"""Test whether retrieval/reliance proxies can substitute for causal utility.

The multi-agent pilot already evaluates every intervention twice: the team sees
the worker report produced with a candidate memory and the team sees the
no-memory worker report.  This module reuses those paired outcomes and asks a
more direct question than a raw correlation:

    If one candidate memory must be selected for each question, do R, B, or
    R+B select the same useful memory as an oracle that has access to U?

The script is intentionally analysis-only.  It does not make additional LLM
calls and therefore can be run repeatedly while changing the selector or
bootstrap settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_R_FIELD = "hybrid_relevance"
DEFAULT_B_FIELD = "behavioral_reliance"
DEFAULT_U_FIELD = "team_utility"
CONDITIONS = ("none", "R", "B", "R+B", "U_oracle", "random")


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if _finite(value) is not None]
    return statistics.fmean(values) if values else None


def _as_scores(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [parsed for item in value if (parsed := _finite(item)) is not None]


def _row_team_baseline(row: dict[str, Any]) -> float | None:
    values = _as_scores(row.get("team_no_scores"))
    if values:
        return _mean(values)
    utility = _finite(row.get("team_utility"))
    with_score = _mean(_as_scores(row.get("team_with_scores")))
    if utility is not None and with_score is not None:
        return with_score - utility
    return None


def _row_team_with(row: dict[str, Any]) -> float | None:
    values = _as_scores(row.get("team_with_scores"))
    if values:
        return _mean(values)
    baseline = _row_team_baseline(row)
    utility = _finite(row.get("team_utility"))
    return baseline + utility if baseline is not None and utility is not None else None


def _row_team_utility(row: dict[str, Any]) -> float | None:
    explicit = _finite(row.get("team_utility"))
    if explicit is not None:
        return explicit
    baseline = _row_team_baseline(row)
    with_score = _row_team_with(row)
    return with_score - baseline if baseline is not None and with_score is not None else None


def _row_team_behavior(row: dict[str, Any]) -> float | None:
    """Return a team-level B when raw team answers are available.

    Existing runs do not always store a team-level reliance field. This derived
    metric measures how much the synthesizer answer changed relative to the
    paired no-memory answer whenever the lexical proxy is available.
    """

    explicit = _finite(row.get("team_behavioral_reliance"))
    if explicit is not None:
        return explicit
    with_outputs = row.get("team_with_memory_outputs")
    no_outputs = row.get("team_no_memory_outputs")
    if not isinstance(with_outputs, list) or not isinstance(no_outputs, list):
        return None
    pairs = []
    for with_text, no_text in zip(with_outputs, no_outputs):
        with_tokens = set(str(with_text or "").lower().split())
        no_tokens = set(str(no_text or "").lower().split())
        union = with_tokens | no_tokens
        jaccard_distance = 0.0 if not union else 1.0 - len(with_tokens & no_tokens) / len(union)
        # A lightweight sequence proxy avoids importing the generation runner.
        with_chars, no_chars = str(with_text or ""), str(no_text or "")
        longest = _longest_common_subsequence(with_chars, no_chars)
        sequence_distance = 1.0 - (2.0 * longest / max(1, len(with_chars) + len(no_chars)))
        pairs.append(max(0.0, min(1.0, 0.5 * jaccard_distance + 0.5 * sequence_distance)))
    return _mean(pairs)


def _metric(row: dict[str, Any], field: str) -> float | None:
    """Resolve ordinary JSON fields and derived team-level measurements."""

    if field == "team_utility":
        return _row_team_utility(row)
    if field == "team_behavioral_reliance":
        return _row_team_behavior(row)
    return _finite(row.get(field))


def _longest_common_subsequence(left: str, right: str) -> int:
    if not left or not right:
        return 0
    # Answers are short; a row-wise DP is clearer and safer than a dependency
    # on a particular difflib implementation.
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lower, upper = min(values), max(values)
    if upper - lower <= 1e-12:
        return [0.5] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


def _select_index(
    rows: list[dict[str, Any]],
    condition: str,
    r_field: str,
    b_field: str,
    u_field: str,
    rng: random.Random,
) -> int | None:
    if not rows or condition == "none":
        return None
    if condition == "random":
        return rng.randrange(len(rows))

    r_values = [_metric(row, r_field) for row in rows]
    b_values = [_metric(row, b_field) for row in rows]
    u_values = [_metric(row, u_field) for row in rows]
    if condition == "R":
        valid = [index for index, value in enumerate(r_values) if value is not None]
        return max(valid, key=lambda index: (r_values[index], str(rows[index].get("memory_id", "")))) if valid else None
    if condition == "B":
        valid = [index for index, value in enumerate(b_values) if value is not None]
        return max(valid, key=lambda index: (b_values[index], str(rows[index].get("memory_id", "")))) if valid else None
    if condition == "U_oracle":
        valid = [index for index, value in enumerate(u_values) if value is not None]
        return max(valid, key=lambda index: (u_values[index], str(rows[index].get("memory_id", "")))) if valid else None
    if condition == "R+B":
        valid = [index for index, (r_value, b_value) in enumerate(zip(r_values, b_values)) if r_value is not None and b_value is not None]
        if not valid:
            return None
        normalized_r = _normalize([r_values[index] for index in valid])
        normalized_b = _normalize([b_values[index] for index in valid])
        combined_scores = {
            index: r_score + b_score
            for index, r_score, b_score in zip(valid, normalized_r, normalized_b)
        }
        return max(valid, key=lambda index: (combined_scores[index], str(rows[index].get("memory_id", ""))))
    raise ValueError(f"Unknown selection condition: {condition}")


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end) / 2.0 + 1.0
        for index in order[cursor : end + 1]:
            ranks[index] = rank
        cursor = end + 1
    return ranks


def _bootstrap_mean(values: list[float], seed: int, n_bootstrap: int) -> dict[str, Any]:
    values = [value for value in values if _finite(value) is not None]
    estimate = _mean(values)
    if not values or len(values) < 2 or n_bootstrap <= 0:
        return {"estimate": estimate, "lower": None, "upper": None, "n": len(values), "method": "question_bootstrap"}
    rng = random.Random(seed)
    samples = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_bootstrap)]
    samples.sort()
    lower = samples[max(0, int(0.025 * len(samples)))]
    upper = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return {"estimate": estimate, "lower": lower, "upper": upper, "n": len(values), "method": "question_bootstrap"}


def _cluster_bootstrap_metric(
    records: list[dict[str, Any]],
    metric: Any,
    seed: int,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Bootstrap an OOF metric by question, preserving intervention clusters."""

    if not records:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "n_clusters": 0, "method": "question_cluster_bootstrap"}
    clusters = sorted({str(record["example_id"]) for record in records})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["example_id"])].append(record)

    def evaluate(sample: list[dict[str, Any]]) -> float | None:
        try:
            value = float(metric([record["target"] for record in sample], [record["probability"] for record in sample]))
        except Exception:  # noqa: BLE001
            return None
        return value if math.isfinite(value) else None

    estimate = evaluate(records)
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "n": len(records),
        "n_clusters": len(clusters),
        "method": "question_cluster_bootstrap",
    }
    if estimate is None or len(clusters) < 2 or n_bootstrap <= 0:
        return result
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled: list[dict[str, Any]] = []
        for _ in clusters:
            sampled.extend(grouped[rng.choice(clusters)])
        value = evaluate(sampled)
        if value is not None:
            values.append(value)
    if values:
        values.sort()
        result["lower"] = values[max(0, int(0.025 * len(values)))]
        result["upper"] = values[min(len(values) - 1, int(0.975 * len(values)))]
    return result


def _utility_sign_prediction(
    rows: list[dict[str, Any]],
    r_field: str,
    b_field: str,
    u_field: str,
    seed: int,
    utility_epsilon: float = 0.0,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Predict positive versus negative U with leave-one-question-out models."""

    labeled = []
    for row in rows:
        r_value, b_value, u_value = _metric(row, r_field), _metric(row, b_field), _metric(row, u_field)
        if r_value is not None and b_value is not None and u_value is not None and abs(u_value) > utility_epsilon:
            labeled.append({"example_id": str(row["example_id"]), "r": r_value, "b": b_value, "u": u_value})
    base = {
        "task": "positive_vs_negative_utility_excluding_neutral",
        "neutral_definition": f"abs(U) <= {utility_epsilon}",
        "n": len(labeled),
        "n_excluded_neutral": len(rows) - len(labeled),
        "n_clusters": len({row["example_id"] for row in labeled}),
    }
    targets = [1 if row["u"] > utility_epsilon else 0 for row in labeled]
    if len(labeled) < 4 or len(set(targets)) < 2 or base["n_clusters"] < 2:
        return {**base, "models": {}, "reason": "insufficient non-neutral rows, classes, or question clusters"}
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        return {**base, "models": {}, "reason": f"scikit-learn unavailable: {exc}"}

    y = np.asarray(targets, dtype=int)
    groups = np.asarray([row["example_id"] for row in labeled])
    feature_sets = {"R": ["r"], "B": ["b"], "R_plus_B": ["r", "b"]}
    models: dict[str, Any] = {}
    for model_index, (name, feature_names) in enumerate(feature_sets.items()):
        x = np.asarray([[row[key] for key in feature_names] for row in labeled], dtype=float)
        probabilities = np.full(len(labeled), np.nan, dtype=float)
        skipped_folds = 0
        for train_indices, test_indices in LeaveOneGroupOut().split(x, y, groups):
            if len(set(y[train_indices].tolist())) < 2:
                skipped_folds += 1
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed),
            )
            model.fit(x[train_indices], y[train_indices])
            probabilities[test_indices] = model.predict_proba(x[test_indices])[:, 1]
        records = [
            {"example_id": labeled[index]["example_id"], "target": int(y[index]), "probability": float(probabilities[index])}
            for index in range(len(labeled))
            if not np.isnan(probabilities[index])
        ]
        actual = [record["target"] for record in records]
        predicted = [1 if record["probability"] >= 0.5 else 0 for record in records]
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
        models[name] = {
            "features": feature_names,
            "cv": "leave_one_question_out",
            "n_oof": len(records),
            "skipped_folds": skipped_folds,
            "roc_auc": _cluster_bootstrap_metric(records, roc_auc_score, seed + 100 + model_index, n_bootstrap),
            "average_precision": _cluster_bootstrap_metric(records, average_precision_score, seed + 200 + model_index, n_bootstrap),
            "precision_at_0_5": float(precision),
            "recall_at_0_5": float(recall),
            "f1_at_0_5": float(f1),
        }
    return {
        **base,
        "positive_rate": statistics.fmean(targets),
        "average_precision_baseline": statistics.fmean(targets),
        "models": models,
    }


def _load_rows(input_dir: Path, input_filename: str) -> list[dict[str, Any]]:
    path = input_dir / input_filename
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(
    input_dir: Path,
    output_dir: Path,
    r_field: str = DEFAULT_R_FIELD,
    b_field: str = DEFAULT_B_FIELD,
    u_field: str = DEFAULT_U_FIELD,
    seed: int = 42,
    n_bootstrap: int = 2000,
    input_filename: str = "team_interventions.jsonl",
) -> dict[str, Any]:
    rows = _load_rows(input_dir, input_filename)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("example_id"):
            grouped[str(row["example_id"])].append(row)
    if not grouped:
        raise ValueError("No rows with example_id were found")

    rng = random.Random(seed)
    selection_rows: list[dict[str, Any]] = []
    by_example_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for example_id, candidates in sorted(grouped.items()):
        baseline = _row_team_baseline(candidates[0])
        task_family = str(candidates[0].get("task_family", "unknown"))
        for condition in CONDITIONS:
            selected_index = _select_index(candidates, condition, r_field, b_field, u_field, rng)
            selected = candidates[selected_index] if selected_index is not None else None
            selected_score = _row_team_with(selected) if selected is not None else baseline
            selected_utility = (
                selected_score - baseline
                if selected_score is not None and baseline is not None
                else (_row_team_utility(selected) if selected is not None else 0.0)
            )
            record = {
                "example_id": example_id,
                "task_family": task_family,
                "condition": condition,
                "memory_id": selected.get("memory_id") if selected else None,
                "label": selected.get("label") if selected else None,
                "r": _metric(selected, r_field) if selected else None,
                "b": _metric(selected, b_field) if selected else None,
                "u": _metric(selected, u_field) if selected else None,
                "local_utility": _finite(selected.get("local_utility")) if selected else None,
                "baseline_team_score": baseline,
                "team_score": selected_score,
                "team_utility": selected_utility,
                "team_behavioral_reliance": _row_team_behavior(selected) if selected else 0.0,
                "selected_rank_candidates": len(candidates),
            }
            selection_rows.append(record)
            by_example_condition[(example_id, condition)] = record

    for record in selection_rows:
        oracle = by_example_condition[(record["example_id"], "U_oracle")]
        record["matches_u_oracle"] = record["memory_id"] == oracle.get("memory_id")
        record["score_gap_vs_u_oracle"] = (
            record["team_score"] - oracle["team_score"]
            if record.get("team_score") is not None and oracle.get("team_score") is not None
            else None
        )
        record["utility_gap_vs_u_oracle"] = (
            record["team_utility"] - oracle["team_utility"]
            if record.get("team_utility") is not None and oracle.get("team_utility") is not None
            else None
        )

    summary_conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in selection_rows if row["condition"] == condition]
        gaps = [row["score_gap_vs_u_oracle"] for row in condition_rows if row["score_gap_vs_u_oracle"] is not None]
        utility_gaps = [row["utility_gap_vs_u_oracle"] for row in condition_rows if row["utility_gap_vs_u_oracle"] is not None]
        scores = [row["team_score"] for row in condition_rows if row["team_score"] is not None]
        utilities = [row["team_utility"] for row in condition_rows if row["team_utility"] is not None]
        summary_conditions[condition] = {
            "n_examples": len(condition_rows),
            "mean_team_score": _mean(scores),
            "mean_team_utility": _mean(utilities),
            "pass_rate": _mean([1.0 if score >= 0.7 else 0.0 for score in scores]),
            "mean_r": _mean(row["r"] for row in condition_rows),
            "mean_b": _mean(row["b"] for row in condition_rows),
            "mean_u": _mean(row["u"] for row in condition_rows),
            "selection_match_rate_to_u_oracle": _mean([1.0 if row["matches_u_oracle"] else 0.0 for row in condition_rows]),
            "score_gap_vs_u_oracle": _bootstrap_mean(gaps, seed + 100 + len(summary_conditions), n_bootstrap),
            "utility_gap_vs_u_oracle": _bootstrap_mean(utility_gaps, seed + 200 + len(summary_conditions), n_bootstrap),
        }

    intervention_metrics = []
    for row in rows:
        r_value = _metric(row, r_field)
        b_value = _metric(row, b_field)
        u_value = _metric(row, u_field)
        team_u = _row_team_utility(row)
        if r_value is not None and b_value is not None and u_value is not None and team_u is not None:
            intervention_metrics.append(
                {
                    "example_id": str(row["example_id"]),
                    "task_family": str(row.get("task_family", "unknown")),
                    "memory_id": row.get("memory_id"),
                    "label": row.get("label", "unknown"),
                    "r": r_value,
                    "b": b_value,
                    "u": u_value,
                    "team_utility": team_u,
                }
            )
    correlation_summary: dict[str, Any] = {}
    for left, right in (("r", "u"), ("b", "u"), ("r", "team_utility"), ("b", "team_utility"), ("u", "team_utility")):
        xs = [row[left] for row in intervention_metrics]
        ys = [row[right] for row in intervention_metrics]
        correlation_summary[f"pearson_{left}_{right}"] = _pearson(xs, ys)
        correlation_summary[f"spearman_{left}_{right}"] = _pearson(_rank(xs), _rank(ys))

    summary = {
        "n_interventions": len(rows),
        "n_examples": len(grouped),
        "n_candidates_per_example": {str(k): sum(1 for values in grouped.values() if len(values) == k) for k in sorted({len(values) for values in grouped.values()})},
        "fields": {"r": r_field, "b": b_field, "u": u_field},
        "conditions": summary_conditions,
        "correlations": correlation_summary,
        "utility_sign_prediction": _utility_sign_prediction(
            rows,
            r_field=r_field,
            b_field=b_field,
            u_field=u_field,
            seed=seed,
            n_bootstrap=n_bootstrap,
        ),
        "interpretation": {
            "primary_test": "R+B versus U_oracle score_gap_vs_u_oracle and utility_gap_vs_u_oracle",
            "negative_gap_means": "the proxy selector underperforms the U oracle on the same question",
            "u_oracle_is": f"an upper-bound selector using observed {u_field}; it is not available at deployment time",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rbu_selection_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in selection_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (output_dir / "rbu_intervention_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in intervention_metrics:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "rbu_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_summary_csv(summary, output_dir / "rbu_summary.csv")
    cases = [row for row in selection_rows if row["condition"] in {"R+B", "R", "B"} and row["score_gap_vs_u_oracle"] is not None]
    cases.sort(key=lambda row: row["score_gap_vs_u_oracle"])
    with (output_dir / "rbu_regret_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "condition",
        "n_examples",
        "mean_team_score",
        "mean_team_utility",
        "pass_rate",
        "mean_r",
        "mean_b",
        "mean_u",
        "selection_match_rate_to_u_oracle",
        "score_gap_estimate",
        "score_gap_lower",
        "score_gap_upper",
        "utility_gap_estimate",
        "utility_gap_lower",
        "utility_gap_upper",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, values in summary["conditions"].items():
            score_gap = values["score_gap_vs_u_oracle"]
            utility_gap = values["utility_gap_vs_u_oracle"]
            writer.writerow(
                {
                    "condition": condition,
                    "n_examples": values["n_examples"],
                    "mean_team_score": values["mean_team_score"],
                    "mean_team_utility": values["mean_team_utility"],
                    "pass_rate": values["pass_rate"],
                    "mean_r": values["mean_r"],
                    "mean_b": values["mean_b"],
                    "mean_u": values["mean_u"],
                    "selection_match_rate_to_u_oracle": values["selection_match_rate_to_u_oracle"],
                    "score_gap_estimate": score_gap["estimate"],
                    "score_gap_lower": score_gap["lower"],
                    "score_gap_upper": score_gap["upper"],
                    "utility_gap_estimate": utility_gap["estimate"],
                    "utility_gap_lower": utility_gap["lower"],
                    "utility_gap_upper": utility_gap["upper"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze R/B/U substitution in a multi-agent CMI run.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Run directory containing team_interventions.jsonl")
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--input-filename", default="team_interventions.jsonl")
    parser.add_argument("--r-field", default=DEFAULT_R_FIELD)
    parser.add_argument("--b-field", default=DEFAULT_B_FIELD)
    parser.add_argument("--u-field", default=DEFAULT_U_FIELD)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = analyze(
        args.input_dir,
        args.output_dir or args.input_dir / "rbu_analysis",
        r_field=args.r_field,
        b_field=args.b_field,
        u_field=args.u_field,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        input_filename=args.input_filename,
    )
    print(json.dumps({"n_examples": result["n_examples"], "output_dir": str(args.output_dir or args.input_dir / "rbu_analysis")}, ensure_ascii=False))
