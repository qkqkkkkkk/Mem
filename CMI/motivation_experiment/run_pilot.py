from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

# Permit `python motivation_experiment/run_pilot.py` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.cmi_agent import CMIAgent
from src.api.openai_client import OpenAIClient
from src.benchmark.load_dataset import load_examples
from src.memory.memory_card import MemoryCard
from src.utils.io import ensure_dir, load_config, write_json, write_jsonl
from src.utils.text_utils import cosine_similarity, deterministic_embedding, keyword_overlap, tokenize


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _tokens(text: str) -> list[str]:
    return tokenize(text or "")


def lexical_answer_divergence(with_text: str, no_text: str) -> float:
    """Measure how much the answer changes when one memory is injected.

    The metric is intentionally model-agnostic: it combines token-set Jaccard
    distance with normalized sequence distance. It is bounded in [0, 1], where
    zero means identical answers and one means maximal lexical/sequence change.
    """
    with_tokens, no_tokens = set(_tokens(with_text)), set(_tokens(no_text))
    union = with_tokens | no_tokens
    jaccard_distance = 0.0 if not union else 1.0 - len(with_tokens & no_tokens) / len(union)
    sequence_distance = 1.0 - SequenceMatcher(None, no_text or "", with_text or "").ratio()
    return round(max(0.0, min(1.0, 0.5 * jaccard_distance + 0.5 * sequence_distance)), 8)


def behavioral_reliance(with_text: str, no_text: str) -> float:
    """Backward-compatible name for the lexical answer-divergence proxy."""
    return lexical_answer_divergence(with_text, no_text)


def _relevance_components(
    query: str,
    memories: list[MemoryCard],
    client: OpenAIClient,
    config: dict[str, Any],
    require_neural_embeddings: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute embedding-only relevance and the original hybrid score."""
    retrieval = config.get("retrieval", {})
    hybrid = retrieval.get("hybrid", {})
    alpha, beta, gamma = (float(hybrid.get(key, default)) for key, default in (("alpha", 0.7), ("beta", 0.2), ("gamma", 0.1)))
    model = config.get("openai", {}).get("embedding_model", "text-embedding-3-small")
    try:
        embeddings = client.embed([query] + [memory.content for memory in memories], model=model)
        query_embedding, memory_embeddings = embeddings[0], embeddings[1:]
    except Exception:
        query_embedding = deterministic_embedding(query)
        memory_embeddings = [deterministic_embedding(memory.content) for memory in memories]
        client.last_embedding_backend = "deterministic_fallback"
    if require_neural_embeddings and client.last_embedding_backend in {None, "cached_unknown", "deterministic_fallback"}:
        raise RuntimeError(
            f"Neural embeddings were required, but {model!r} was unavailable and retrieval fell back to deterministic embeddings. "
            "For Ollama run `ollama pull nomic-embed-text`."
        )
    if not memories:
        return {}
    min_timestamp = min(memory.timestamp for memory in memories)
    span = max(1, max(memory.timestamp for memory in memories) - min_timestamp)
    scores: dict[str, dict[str, float]] = {}
    for memory, embedding in zip(memories, memory_embeddings):
        embedding_similarity = max(-1.0, min(1.0, cosine_similarity(query_embedding, memory.embedding or embedding)))
        embedding_score = (embedding_similarity + 1.0) / 2.0
        recency = (memory.timestamp - min_timestamp) / span
        lexical = keyword_overlap(query, memory.content)
        scores[memory.memory_id] = {
            "embedding_relevance": round(embedding_similarity, 8),
            "hybrid_relevance": round(alpha * embedding_score + beta * recency + gamma * lexical, 8),
            "lexical_relevance": round(lexical, 8),
            "recency_relevance": round(recency, 8),
        }
    return scores


def _bounded_score(value: Any, default: float | None = None) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return default
    return max(0.0, min(1.0, parsed))


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "1"}:
            return True
        if value.strip().lower() in {"false", "no", "0"}:
            return False
    return default


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def judge_intervention_pair(
    client: OpenAIClient,
    config: dict[str, Any],
    example: Any,
    no_memory_output: str,
    with_memory_output: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge utility and decision change in one label-blind comparison."""
    criteria = _model_dump(example.scoring_criteria)
    prompt = f"""You are an impartial evaluator comparing two answers to the same task.

Task:
{example.current_task.instruction}

Expected behavior:
{example.gold_behavior}

Scoring criteria:
{json.dumps(criteria, ensure_ascii=False)}

Answer WITHOUT the candidate memory:
{no_memory_output}

Answer WITH the candidate memory:
{with_memory_output}

Evaluate task correctness independently of writing style unless style is explicitly required. Then decide whether the substantive final answer, recommendation, tool choice, or action materially changed. Pure paraphrases and extra explanation with the same conclusion are not a decision change. A refusal or "insufficient information" response changing into a concrete answer is a material decision change, as is the reverse direction.

The decision fields must contain the actual short substantive conclusion from each answer. Use "NO_ANSWER" for a refusal or failure to answer. Do not copy the field descriptions shown below.

Return JSON only:
{{
  "score_without": 0.0,
  "score_with": 0.0,
  "decision_without": "actual conclusion or NO_ANSWER",
  "decision_with": "actual conclusion or NO_ANSWER",
  "same_decision": true,
  "decision_change_score": 0.0,
  "confidence": 0.0,
  "explanation": "brief comparison"
}}

Scores and confidence must be in [0,1]. decision_change_score is 0 for the same substantive decision, 1 for a clearly different answer/decision including NO_ANSWER versus a concrete answer, and an intermediate value for a partial material change.
"""
    result = client.complete(
        prompt,
        model=config.get("openai", {}).get("judge_model", config.get("openai", {}).get("agent_model", "gpt-4.1-mini")),
        temperature=0.0,
        max_output_tokens=500,
        json_mode=True,
        metadata=metadata,
    )
    data = result.get("json") or {}
    reported_same_decision = _parse_bool(data.get("same_decision"), False)
    decision_change = _bounded_score(data.get("decision_change_score"), 0.0 if reported_same_decision else 1.0)
    same_decision = bool(decision_change is not None and decision_change <= 0.05)
    decision_without = str(data.get("decision_without", ""))
    decision_with = str(data.get("decision_with", ""))
    copied_placeholders = {
        "concise final answer or decision",
        "actual conclusion or NO_ANSWER",
    }
    validation_errors: list[str] = []
    normalization_warnings: list[str] = []
    if decision_without.strip() in copied_placeholders or decision_with.strip() in copied_placeholders:
        validation_errors.append("judge copied a decision-field placeholder")
    without_is_no_answer = decision_without.strip().upper() == "NO_ANSWER"
    with_is_no_answer = decision_with.strip().upper() == "NO_ANSWER"
    if same_decision and without_is_no_answer != with_is_no_answer:
        validation_errors.append("same_decision=true conflicts with NO_ANSWER versus a concrete decision")
    if reported_same_decision != same_decision:
        normalization_warnings.append(
            "reported same_decision was normalized from decision_change_score"
        )
    return {
        "score_without": _bounded_score(data.get("score_without"), 0.0),
        "score_with": _bounded_score(data.get("score_with"), 0.0),
        "decision_without": decision_without,
        "decision_with": decision_with,
        "same_decision": same_decision,
        "reported_same_decision": reported_same_decision,
        "decision_change_score": decision_change,
        "confidence": _bounded_score(data.get("confidence"), 0.0),
        "explanation": str(data.get("explanation", "")),
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "normalization_warnings": normalization_warnings,
    }


def load_human_annotations(path: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("example_id", ""), row.get("memory_id", ""))
            annotations[key] = row
    return annotations


def _human_scores(annotation: dict[str, Any] | None) -> tuple[float | None, float | None, float | None]:
    if not annotation:
        return None, None, None
    return (
        _bounded_score(annotation.get("human_score_without")),
        _bounded_score(annotation.get("human_score_with")),
        _bounded_score(annotation.get("human_decision_change")),
    )


def _combine_scores(deterministic: float, judged: float | None, deterministic_weight: float, judge_weight: float) -> float:
    if judged is None:
        return deterministic
    total = deterministic_weight + judge_weight
    return deterministic if total <= 0 else (deterministic_weight * deterministic + judge_weight * judged) / total


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    # Correlation is undefined, rather than zero, when either variable is
    # constant. Returning None keeps a degenerate pilot from looking conclusive.
    return numerator / (denom_x * denom_y) if denom_x and denom_y else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for index in order[i : j + 1]:
            ranks[index] = rank
        i = j + 1
    return ranks


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(values: list[float], statistic=_mean, seed: int = 42, n_bootstrap: int = 2000) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "n": 0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimate = statistic(sample)
        if estimate is not None:
            estimates.append(float(estimate))
    return {
        "estimate": float(statistic(values)),
        "lower": _quantile(estimates, 0.025),
        "upper": _quantile(estimates, 0.975),
        "n": len(values),
    }


def _bootstrap_correlation(xs: list[float], ys: list[float], seed: int = 42, n_bootstrap: int = 2000) -> dict[str, Any]:
    if len(xs) != len(ys) or not xs:
        return {"estimate": None, "lower": None, "upper": None, "n": 0}
    observed = _pearson(xs, ys)
    if observed is None:
        return {"estimate": None, "lower": None, "upper": None, "n": len(xs), "reason": "undefined because at least one variable has zero variance"}
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(len(xs)) for _ in xs]
        value = _pearson([xs[i] for i in indices], [ys[i] for i in indices])
        if value is not None:
            estimates.append(value)
    if not estimates:
        return {"estimate": observed, "lower": None, "upper": None, "n": len(xs), "reason": "bootstrap samples had zero variance"}
    return {"estimate": observed, "lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975), "n": len(xs)}


def _cluster_bootstrap_correlation(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    seed: int = 42,
    n_bootstrap: int = 2000,
    rank: bool = False,
    cluster_key: str = "example_id",
) -> dict[str, Any]:
    """Bootstrap complete questions so interventions from one question stay together."""

    if not rows:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "n_clusters": 0, "method": "cluster_bootstrap"}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[cluster_key]), []).append(row)
    clusters = sorted(grouped)

    def statistic(sample: list[dict[str, Any]]) -> float | None:
        xs = [float(row[x_key]) for row in sample]
        ys = [float(row[y_key]) for row in sample]
        return _pearson(_rank(xs), _rank(ys)) if rank else _pearson(xs, ys)

    observed = statistic(rows)
    result = {
        "estimate": observed,
        "lower": None,
        "upper": None,
        "n": len(rows),
        "n_clusters": len(clusters),
        "method": "question_cluster_bootstrap",
    }
    if observed is None:
        result["reason"] = "undefined because at least one variable has zero variance"
        return result

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[dict[str, Any]] = []
        for _ in clusters:
            sample.extend(grouped[rng.choice(clusters)])
        estimate = statistic(sample)
        if estimate is not None:
            estimates.append(estimate)
    if estimates:
        result["lower"] = _quantile(estimates, 0.025)
        result["upper"] = _quantile(estimates, 0.975)
    else:
        result["reason"] = "cluster bootstrap samples had zero variance"
    return result


def _cluster_bootstrap_metric(
    records: list[dict[str, Any]],
    metric,
    seed: int,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["example_id"]), []).append(record)
    clusters = sorted(grouped)

    def evaluate(sample: list[dict[str, Any]]) -> float | None:
        labels = [int(record["target"]) for record in sample]
        if len(set(labels)) < 2:
            return None
        return float(metric(labels, [float(record["probability"]) for record in sample]))

    observed = evaluate(records)
    result = {
        "estimate": observed,
        "lower": None,
        "upper": None,
        "n": len(records),
        "n_clusters": len(clusters),
        "method": "question_cluster_bootstrap_on_oof_predictions",
    }
    if observed is None:
        result["reason"] = "undefined because the target has fewer than two classes"
        return result
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[dict[str, Any]] = []
        for _ in clusters:
            sample.extend(grouped[rng.choice(clusters)])
        value = evaluate(sample)
        if value is not None:
            estimates.append(value)
    if estimates:
        result["lower"] = _quantile(estimates, 0.025)
        result["upper"] = _quantile(estimates, 0.975)
    return result


def _utility_sign_prediction(rows: list[dict[str, Any]], u_epsilon: float, seed: int) -> dict[str, Any]:
    """Predict positive versus negative U using leave-one-question-out predictions."""

    labeled = [row for row in rows if abs(float(row["utility"])) > u_epsilon]
    base = {
        "task": "positive_vs_negative_utility_excluding_neutral",
        "neutral_definition": f"abs(U) <= {u_epsilon}",
        "n": len(labeled),
        "n_excluded_neutral": len(rows) - len(labeled),
        "n_clusters": len({str(row["example_id"]) for row in labeled}),
    }
    targets = [1 if float(row["utility"]) > u_epsilon else 0 for row in labeled]
    if len(labeled) < 4 or len(set(targets)) < 2 or base["n_clusters"] < 2:
        return {**base, "models": {}, "reason": "insufficient non-neutral rows, classes, or question clusters"}

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - dependencies are declared by the repository
        return {**base, "models": {}, "reason": f"scikit-learn unavailable: {exc}"}

    y = np.asarray(targets, dtype=int)
    groups = np.asarray([str(row["example_id"]) for row in labeled])
    feature_sets = {
        "R": ["relevance_score"],
        "B": ["behavioral_reliance"],
        "R_plus_B": ["relevance_score", "behavioral_reliance"],
    }
    models: dict[str, Any] = {}
    splitter = LeaveOneGroupOut()
    for model_index, (name, feature_names) in enumerate(feature_sets.items()):
        x = np.asarray([[float(row[key]) for key in feature_names] for row in labeled], dtype=float)
        probabilities = np.full(len(labeled), np.nan, dtype=float)
        skipped_folds = 0
        for train_indices, test_indices in splitter.split(x, y, groups):
            if len(set(y[train_indices].tolist())) < 2:
                skipped_folds += 1
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed),
            )
            model.fit(x[train_indices], y[train_indices])
            probabilities[test_indices] = model.predict_proba(x[test_indices])[:, 1]

        valid = ~np.isnan(probabilities)
        records = [
            {
                "example_id": str(labeled[index]["example_id"]),
                "target": int(y[index]),
                "probability": float(probabilities[index]),
            }
            for index in range(len(labeled))
            if valid[index]
        ]
        predicted = [1 if record["probability"] >= 0.5 else 0 for record in records]
        actual = [int(record["target"]) for record in records]
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
        models[name] = {
            "features": feature_names,
            "cv": "leave_one_question_out",
            "n_oof": len(records),
            "skipped_folds": skipped_folds,
            "roc_auc": _cluster_bootstrap_metric(records, roc_auc_score, seed + 100 + model_index),
            "average_precision": _cluster_bootstrap_metric(records, average_precision_score, seed + 200 + model_index),
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


def summarize(rows: list[dict[str, Any]], top_relevance_quantile: float, b_quantile: float, u_epsilon: float, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_interventions": 0, "message": "No intervention rows were produced."}
    relevance = [float(row["relevance_score"]) for row in rows]
    embedding_relevance = [float(row["embedding_relevance"]) for row in rows]
    hybrid_relevance = [float(row["hybrid_relevance"]) for row in rows]
    reliance = [float(row["behavioral_reliance"]) for row in rows]
    utility = [float(row["utility"]) for row in rows]
    relevance_cut = _quantile(relevance, top_relevance_quantile)
    b_high_cut = _quantile(reliance, b_quantile)
    b_low_cut = _quantile(reliance, 1.0 - b_quantile)
    separated_bands = b_high_cut > b_low_cut
    top_relevance = [row for row in rows if row["relevance_score"] >= relevance_cut]
    positive = lambda row: float(row["utility"]) > u_epsilon
    negative = lambda row: float(row["utility"]) < -u_epsilon
    neutral = lambda row: not positive(row) and not negative(row)
    quadrants = {
        "high_B_positive_U": [row for row in rows if separated_bands and row["behavioral_reliance"] >= b_high_cut and positive(row)],
        "high_B_negative_U": [row for row in rows if separated_bands and row["behavioral_reliance"] >= b_high_cut and negative(row)],
        "low_B_positive_U": [row for row in rows if separated_bands and row["behavioral_reliance"] <= b_low_cut and positive(row)],
        "low_B_negative_U": [row for row in rows if separated_bands and row["behavioral_reliance"] <= b_low_cut and negative(row)],
    }
    label_stats: dict[str, Any] = {}
    within_label_correlations: dict[str, Any] = {}
    for label in sorted({str(row.get("label", "unknown")) for row in rows}):
        subset = [row for row in rows if row.get("label") == label]
        within_label_correlations[label] = _cluster_bootstrap_correlation(
            subset,
            "relevance_score",
            "utility",
            seed=seed + 20 + len(within_label_correlations),
        )
        label_stats[label] = {
            "n": len(subset),
            "mean_relevance": _mean(float(row["relevance_score"]) for row in subset),
            "mean_embedding_relevance": _mean(float(row["embedding_relevance"]) for row in subset),
            "mean_hybrid_relevance": _mean(float(row["hybrid_relevance"]) for row in subset),
            "mean_B": _mean(float(row["behavioral_reliance"]) for row in subset),
            "mean_U": _mean(float(row["utility"]) for row in subset),
            "negative_U_rate": bootstrap_ci([1.0 if negative(row) else 0.0 for row in subset], seed=seed),
            "neutral_U_rate": bootstrap_ci([1.0 if neutral(row) else 0.0 for row in subset], seed=seed + 10),
        }
    correlations = {
        "pearson_B_U": _cluster_bootstrap_correlation(rows, "behavioral_reliance", "utility", seed=seed),
        "spearman_B_U": _cluster_bootstrap_correlation(rows, "behavioral_reliance", "utility", seed=seed + 1, rank=True),
        "pearson_R_U": _cluster_bootstrap_correlation(rows, "relevance_score", "utility", seed=seed + 2),
        "pearson_embedding_R_U": _cluster_bootstrap_correlation(rows, "embedding_relevance", "utility", seed=seed + 6),
        "pearson_hybrid_R_U": _cluster_bootstrap_correlation(rows, "hybrid_relevance", "utility", seed=seed + 7),
    }
    naive_correlations = {
        "pearson_B_U": _bootstrap_correlation(reliance, utility, seed=seed),
        "spearman_B_U": _bootstrap_correlation(_rank(reliance), _rank(utility), seed=seed + 1),
        "pearson_R_U": _bootstrap_correlation(relevance, utility, seed=seed + 2),
        "pearson_embedding_R_U": _bootstrap_correlation(embedding_relevance, utility, seed=seed + 6),
        "pearson_hybrid_R_U": _bootstrap_correlation(hybrid_relevance, utility, seed=seed + 7),
    }
    quadrant_summary = {
        name: {"n": len(items), "rate": len(items) / len(rows), "memory_ids": [item["memory_id"] for item in items[:20]]}
        for name, items in quadrants.items()
    }
    return {
        "n_interventions": len(rows),
        "n_examples": len({row["example_id"] for row in rows}),
        "thresholds": {
            "top_relevance_quantile": top_relevance_quantile,
            "top_relevance_cutoff": relevance_cut,
            "relevance_metric": rows[0].get("relevance_metric", "embedding"),
            "high_B_quantile": b_quantile,
            "high_B_cutoff": b_high_cut,
            "low_B_cutoff": b_low_cut,
            "B_bands_separated": separated_bands,
            "utility_epsilon": u_epsilon,
        },
        "overall": {
            "mean_R": _mean(relevance),
            "mean_embedding_R": _mean(embedding_relevance),
            "mean_hybrid_R": _mean(hybrid_relevance),
            "mean_B": _mean(reliance),
            "behavior_metric": rows[0].get("behavior_metric", "lexical"),
            "utility_scorer": rows[0].get("utility_scorer", "deterministic"),
            "mean_U": _mean(utility),
            "negative_U_rate": bootstrap_ci([1.0 if negative(row) else 0.0 for row in rows], seed=seed),
            "neutral_U_rate": bootstrap_ci([1.0 if neutral(row) else 0.0 for row in rows], seed=seed + 5),
            "behavior_changed_rate": bootstrap_ci([float(row["behavior_changed"]) for row in rows], seed=seed + 3),
            "rollout_uncertainty": {
                "n_with_estimable_ci": sum(int(row.get("utility_n_rollouts", 0)) >= 3 for row in rows),
                "n_negative_with_95ci": sum(bool(row.get("utility_negative_with_95ci", False)) for row in rows),
                "minimum_rollouts_for_claim": 3,
            },
        },
        "correlations": correlations,
        "naive_correlations": naive_correlations,
        "within_label_correlations": within_label_correlations,
        "utility_sign_prediction": _utility_sign_prediction(rows, u_epsilon, seed),
        "h1_top_relevance_negative_U": bootstrap_ci([1.0 if negative(row) else 0.0 for row in top_relevance], seed=seed + 4),
        "labels": label_stats,
        "quadrants": quadrant_summary,
        "retrieved_candidate_composition": {
            "useful_candidate_rate": _rate(rows, lambda row: row.get("label") == "useful"),
            "harmful_candidate_rate": _rate(rows, lambda row: row.get("label") in {"harmful", "poisoned"}),
        },
    }


def _rate(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    # Rows only contain retrieved candidates; report candidate composition rather
    # than pretending that absent memories were observed interventions.
    matches = sum(1 for row in rows if predicate(row))
    return {"count": matches, "denominator": len(rows), "rate": matches / len(rows) if rows else None}


def _plot(rows: list[dict[str, Any]], summary: dict[str, Any], figure_dir: Path) -> list[str]:
    if not rows or "thresholds" not in summary:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return [f"Plotting skipped: {exc}"]
    ensure_dir(figure_dir)
    saved: list[str] = []
    labels = [str(row.get("label", "unknown")) for row in rows]
    colors = {"useful": "#168aad", "harmful": "#d1495b", "poisoned": "#d1495b", "irrelevant": "#6c757d", "outdated": "#f0a202"}
    degenerate_bu = len({row["behavioral_reliance"] for row in rows}) < 2 or len({row["utility"] for row in rows}) < 2
    plt.figure(figsize=(7, 5))
    for label in sorted(set(labels)):
        subset = [row for row in rows if row.get("label") == label]
        plt.scatter([row["behavioral_reliance"] for row in subset], [row["utility"] for row in subset], label=label, alpha=0.75, color=colors.get(label, "#333333"))
    plt.axhline(0.0, color="#222222", linewidth=0.8)
    b_cut = summary["thresholds"]["high_B_cutoff"]
    plt.axvline(b_cut, color="#555555", linestyle="--", linewidth=0.8)
    behavior_metric = summary.get("overall", {}).get("behavior_metric", "lexical")
    behavior_label = "Decision-change reliance B" if behavior_metric in {"llm_decision", "human_decision"} else "Lexical answer divergence B"
    plt.xlabel(behavior_label)
    plt.ylabel("Causal utility U (score with - score without)")
    plt.title("Behavioral reliance vs causal utility")
    if degenerate_bu:
        plt.text(0.5, 0.06, "Degenerate result: B and/or U has zero variance; no relationship is estimable.", transform=plt.gca().transAxes, ha="center", va="bottom", fontsize=9, color="#8b0000")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = figure_dir / "utility_vs_reliance.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(str(path))

    plt.figure(figsize=(7, 5))
    for label in sorted(set(labels)):
        subset = [row for row in rows if row.get("label") == label]
        plt.scatter([row["relevance_score"] for row in subset], [row["utility"] for row in subset], label=label, alpha=0.75, color=colors.get(label, "#333333"))
    plt.axhline(0.0, color="#222222", linewidth=0.8)
    plt.axvline(summary["thresholds"]["top_relevance_cutoff"], color="#555555", linestyle="--", linewidth=0.8)
    relevance_metric = summary.get("thresholds", {}).get("relevance_metric", "embedding")
    relevance_label = "Embedding-only relevance R" if relevance_metric == "embedding" else "Hybrid retrieval relevance R"
    plt.xlabel(relevance_label)
    plt.ylabel("Causal utility U")
    plt.title("Relevance vs causal utility")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = figure_dir / "relevance_vs_utility.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(str(path))

    prediction = summary.get("utility_sign_prediction", {})
    prediction_models = prediction.get("models", {})
    if prediction_models:
        display_names = [("R", "R"), ("B", "B"), ("R_plus_B", "R + B")]
        display_names = [(key, label) for key, label in display_names if key in prediction_models]
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
        metric_specs = [
            ("roc_auc", "ROC AUC", 0.5),
            ("average_precision", "Average precision", float(prediction.get("average_precision_baseline", 0.0))),
        ]
        for axis, (metric_key, title, baseline) in zip(axes, metric_specs):
            estimates = [float(prediction_models[key][metric_key]["estimate"]) for key, _ in display_names]
            lower = [float(prediction_models[key][metric_key]["lower"]) for key, _ in display_names]
            upper = [float(prediction_models[key][metric_key]["upper"]) for key, _ in display_names]
            positions = list(range(len(display_names)))
            errors = [
                [max(0.0, estimate - low) for estimate, low in zip(estimates, lower)],
                [max(0.0, high - estimate) for estimate, high in zip(estimates, upper)],
            ]
            axis.bar(positions, estimates, color=["#6c757d", "#168aad", "#2a9d8f"][: len(positions)], alpha=0.85)
            axis.errorbar(positions, estimates, yerr=errors, fmt="none", ecolor="#222222", capsize=4, linewidth=1)
            axis.axhline(baseline, color="#555555", linestyle="--", linewidth=0.9)
            axis.set_xticks(positions, [label for _, label in display_names])
            axis.set_title(title)
            axis.set_ylim(0.0, 1.05)
        axes[0].set_ylabel("Question-level out-of-fold performance")
        figure.suptitle("Predicting positive vs negative causal utility")
        figure.tight_layout()
        path = figure_dir / "utility_sign_prediction.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved.append(str(path))
    return saved


def write_human_annotation_template(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "example_id",
        "memory_id",
        "task",
        "gold_behavior",
        "scoring_criteria",
        "memory_content",
        "no_memory_output",
        "with_memory_output",
        "human_score_without",
        "human_score_with",
        "human_decision_change",
        "human_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "example_id": row["example_id"],
                    "memory_id": row["memory_id"],
                    "task": row["task"],
                    "gold_behavior": row["gold_behavior"],
                    "scoring_criteria": json.dumps(row["scoring_criteria"], ensure_ascii=False),
                    "memory_content": row["memory_content"],
                    "no_memory_output": json.dumps(row["no_memory_outputs"], ensure_ascii=False),
                    "with_memory_output": json.dumps(row["with_memory_outputs"], ensure_ascii=False),
                    "human_score_without": row.get("human_s_no") if row.get("human_s_no") is not None else "",
                    "human_score_with": row.get("human_s_with") if row.get("human_s_with") is not None else "",
                    "human_decision_change": row.get("human_decision_change") if row.get("human_decision_change") is not None else "",
                    "human_notes": row.get("human_notes", ""),
                }
            )


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    records: list[dict[str, Any]] = []

    def add(metric: str, value: dict[str, Any] | None, **dimensions: Any) -> None:
        if not value:
            return
        records.append(
            {
                "metric": metric,
                "estimate": value.get("estimate"),
                "lower": value.get("lower"),
                "upper": value.get("upper"),
                "n": value.get("n"),
                "n_clusters": value.get("n_clusters"),
                "method": value.get("method"),
                "label": dimensions.get("label", ""),
                "model": dimensions.get("model", ""),
            }
        )

    for name, value in summary.get("correlations", {}).items():
        add(name, value)
    for name, value in summary.get("naive_correlations", {}).items():
        add(f"naive_{name}", value)
    for label, value in summary.get("within_label_correlations", {}).items():
        add("pearson_R_U_within_label", value, label=label)
    add("top_relevance_negative_U", summary.get("h1_top_relevance_negative_U"))
    for model_name, model in summary.get("utility_sign_prediction", {}).get("models", {}).items():
        add("roc_auc", model.get("roc_auc"), model=model_name)
        add("average_precision", model.get("average_precision"), model=model_name)

    fields = ["metric", "estimate", "lower", "upper", "n", "n_clusters", "method", "label", "model"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    if args.judge_model:
        config.setdefault("openai", {})["judge_model"] = args.judge_model
    if args.generation_temperature is not None:
        config.setdefault("openai", {})["temperature"] = args.generation_temperature
    if args.use_api:
        config.setdefault("openai", {})["use_api"] = True
        config.setdefault("experiment", {})["deterministic_only"] = False
    random.seed(args.seed if args.seed is not None else int(config.get("seed", 42)))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    output_dir = ensure_dir(args.output_dir or Path("motivation_experiment/results") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    client = OpenAIClient(
        use_cache=not args.no_cache,
        use_api=bool(config.get("openai", {}).get("use_api", False)),
        provider=config.get("openai", {}).get("provider"),
        base_url=config.get("openai", {}).get("base_url") or config.get("openai", {}).get("api_url"),
        cache_dir=str(output_dir / "cache"),
    )
    if config.get("openai", {}).get("provider") == "ollama" and (config.get("experiment", {}).get("deterministic_only", False) or not client.use_api):
        raise RuntimeError(
            "Ollama is configured but the pilot would use the deterministic fallback. "
            "Set openai.use_api: true and experiment.deterministic_only: false in the config."
        )
    examples = load_examples(args.dataset, max_examples=args.max_examples)
    human_annotations = load_human_annotations(args.human_annotations)
    if (args.utility_scorer == "human" or args.behavior_scorer == "human_decision") and not human_annotations:
        raise ValueError("Human scoring requires a completed --human-annotations CSV.")
    needs_llm_judge = args.utility_scorer in {"llm", "hybrid"} or args.behavior_scorer == "llm_decision"
    agent_model = str(config.get("openai", {}).get("agent_model", ""))
    judge_model = str(config.get("openai", {}).get("judge_model", agent_model))
    if args.require_independent_judge and needs_llm_judge and judge_model == agent_model:
        raise ValueError(
            f"Independent judging was required, but agent_model and judge_model are both {agent_model!r}. "
            "Set --judge-model to a different local model, for example gemma3:4b."
        )
    generation_temperature = float(config.get("openai", {}).get("temperature", 0.0))
    if args.rollouts > 1 and generation_temperature <= 0.0:
        raise ValueError(
            "Multiple rollouts require a positive generation temperature; otherwise repeated calls are not a meaningful uncertainty estimate. "
            "Set --generation-temperature 0.3 (or another value > 0)."
        )
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for example_index, example in enumerate(examples):
        agent = CMIAgent(config=config, client=client, variant="standard")
        memories = agent.memories_from_example(example)
        query = example.current_task.instruction
        relevance_scores = _relevance_components(
            query,
            memories,
            client,
            config,
            require_neural_embeddings=args.require_neural_embeddings,
        )
        retrieval_key = "embedding_relevance" if args.retrieval_metric == "embedding" else "hybrid_relevance"
        candidates = sorted(memories, key=lambda memory: relevance_scores[memory.memory_id][retrieval_key], reverse=True)[
            : args.top_k or int(config.get("retrieval", {}).get("top_k", 5))
        ]
        candidate_ranks = {memory.memory_id: rank for rank, memory in enumerate(candidates, start=1)}
        example_rows: list[dict[str, Any]] = []
        try:
            no_results = [
                agent._answer_with_memories(
                    example,
                    [],
                    prompt_kind="no_memory",
                    generation_metadata={
                        "purpose": "motivation_no_memory_rollout",
                        "example_id": example.example_id,
                        "rollout_id": rollout_index,
                    },
                )
                for rollout_index in range(args.rollouts)
            ]
            no_texts = [result["text"] for result in no_results]
            deterministic_no_values = [agent.score_text(text, example) for text in no_texts]
            deterministic_s_no = _mean(deterministic_no_values) or 0.0
            for memory in candidates:
                with_results = [
                    agent._answer_with_memories(
                        example,
                        [memory],
                        generation_metadata={
                            "purpose": "motivation_with_memory_rollout",
                            "example_id": example.example_id,
                            "memory_id": memory.memory_id,
                            "rollout_id": rollout_index,
                        },
                    )
                    for rollout_index in range(args.rollouts)
                ]
                with_texts = [result["text"] for result in with_results]
                deterministic_with_values = [agent.score_text(text, example) for text in with_texts]
                deterministic_s_with = _mean(deterministic_with_values) or 0.0
                lexical_b_values = [lexical_answer_divergence(with_text, no_text) for with_text, no_text in zip(with_texts, no_texts)]
                judge_results = (
                    [
                        judge_intervention_pair(
                            client,
                            config,
                            example,
                            no_text,
                            with_text,
                            metadata={
                                "purpose": "motivation_intervention_judge",
                                "example_id": example.example_id,
                                "memory_id": memory.memory_id,
                                "rollout_id": rollout_index,
                            },
                        )
                        for rollout_index, (no_text, with_text) in enumerate(zip(no_texts, with_texts))
                    ]
                    if needs_llm_judge
                    else []
                )
                invalid_judgments = [result for result in judge_results if not result.get("valid", True)]
                if invalid_judgments and not args.allow_invalid_judge:
                    errors = sorted({error for result in invalid_judgments for error in result.get("validation_errors", [])})
                    raise RuntimeError(f"Invalid judge output for {example.example_id}/{memory.memory_id}: {errors}")
                llm_s_no = _mean(result["score_without"] for result in judge_results)
                llm_s_with = _mean(result["score_with"] for result in judge_results)
                llm_decision_change = _mean(result["decision_change_score"] for result in judge_results)
                annotation = human_annotations.get((example.example_id, memory.memory_id))
                human_s_no, human_s_with, human_decision_change = _human_scores(annotation)

                if args.utility_scorer == "deterministic":
                    rollout_s_no = deterministic_no_values
                    rollout_s_with = deterministic_with_values
                elif args.utility_scorer == "llm":
                    if llm_s_no is None or llm_s_with is None:
                        raise RuntimeError("LLM utility scoring returned no valid scores.")
                    rollout_s_no = [float(result["score_without"]) for result in judge_results]
                    rollout_s_with = [float(result["score_with"]) for result in judge_results]
                elif args.utility_scorer == "hybrid":
                    rollout_s_no = [
                        _combine_scores(deterministic, result["score_without"], args.deterministic_weight, args.judge_weight)
                        for deterministic, result in zip(deterministic_no_values, judge_results)
                    ]
                    rollout_s_with = [
                        _combine_scores(deterministic, result["score_with"], args.deterministic_weight, args.judge_weight)
                        for deterministic, result in zip(deterministic_with_values, judge_results)
                    ]
                else:
                    if human_s_no is None or human_s_with is None:
                        raise ValueError(f"Missing human utility scores for {example.example_id}/{memory.memory_id}")
                    rollout_s_no = [human_s_no]
                    rollout_s_with = [human_s_with]
                s_no = _mean(rollout_s_no) or 0.0
                s_with = _mean(rollout_s_with) or 0.0
                rollout_utilities = [with_score - no_score for no_score, with_score in zip(rollout_s_no, rollout_s_with)]
                utility_interval = bootstrap_ci(
                    rollout_utilities,
                    seed=seed + example_index * 100 + candidate_ranks[memory.memory_id],
                )
                utility_ci_estimable = len(rollout_utilities) >= 3

                if args.behavior_scorer == "lexical":
                    b_values = lexical_b_values
                elif args.behavior_scorer == "llm_decision":
                    if llm_decision_change is None:
                        raise RuntimeError("LLM decision scoring returned no valid score.")
                    b_values = [float(result["decision_change_score"]) for result in judge_results]
                else:
                    if human_decision_change is None:
                        raise ValueError(f"Missing human decision-change score for {example.example_id}/{memory.memory_id}")
                    b_values = [human_decision_change]
                perturbed_texts: list[str] = []
                stability = None
                if not args.no_perturbation and bool(config.get("cmi", {}).get("use_perturbation", True)):
                    from src.memory.perturb_memory import choose_perturbation, perturb_memory
                    ptype = choose_perturbation(memory, config.get("cmi", {}).get("perturbation_types"))
                    perturbed = perturb_memory(memory, ptype)
                    perturbed_results = [
                        agent._answer_with_memories(
                            example,
                            [perturbed],
                            generation_metadata={
                                "purpose": "motivation_perturbed_rollout",
                                "example_id": example.example_id,
                                "memory_id": memory.memory_id,
                                "rollout_id": rollout_index,
                            },
                        )
                        for rollout_index in range(args.rollouts)
                    ]
                    perturbed_texts = [result["text"] for result in perturbed_results]
                    s_perturbed = _mean(agent.score_text(text, example) for text in perturbed_texts) or 0.0
                    stability = deterministic_s_with - s_perturbed
                relevance = relevance_scores[memory.memory_id]
                row = {
                    "example_id": example.example_id,
                    "example_index": example_index,
                    "task_family": example.task_family,
                    "task": query,
                    "gold_behavior": example.gold_behavior,
                    "scoring_criteria": _model_dump(example.scoring_criteria),
                    "memory_id": memory.memory_id,
                    "retrieval_rank": candidate_ranks[memory.memory_id],
                    "label": memory.label,
                    "memory_type": memory.memory_type,
                    "memory_content": memory.content,
                    "relevance_metric": args.relevance_metric,
                    "relevance_score": relevance["embedding_relevance"] if args.relevance_metric == "embedding" else relevance["hybrid_relevance"],
                    **relevance,
                    "utility_scorer": args.utility_scorer,
                    "s_no": s_no,
                    "s_with": s_with,
                    "utility": _mean(rollout_utilities) or 0.0,
                    "rollout_utilities": rollout_utilities,
                    "utility_sd": statistics.pstdev(rollout_utilities) if len(rollout_utilities) > 1 else 0.0,
                    "utility_ci_lower": utility_interval["lower"],
                    "utility_ci_upper": utility_interval["upper"],
                    "utility_n_rollouts": len(rollout_utilities),
                    "utility_ci_method": "rollout_bootstrap" if utility_ci_estimable else "not_estimable_fewer_than_3_rollouts",
                    "utility_negative_with_95ci": bool(
                        utility_ci_estimable
                        and utility_interval["upper"] is not None
                        and float(utility_interval["upper"]) < -args.utility_epsilon
                    ),
                    "deterministic_s_no": deterministic_s_no,
                    "deterministic_s_with": deterministic_s_with,
                    "deterministic_utility": deterministic_s_with - deterministic_s_no,
                    "llm_s_no": llm_s_no,
                    "llm_s_with": llm_s_with,
                    "llm_utility": llm_s_with - llm_s_no if llm_s_no is not None and llm_s_with is not None else None,
                    "human_s_no": human_s_no,
                    "human_s_with": human_s_with,
                    "human_utility": human_s_with - human_s_no if human_s_no is not None and human_s_with is not None else None,
                    "behavior_metric": args.behavior_scorer,
                    "behavioral_reliance": _mean(b_values) or 0.0,
                    "behavioral_reliance_sd": statistics.pstdev(b_values) if len(b_values) > 1 else 0.0,
                    "behavior_changed": int(any(value > args.behavior_change_threshold for value in b_values)),
                    "lexical_answer_divergence": _mean(lexical_b_values) or 0.0,
                    "llm_decision_change": llm_decision_change,
                    "human_decision_change": human_decision_change,
                    "human_notes": annotation.get("human_notes", "") if annotation else "",
                    "llm_judgments": judge_results,
                    "invalid_judge_count": len(invalid_judgments),
                    "judge_warning_count": sum(len(result.get("normalization_warnings", [])) for result in judge_results),
                    "stability": stability,
                    "no_memory_outputs": no_texts,
                    "with_memory_outputs": with_texts,
                    "perturbed_memory_outputs": perturbed_texts,
                }
                example_rows.append(row)
            rows.extend(example_rows)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"example_id": example.example_id, "error": str(exc)})
    summary = summarize(rows, args.top_relevance_quantile, args.b_quantile, args.utility_epsilon, seed)
    summary["run"] = {
        "dataset": str(args.dataset),
        "config": str(args.config),
        "provider": client.provider,
        "agent_model": config.get("openai", {}).get("agent_model"),
        "judge_model": config.get("openai", {}).get("judge_model"),
        "judge_independent_from_agent": config.get("openai", {}).get("judge_model") != config.get("openai", {}).get("agent_model"),
        "embedding_model": config.get("openai", {}).get("embedding_model"),
        "embedding_backend": client.last_embedding_backend,
        "retrieval_metric": args.retrieval_metric,
        "relevance_metric": args.relevance_metric,
        "utility_scorer": args.utility_scorer,
        "behavior_scorer": args.behavior_scorer,
        "deterministic_only": bool(config.get("experiment", {}).get("deterministic_only", False)),
        "rollouts": args.rollouts,
        "generation_temperature": generation_temperature,
        "invalid_judge_count": sum(int(row.get("invalid_judge_count", 0)) for row in rows),
        "judge_warning_count": sum(int(row.get("judge_warning_count", 0)) for row in rows),
        "top_k": args.top_k,
        "skipped_examples": skipped,
    }
    write_jsonl(rows, output_dir / "memory_interventions.jsonl")
    write_human_annotation_template(rows, output_dir / "human_annotations.csv")
    write_json(summary, output_dir / "summary.json")
    write_summary_csv(summary, output_dir / "summary.csv")
    summary["figures"] = _plot(rows, summary, output_dir / "figures")
    write_json(summary, output_dir / "summary.json")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CMI behavioral-reliance/causal-utility motivation pilot.")
    parser.add_argument("--dataset", required=True, help="Input CausalMemBench/CAUSAL-LoCoMo JSONL.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("pilot_config.yaml")))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--generation-temperature", type=float, default=None, help="Override openai.temperature. Must be > 0 when --rollouts is greater than 1.")
    parser.add_argument("--judge-model", default=None, help="Override the judge model, for example gemma3:4b with a Qwen agent.")
    parser.add_argument("--require-independent-judge", action="store_true", help="Fail when the judge model is identical to the agent model.")
    parser.add_argument("--allow-invalid-judge", action="store_true", help="Keep judge outputs that fail structural consistency checks instead of skipping the question.")
    parser.add_argument("--retrieval-metric", choices=["hybrid", "embedding"], default="hybrid", help="Metric used to choose top-k candidate memories.")
    parser.add_argument("--relevance-metric", choices=["embedding", "hybrid"], default="embedding", help="R shown in analysis and the relevance/utility figure.")
    parser.add_argument("--require-neural-embeddings", action="store_true", help="Fail instead of silently using deterministic hash embeddings.")
    parser.add_argument("--utility-scorer", choices=["deterministic", "llm", "hybrid", "human"], default="deterministic")
    parser.add_argument("--behavior-scorer", choices=["lexical", "llm_decision", "human_decision"], default="lexical")
    parser.add_argument("--deterministic-weight", type=float, default=0.5, help="Deterministic-score weight when --utility-scorer hybrid.")
    parser.add_argument("--judge-weight", type=float, default=0.5, help="LLM-judge weight when --utility-scorer hybrid.")
    parser.add_argument("--human-annotations", default=None, help="Completed human_annotations.csv used by human scorers.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-relevance-quantile", type=float, default=0.8)
    parser.add_argument("--b-quantile", type=float, default=0.8, help="High-B cutoff quantile; low-B uses its symmetric complement (default: top/bottom 20%%).")
    parser.add_argument("--utility-epsilon", type=float, default=0.0)
    parser.add_argument("--behavior-change-threshold", type=float, default=0.05)
    parser.add_argument("--use-api", action="store_true", help="Force configured live provider. Usually unnecessary when the config already sets use_api: true, including Ollama.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-perturbation", action="store_true")
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")
    if not 0.0 < args.top_relevance_quantile <= 1.0:
        parser.error("--top-relevance-quantile must be in (0, 1]")
    if not 0.5 < args.b_quantile <= 1.0:
        parser.error("--b-quantile must be in (0.5, 1]")
    if args.deterministic_weight < 0 or args.judge_weight < 0:
        parser.error("score weights must be non-negative")
    if args.generation_temperature is not None and args.generation_temperature < 0:
        parser.error("--generation-temperature must be non-negative")
    return args


if __name__ == "__main__":
    path = run(parse_args())
    print(path)
