from __future__ import annotations

import argparse
import csv
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
from src.memory.retrievers import HybridRetriever
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


def behavioral_reliance(with_text: str, no_text: str) -> float:
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


def _hybrid_scores(query: str, memories: list[MemoryCard], client: OpenAIClient, config: dict[str, Any]) -> dict[str, float]:
    retrieval = config.get("retrieval", {})
    hybrid = retrieval.get("hybrid", {})
    alpha, beta, gamma = (float(hybrid.get(key, default)) for key, default in (("alpha", 0.7), ("beta", 0.2), ("gamma", 0.1)))
    model = config.get("openai", {}).get("embedding_model", "text-embedding-3-small")
    try:
        query_embedding = client.embed([query], model=model)[0]
    except Exception:
        query_embedding = deterministic_embedding(query)
    if not memories:
        return {}
    min_timestamp = min(memory.timestamp for memory in memories)
    span = max(1, max(memory.timestamp for memory in memories) - min_timestamp)
    scores: dict[str, float] = {}
    for memory in memories:
        try:
            embedding = memory.embedding or client.embed([memory.content], model=model)[0]
        except Exception:
            embedding = memory.embedding or deterministic_embedding(memory.content)
        embedding_score = (cosine_similarity(query_embedding, embedding) + 1.0) / 2.0
        recency = (memory.timestamp - min_timestamp) / span
        lexical = keyword_overlap(query, memory.content)
        scores[memory.memory_id] = round(alpha * embedding_score + beta * recency + gamma * lexical, 8)
    return scores


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
    return numerator / (denom_x * denom_y) if denom_x and denom_y else 0.0


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
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(len(xs)) for _ in xs]
        value = _pearson([xs[i] for i in indices], [ys[i] for i in indices])
        if value is not None:
            estimates.append(value)
    return {"estimate": observed, "lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975), "n": len(xs)}


def summarize(rows: list[dict[str, Any]], top_relevance_quantile: float, b_quantile: float, u_epsilon: float, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_interventions": 0, "message": "No intervention rows were produced."}
    relevance = [float(row["relevance_score"]) for row in rows]
    reliance = [float(row["behavioral_reliance"]) for row in rows]
    utility = [float(row["utility"]) for row in rows]
    relevance_cut = _quantile(relevance, top_relevance_quantile)
    b_high_cut = _quantile(reliance, b_quantile)
    b_low_cut = _quantile(reliance, 1.0 - b_quantile)
    separated_bands = b_high_cut > b_low_cut
    top_relevance = [row for row in rows if row["relevance_score"] >= relevance_cut]
    negative = lambda row: float(row["utility"]) <= u_epsilon
    quadrants = {
        "high_B_positive_U": [row for row in rows if separated_bands and row["behavioral_reliance"] >= b_high_cut and not negative(row)],
        "high_B_negative_U": [row for row in rows if separated_bands and row["behavioral_reliance"] >= b_high_cut and negative(row)],
        "low_B_positive_U": [row for row in rows if separated_bands and row["behavioral_reliance"] <= b_low_cut and not negative(row)],
        "low_B_negative_U": [row for row in rows if separated_bands and row["behavioral_reliance"] <= b_low_cut and negative(row)],
    }
    label_stats: dict[str, Any] = {}
    for label in sorted({str(row.get("label", "unknown")) for row in rows}):
        subset = [row for row in rows if row.get("label") == label]
        label_stats[label] = {
            "n": len(subset),
            "mean_relevance": _mean(float(row["relevance_score"]) for row in subset),
            "mean_B": _mean(float(row["behavioral_reliance"]) for row in subset),
            "mean_U": _mean(float(row["utility"]) for row in subset),
            "negative_U_rate": bootstrap_ci([1.0 if negative(row) else 0.0 for row in subset], seed=seed),
        }
    correlations = {
        "pearson_B_U": _bootstrap_correlation(reliance, utility, seed=seed),
        "spearman_B_U": _bootstrap_correlation(_rank(reliance), _rank(utility), seed=seed + 1),
        "pearson_R_U": _bootstrap_correlation(relevance, utility, seed=seed + 2),
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
            "high_B_quantile": b_quantile,
            "high_B_cutoff": b_high_cut,
            "low_B_cutoff": b_low_cut,
            "B_bands_separated": separated_bands,
            "utility_epsilon": u_epsilon,
        },
        "overall": {
            "mean_R": _mean(relevance),
            "mean_B": _mean(reliance),
            "mean_U": _mean(utility),
            "negative_U_rate": bootstrap_ci([1.0 if negative(row) else 0.0 for row in rows], seed=seed),
            "behavior_changed_rate": bootstrap_ci([float(row["behavior_changed"]) for row in rows], seed=seed + 3),
        },
        "correlations": correlations,
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
    plt.figure(figsize=(7, 5))
    for label in sorted(set(labels)):
        subset = [row for row in rows if row.get("label") == label]
        plt.scatter([row["behavioral_reliance"] for row in subset], [row["utility"] for row in subset], label=label, alpha=0.75, color=colors.get(label, "#333333"))
    plt.axhline(0.0, color="#222222", linewidth=0.8)
    b_cut = summary["thresholds"]["high_B_cutoff"]
    plt.axvline(b_cut, color="#555555", linestyle="--", linewidth=0.8)
    plt.xlabel("Behavioral reliance B (answer divergence)")
    plt.ylabel("Causal utility U (score with - score without)")
    plt.title("Behavioral reliance vs causal utility")
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
    plt.xlabel("Retrieval relevance R")
    plt.ylabel("Causal utility U")
    plt.title("Relevance vs causal utility")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = figure_dir / "relevance_vs_utility.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(str(path))
    return saved


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    config.setdefault("experiment", {})["deterministic_only"] = not args.use_api
    if args.use_api:
        config.setdefault("openai", {})["use_api"] = True
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
    examples = load_examples(args.dataset, max_examples=args.max_examples)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for example_index, example in enumerate(examples):
        agent = CMIAgent(config=config, client=client, variant="standard")
        memories = agent.memories_from_example(example)
        query = example.current_task.instruction
        relevance_scores = _hybrid_scores(query, memories, client, config)
        retriever = HybridRetriever(
            alpha=float(config.get("retrieval", {}).get("hybrid", {}).get("alpha", 0.7)),
            beta=float(config.get("retrieval", {}).get("hybrid", {}).get("beta", 0.2)),
            gamma=float(config.get("retrieval", {}).get("hybrid", {}).get("gamma", 0.1)),
            client=client,
            model=config.get("openai", {}).get("embedding_model", "text-embedding-3-small"),
        )
        candidates = retriever.retrieve(query, memories, k=args.top_k or int(config.get("retrieval", {}).get("top_k", 5)))
        candidate_ranks = {memory.memory_id: rank for rank, memory in enumerate(candidates, start=1)}
        try:
            no_results = [agent._answer_with_memories(example, [], prompt_kind="no_memory") for _ in range(args.rollouts)]
            no_texts = [result["text"] for result in no_results]
            s_no = _mean(agent.score_text(text, example) for text in no_texts) or 0.0
            for memory in candidates:
                with_results = [agent._answer_with_memories(example, [memory]) for _ in range(args.rollouts)]
                with_texts = [result["text"] for result in with_results]
                s_with = _mean(agent.score_text(text, example) for text in with_texts) or 0.0
                b_values = [behavioral_reliance(with_text, no_text) for with_text, no_text in zip(with_texts, no_texts)]
                perturbed_texts: list[str] = []
                stability = None
                if not args.no_perturbation and bool(config.get("cmi", {}).get("use_perturbation", True)):
                    from src.memory.perturb_memory import choose_perturbation, perturb_memory
                    ptype = choose_perturbation(memory, config.get("cmi", {}).get("perturbation_types"))
                    perturbed = perturb_memory(memory, ptype)
                    perturbed_results = [agent._answer_with_memories(example, [perturbed]) for _ in range(args.rollouts)]
                    perturbed_texts = [result["text"] for result in perturbed_results]
                    s_perturbed = _mean(agent.score_text(text, example) for text in perturbed_texts) or 0.0
                    stability = s_with - s_perturbed
                row = {
                    "example_id": example.example_id,
                    "example_index": example_index,
                    "task_family": example.task_family,
                    "memory_id": memory.memory_id,
                    "retrieval_rank": candidate_ranks[memory.memory_id],
                    "label": memory.label,
                    "memory_type": memory.memory_type,
                    "memory_content": memory.content,
                    "relevance_score": relevance_scores.get(memory.memory_id, 0.0),
                    "s_no": s_no,
                    "s_with": s_with,
                    "utility": s_with - s_no,
                    "behavioral_reliance": _mean(b_values) or 0.0,
                    "behavioral_reliance_sd": statistics.pstdev(b_values) if len(b_values) > 1 else 0.0,
                    "behavior_changed": int(any(value > args.behavior_change_threshold for value in b_values)),
                    "stability": stability,
                    "no_memory_outputs": no_texts,
                    "with_memory_outputs": with_texts,
                    "perturbed_memory_outputs": perturbed_texts,
                }
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"example_id": example.example_id, "error": str(exc)})
    summary = summarize(rows, args.top_relevance_quantile, args.b_quantile, args.utility_epsilon, seed)
    summary["run"] = {"dataset": str(args.dataset), "config": str(args.config), "rollouts": args.rollouts, "top_k": args.top_k, "skipped_examples": skipped}
    write_jsonl(rows, output_dir / "memory_interventions.jsonl")
    write_json(summary, output_dir / "summary.json")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "estimate", "lower", "upper", "n"])
        for name, value in [("pearson_B_U", summary.get("correlations", {}).get("pearson_B_U")), ("spearman_B_U", summary.get("correlations", {}).get("spearman_B_U")), ("pearson_R_U", summary.get("correlations", {}).get("pearson_R_U")), ("top_relevance_negative_U", summary.get("h1_top_relevance_negative_U"))]:
            if value:
                writer.writerow([name, value.get("estimate"), value.get("lower"), value.get("upper"), value.get("n")])
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-relevance-quantile", type=float, default=0.8)
    parser.add_argument("--b-quantile", type=float, default=0.8, help="High-B cutoff quantile; low-B uses its symmetric complement (default: top/bottom 20%%).")
    parser.add_argument("--utility-epsilon", type=float, default=0.0)
    parser.add_argument("--behavior-change-threshold", type=float, default=0.05)
    parser.add_argument("--use-api", action="store_true", help="Use the configured OpenAI model instead of local fallback.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-perturbation", action="store_true")
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")
    if not 0.0 < args.top_relevance_quantile <= 1.0:
        parser.error("--top-relevance-quantile must be in (0, 1]")
    if not 0.5 < args.b_quantile <= 1.0:
        parser.error("--b-quantile must be in (0.5, 1]")
    return args


if __name__ == "__main__":
    path = run(parse_args())
    print(path)
