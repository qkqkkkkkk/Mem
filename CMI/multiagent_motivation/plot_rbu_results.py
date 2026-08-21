"""Plot R/B/U substitution analysis outputs without making LLM calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_figures(input_dir: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    summary = json.loads((input_dir / "rbu_summary.json").read_text(encoding="utf-8"))
    selections = _load_jsonl(input_dir / "rbu_selection_results.jsonl")
    interventions = _load_jsonl(input_dir / "rbu_intervention_metrics.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    names = [name for name in ("none", "R", "B", "R+B", "U_oracle", "random") if name in summary["conditions"]]
    labels = ["None", "R", "B", "R+B", "U oracle", "Random"]
    colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#bab0ac"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    scores = [summary["conditions"][name]["mean_team_score"] for name in names]
    utilities = [summary["conditions"][name]["mean_team_utility"] for name in names]
    axes[0].bar(labels, scores, color=colors)
    axes[0].set_ylabel("Mean team score")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Selector outcome")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(labels, utilities, color=colors)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("Mean team utility")
    axes[1].set_title("Utility relative to no memory")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = output_dir / "selector_performance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    regret_names = [name for name in ("R", "B", "R+B", "random") if name in summary["conditions"]]
    regret_labels = ["R", "B", "R+B", "Random"]
    estimates = [summary["conditions"][name]["score_gap_vs_u_oracle"]["estimate"] for name in regret_names]
    lower = [summary["conditions"][name]["score_gap_vs_u_oracle"]["lower"] for name in regret_names]
    upper = [summary["conditions"][name]["score_gap_vs_u_oracle"]["upper"] for name in regret_names]
    yerr = [[estimate - lo for estimate, lo in zip(estimates, lower)], [hi - estimate for estimate, hi in zip(estimates, upper)]]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(range(len(regret_names)), estimates, yerr=yerr, fmt="o", capsize=5, color="#c44e52")
    ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(regret_names)), regret_labels)
    ax.set_ylabel("Team score gap vs U oracle")
    ax.set_title("R/B selectors underperform causal-utility oracle")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "regret_vs_u_oracle.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if interventions:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        label_colors = {
            "useful": "#168aad",
            "harmful": "#d1495b",
            "poisoned": "#d1495b",
            "irrelevant": "#6c757d",
            "outdated": "#f0a202",
        }
        for label in sorted({str(row.get("label", "unknown")) for row in interventions}):
            subset = [row for row in interventions if str(row.get("label", "unknown")) == label]
            color = label_colors.get(label, "#333333")
            axes[0].scatter([row["r"] for row in subset], [row["u"] for row in subset], alpha=0.72, color=color, label=label, edgecolor="white", linewidth=0.35)
            axes[1].scatter([row["b"] for row in subset], [row["u"] for row in subset], alpha=0.72, color=color, label=label, edgecolor="white", linewidth=0.35)
        axes[0].set(xlabel="R: hybrid relevance", ylabel="U: team utility", title="Relevance vs causal utility")
        axes[1].set(xlabel="B: behavioral reliance", ylabel="U: team utility", title="Behavioral reliance vs causal utility")
        for ax in axes:
            ax.axhline(0, color="#cccccc", linewidth=0.8)
            ax.legend(frameon=False, fontsize=8, loc="best")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / "relevance_reliance_vs_utility.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

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
        path = output_dir / "utility_sign_prediction.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    # One row per question and selector; useful for auditing mistaken choices.
    if selections:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        condition_order = {name: index for index, name in enumerate(names)}
        for condition in ("R", "B", "R+B", "U_oracle"):
            rows = [row for row in selections if row["condition"] == condition]
            x = [condition_order[condition] + (index % 5 - 2) * 0.035 for index in range(len(rows))]
            ax.scatter(x, [row["team_score"] for row in rows], label=condition, alpha=0.7)
        ax.set_xticks([condition_order[name] for name in names], labels)
        ax.set_ylabel("Team score")
        ax.set_title("Per-question selector scores")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, ncol=4)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        path = output_dir / "per_question_selector_scores.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot R/B/U substitution results.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing rbu_summary.json")
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    paths = make_figures(args.input_dir, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "figures": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
