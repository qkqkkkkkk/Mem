"""Create compact figures for a multi-agent motivation result directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_figures(input_dir: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = _load_rows(input_dir / "team_interventions.jsonl")
    summary = _load_summary(input_dir / "analysis" / "summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Utility relationship: each point is one intervention; colors show sign agreement.
    colors = []
    for row in rows:
        local_positive = float(row["local_utility"]) > summary["epsilon"]
        team_positive = float(row["team_utility"]) > summary["epsilon"]
        colors.append("#2f6f9f" if local_positive == team_positive else "#c44e52")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter([float(r["local_utility"]) for r in rows], [float(r["team_utility"]) for r in rows], c=colors, alpha=0.78, edgecolor="white", linewidth=0.45)
    limits = [min(-0.1, min(float(r["local_utility"]) for r in rows), min(float(r["team_utility"]) for r in rows)), max(0.1, max(float(r["local_utility"]) for r in rows), max(float(r["team_utility"]) for r in rows))]
    ax.plot(limits, limits, color="#777777", linewidth=1, linestyle="--")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set(xlim=limits, ylim=limits, xlabel="Local utility", ylabel="Team utility", title="Local vs team utility")
    ax.text(0.02, 0.03, f"mismatch = {summary['mismatch_rate']['estimate']:.1%}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    path = output_dir / "utility_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Sign table: useful for the structural hypothesis and direction of mismatches.
    table = summary["contingency_table"]
    matrix = [[table["local_positive_team_positive"], table["local_positive_team_nonpositive"]], [table["local_nonpositive_team_positive"], table["local_nonpositive_team_nonpositive"]]]
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=15)
    ax.set(xticks=[0, 1], xticklabels=["Team +", "Team -"], yticks=[0, 1], yticklabels=["Local +", "Local -"], title="Utility sign contingency")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Interventions")
    fig.tight_layout()
    path = output_dir / "sign_contingency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Task-family mismatch rates with the stored cluster-bootstrap intervals.
    families = sorted(summary["task_family"])
    estimates = [summary["task_family"][f]["mismatch_rate"]["estimate"] for f in families]
    lower = [summary["task_family"][f]["mismatch_rate"]["lower"] for f in families]
    upper = [summary["task_family"][f]["mismatch_rate"]["upper"] for f in families]
    errors = [[e - lo for e, lo in zip(estimates, lower)], [hi - e for e, hi in zip(estimates, upper)]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(range(len(families)), estimates, yerr=errors, fmt="o", color="#3d7c57", capsize=4)
    ax.set(xticks=range(len(families)), xticklabels=[f.replace("_", "\n") for f in families], ylabel="Mismatch rate", ylim=(0, 1), title="Mismatch by task family")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "mismatch_by_task_family.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-agent motivation results.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis" / "figures"
    paths = make_figures(args.input_dir, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "figures": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
