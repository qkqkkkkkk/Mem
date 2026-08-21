from __future__ import annotations

import json

import pytest

from multiagent_motivation.analyze_rbu_substitution import analyze


def _intervention(
    example_id: str,
    memory_id: str,
    relevance: float,
    reliance: float,
    team_with: float,
) -> dict:
    return {
        "example_id": example_id,
        "task_family": "synthetic",
        "memory_id": memory_id,
        "label": "useful" if memory_id == "useful" else "harmful",
        "hybrid_relevance": relevance,
        "behavioral_reliance": reliance,
        "team_no_scores": [0.1],
        "team_with_scores": [team_with],
        "team_utility": team_with - 0.1,
    }


def test_r_and_b_can_underperform_observed_team_utility(tmp_path):
    rows = [
        _intervention("q1", "useful", 0.1, 0.1, 0.9),
        _intervention("q1", "harmful", 0.9, 0.9, 0.2),
        _intervention("q2", "useful", 0.2, 0.2, 0.8),
        _intervention("q2", "harmful", 0.8, 0.8, 0.3),
    ]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with (input_dir / "team_interventions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = analyze(input_dir, tmp_path / "output", n_bootstrap=20)

    r_plus_b = summary["conditions"]["R+B"]
    assert r_plus_b["selection_match_rate_to_u_oracle"] == 0.0
    assert r_plus_b["score_gap_vs_u_oracle"]["estimate"] == pytest.approx(-0.6)
    assert summary["conditions"]["U_oracle"]["mean_team_score"] == pytest.approx(0.85)
    assert (tmp_path / "output" / "rbu_regret_cases.jsonl").exists()
