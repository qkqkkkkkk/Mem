from __future__ import annotations

import json
from pathlib import Path

from multiagent_motivation.analyze_team_results import analyze
from multiagent_motivation.rescore_same_judge import _stored_team_utility
from multiagent_motivation.synthetic_diagnostic.run_synthetic_diagnostic import run
from multiagent_motivation.synthetic_diagnostic.combine_real_and_synthetic import combine
from multiagent_motivation.run_team_pilot import _synth_prompt


def test_synthesizer_prompt_does_not_expose_memory_or_gold_answer():
    prompt = _synth_prompt(
        "When did Caroline attend the group?",
        "Caroline attended the group on 7 May 2023.",
    )
    assert "7 May 2023" in prompt
    assert "Candidate memory" not in prompt
    assert "Expected behavior" not in prompt


def test_team_analysis_reports_both_mismatch_directions(tmp_path: Path):
    input_dir = tmp_path / "team"
    input_dir.mkdir()
    rows = []
    for index, (local, team, family) in enumerate(
        [
            (0.2, 0.3, "temporal_memory_qa"),
            (0.2, -0.1, "temporal_memory_qa"),
            (-0.1, 0.2, "multi_evidence"),
            (-0.1, -0.2, "inferential"),
        ]
    ):
        rows.append(
            {
                "example_id": f"q{index}",
                "memory_id": "m1",
                "task_family": family,
                "local_utility": local,
                "team_utility": team,
                "team_no_memory_outputs": ["no"],
                "team_with_memory_outputs": ["with"],
            }
        )
    (input_dir / "team_interventions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = analyze(input_dir, tmp_path / "analysis", epsilon=0.0, seed=3, n_bootstrap=100)

    assert summary["n_interventions"] == 4
    assert summary["case_counts"]["all_mismatches"] == 2
    assert summary["case_counts"]["local_positive_team_nonpositive"] == 1
    assert summary["case_counts"]["local_nonpositive_team_positive"] == 1
    assert summary["contingency_table"]["local_positive_team_positive"] == 1
    assert summary["contingency_table"]["local_nonpositive_team_nonpositive"] == 1
    assert (tmp_path / "analysis" / "mismatch_cases.jsonl").exists()


def test_same_judge_rescore_reuses_matching_team_score_arrays():
    no_scores, with_scores = _stored_team_utility(
        {"team_no_scores": [0.2, 0.3], "team_with_scores": [0.5, 0.4]}
    )
    assert no_scores == [0.2, 0.3]
    assert with_scores == [0.5, 0.4]


def test_synthetic_oracle_separates_answer_and_resource_mismatch(tmp_path: Path):
    output_dir = tmp_path / "synthetic"
    args = type("Args", (), {
        "cases": Path("multiagent_motivation/synthetic_diagnostic/cases.jsonl"),
        "config": Path("multiagent_motivation/config_local.yaml"),
        "output_dir": output_dir,
        "mode": "oracle",
        "utility_scorer": "deterministic",
        "max_cases": None,
        "utility_epsilon": 0.0,
        "no_cache": False,
    })()
    run(args)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["standard_detected_cases"] == 4
    assert summary["resource_case_standard_detection"] == 0
    assert summary["structural_detected_cases"] == 8


def test_combined_diagnostic_marks_provenance_and_keeps_two_summaries(tmp_path: Path):
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    real = {"example_id": "real_q", "task_family": "real", "local_utility": 1.0, "team_utility": 1.0}
    synthetic = {"case_id": "syn_q", "task_family": "synthetic", "local_utility": 1.0, "team_utility": 1.0, "resource_adjusted_team_utility": -0.5}
    (real_dir / "team_interventions.jsonl").write_text(json.dumps(real) + "\n", encoding="utf-8")
    (synthetic_dir / "synthetic_results.jsonl").write_text(json.dumps(synthetic) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    result = combine(real_dir, synthetic_dir, out)
    assert result["n_total"] == 2
    rows = [json.loads(line) for line in (out / "team_interventions.jsonl").open()]
    assert [row["origin"] for row in rows] == ["real", "synthetic"]
    assert (out / "analysis_answer_only" / "summary.json").exists()
    assert (out / "analysis_structural" / "summary.json").exists()
