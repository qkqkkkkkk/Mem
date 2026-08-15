from __future__ import annotations

import csv
import json
from pathlib import Path

from motivation_experiment.make_diagnostic_bundle import make_bundle
from motivation_experiment.rejudge_multidim import _multidim_judge


class FakeMultidimJudgeClient:
    def __init__(self):
        self.prompt = ""

    def complete(self, prompt, **kwargs):
        self.prompt = prompt
        return {
            "json": {
                "conclusion_change_score": 0.0,
                "conclusion_confidence": 0.9,
                "factual_change_score": 1.0,
                "factual_confidence": 0.8,
                "action_applicable": False,
                "action_change_score": 0.0,
                "action_confidence": 0.0,
                "conclusion_without": "The project was in marketing.",
                "conclusion_with": "The project was in civil engineering.",
                "explanation": "The answer's factual claim changed.",
            }
        }


def test_multidim_judge_excludes_gold_behavior_and_computes_own_overall_score():
    client = FakeMultidimJudgeClient()
    row = {
        "example_id": "q1",
        "memory_id": "m1",
        "task": "What project was Jolene working on?",
        "gold_behavior": "electricity engineering project",
        "scoring_criteria": {"expected_answer": "electricity engineering project"},
    }

    result = _multidim_judge(
        client,
        {"openai": {"judge_model": "llama3:8b"}},
        row,
        "The project was in marketing.",
        "The project was in civil engineering.",
        rollout_id=0,
    )

    assert "electricity engineering" not in client.prompt
    assert result["valid"] is True
    assert result["overall_change_score"] == 0.5
    assert abs(result["overall_confidence"] - 0.85) < 1e-9


def test_diagnostic_bundle_groups_repeated_warnings_and_marks_harmful_positive_outlier(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    row = {
        "example_id": "q1",
        "memory_id": "m1",
        "label": "harmful",
        "task": "What happened?",
        "gold_behavior": "Correct outcome",
        "memory_content": "Candidate memory",
        "relevance_score": 0.8,
        "behavioral_reliance": 1.0,
        "utility": 0.5,
        "utility_ci_lower": 0.2,
        "utility_ci_upper": 0.7,
        "deterministic_utility": 0.5,
        "llm_utility": 0.5,
        "no_memory_outputs": ["same", "same"],
        "with_memory_outputs": ["same", "same"],
        "llm_judgments": [
            {
                "normalization_warnings": ["old judge contradiction"],
                "decision_without": "NO_ANSWER",
                "decision_with": "NO_ANSWER",
                "reported_same_decision": True,
                "same_decision": False,
                "decision_change_score": 1.0,
                "confidence": 0.0,
                "explanation": "same output",
            },
            {
                "normalization_warnings": ["old judge contradiction"],
                "decision_without": "NO_ANSWER",
                "decision_with": "NO_ANSWER",
                "reported_same_decision": True,
                "same_decision": False,
                "decision_change_score": 1.0,
                "confidence": 0.0,
                "explanation": "same output",
            },
        ],
    }
    (input_dir / "memory_interventions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    output_dir = tmp_path / "diagnostics"
    make_bundle(input_dir, output_dir)

    pairs = list(csv.DictReader((output_dir / "judge_warning_pair_review.csv").open(encoding="utf-8")))
    anomalies = list(csv.DictReader((output_dir / "anomaly_review.csv").open(encoding="utf-8")))
    assert len(pairs) == 1
    assert pairs[0]["warning_rollouts"] == "0;1"
    assert anomalies[0]["review_tags"] == "all_harmful;harmful_positive_utility;harmful_positive_outlier"
