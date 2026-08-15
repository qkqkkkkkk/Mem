from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from motivation_experiment.run_pilot import (
    _cluster_bootstrap_correlation,
    _parse_bool,
    _relevance_components,
    _utility_sign_prediction,
    judge_intervention_pair,
    lexical_answer_divergence,
    summarize,
    write_human_annotation_template,
)
from src.api.openai_client import OpenAIClient
from src.memory.memory_card import MemoryCard


class FakeEmbeddingClient:
    last_embedding_backend = None

    def embed(self, texts, model):
        self.last_embedding_backend = "ollama"
        return [[1.0, 0.0] if index == 0 else [0.8, 0.2] for index, _ in enumerate(texts)]


class FakeJudgeClient:
    def complete(self, *args, **kwargs):
        return {
            "json": {
                "score_without": 0.0,
                "score_with": 1.0,
                "decision_without": "NO_ANSWER",
                "decision_with": "psychology and counseling certification",
                "same_decision": True,
                "decision_change_score": 0.0,
                "confidence": 1.0,
                "explanation": "The second answer is correct.",
            }
        }


class FakeSoftMismatchJudgeClient:
    def complete(self, *args, **kwargs):
        return {
            "json": {
                "score_without": 0.5,
                "score_with": 0.6,
                "decision_without": "recommend option A",
                "decision_with": "recommend option A with a material caveat",
                "same_decision": True,
                "decision_change_score": 0.2,
                "confidence": 0.8,
                "explanation": "The recommendation changed partially.",
            }
        }


def test_embedding_relevance_is_separate_from_hybrid_relevance():
    memory = MemoryCard(
        memory_id="m1",
        content="candidate memory",
        memory_type="fact",
        label="useful",
        timestamp=10,
    )
    scores = _relevance_components(
        "task",
        [memory],
        FakeEmbeddingClient(),
        {"openai": {"embedding_model": "nomic-embed-text"}, "retrieval": {"hybrid": {}}},
        require_neural_embeddings=True,
    )["m1"]

    assert scores["embedding_relevance"] != scores["hybrid_relevance"]
    assert set(scores) == {
        "embedding_relevance",
        "hybrid_relevance",
        "lexical_relevance",
        "recency_relevance",
    }


def test_ollama_embedding_retries_cached_deterministic_fallback(tmp_path: Path, monkeypatch):
    client = OpenAIClient(
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        use_api=True,
        provider="ollama",
        base_url="http://127.0.0.1:11434",
    )
    texts = ["query"]
    model = "nomic-embed-text"
    payload = {
        "kind": "embedding",
        "provider": "ollama",
        "use_api": True,
        "base_url": "http://127.0.0.1:11434",
        "model": model,
        "texts": texts,
    }
    client.cache.set(payload, {"embeddings": [[0.0, 1.0]], "backend": "deterministic_fallback"})
    monkeypatch.setattr(client, "_embed_ollama", lambda requested_texts, requested_model: [[1.0, 0.0]])

    embeddings = client.embed(texts, model=model)

    assert embeddings == [[1.0, 0.0]]
    assert client.last_embedding_backend == "ollama"
    assert client.cache.get(payload)["backend"] == "ollama"


def test_constant_metrics_are_neutral_and_correlation_is_undefined():
    rows = [
        {
            "example_id": "e1",
            "memory_id": "m1",
            "label": "useful",
            "relevance_metric": "embedding",
            "relevance_score": 0.5,
            "embedding_relevance": 0.5,
            "hybrid_relevance": 0.6,
            "utility_scorer": "human",
            "utility": 0.0,
            "behavior_metric": "human_decision",
            "behavioral_reliance": 0.0,
            "behavior_changed": 0,
        },
        {
            "example_id": "e2",
            "memory_id": "m2",
            "label": "harmful",
            "relevance_metric": "embedding",
            "relevance_score": 0.5,
            "embedding_relevance": 0.5,
            "hybrid_relevance": 0.7,
            "utility_scorer": "human",
            "utility": 0.0,
            "behavior_metric": "human_decision",
            "behavioral_reliance": 0.0,
            "behavior_changed": 0,
        },
    ]

    summary = summarize(rows, 0.8, 0.8, 0.0, 42)

    assert summary["correlations"]["pearson_B_U"]["estimate"] is None
    assert summary["correlations"]["pearson_B_U"]["method"] == "question_cluster_bootstrap"
    assert summary["overall"]["negative_U_rate"]["estimate"] == 0.0
    assert summary["overall"]["neutral_U_rate"]["estimate"] == 1.0


def test_human_template_is_blinded_to_role_label(tmp_path: Path):
    path = tmp_path / "human.csv"
    row = {
        "example_id": "e1",
        "memory_id": "m1",
        "task": "Answer the question",
        "gold_behavior": "Answer yes",
        "scoring_criteria": {"expected_answer": "yes"},
        "label": "harmful",
        "memory_content": "candidate",
        "no_memory_outputs": ["no"],
        "with_memory_outputs": ["yes"],
    }

    write_human_annotation_template([row], path)
    record = next(csv.DictReader(path.open(encoding="utf-8")))

    assert "label" not in record
    assert record["gold_behavior"] == "Answer yes"
    assert record["human_score_with"] == ""


def test_decision_helpers_do_not_treat_false_string_as_true():
    assert _parse_bool("false") is False
    assert _parse_bool("true") is True
    assert lexical_answer_divergence("same answer", "same answer") == 0.0
    assert lexical_answer_divergence("yes", "no") > 0.9


def test_judge_rejects_no_answer_concrete_answer_as_same_decision():
    example = SimpleNamespace(
        current_task=SimpleNamespace(instruction="What should Caroline study?"),
        gold_behavior="Psychology and counseling certification",
        scoring_criteria={"expected_answer": "Psychology and counseling certification"},
    )

    result = judge_intervention_pair(
        FakeJudgeClient(),
        {"openai": {"agent_model": "qwen", "judge_model": "gemma"}},
        example,
        "There is not enough information.",
        "Psychology and counseling certification.",
    )

    assert result["valid"] is False
    assert "same_decision=true conflicts with NO_ANSWER versus a concrete decision" in result["validation_errors"]


def test_judge_normalizes_redundant_same_decision_mismatch():
    example = SimpleNamespace(
        current_task=SimpleNamespace(instruction="Choose an option"),
        gold_behavior="Recommend the best option",
        scoring_criteria={"expected_answer": "option A"},
    )

    result = judge_intervention_pair(
        FakeSoftMismatchJudgeClient(),
        {"openai": {"agent_model": "qwen", "judge_model": "gemma"}},
        example,
        "Choose option A.",
        "Choose option A, but only under condition X.",
    )

    assert result["valid"] is True
    assert result["reported_same_decision"] is True
    assert result["same_decision"] is False
    assert result["decision_change_score"] == 0.2
    assert result["normalization_warnings"]


def test_cluster_bootstrap_keeps_question_interventions_together():
    rows = [
        {"example_id": "q1", "relevance_score": 0.1, "utility": -0.4},
        {"example_id": "q1", "relevance_score": 0.2, "utility": -0.2},
        {"example_id": "q2", "relevance_score": 0.8, "utility": 0.3},
        {"example_id": "q2", "relevance_score": 0.9, "utility": 0.5},
    ]

    result = _cluster_bootstrap_correlation(rows, "relevance_score", "utility", seed=7, n_bootstrap=100)

    assert result["method"] == "question_cluster_bootstrap"
    assert result["n"] == 4
    assert result["n_clusters"] == 2
    assert result["estimate"] > 0.9


def test_utility_sign_prediction_uses_question_level_oof_predictions():
    rows = []
    for question_index in range(6):
        rows.extend(
            [
                {
                    "example_id": f"q{question_index}",
                    "relevance_score": 0.9,
                    "behavioral_reliance": 1.0,
                    "utility": 0.5,
                },
                {
                    "example_id": f"q{question_index}",
                    "relevance_score": 0.1,
                    "behavioral_reliance": 0.0,
                    "utility": -0.5,
                },
            ]
        )

    result = _utility_sign_prediction(rows, u_epsilon=0.0, seed=11)

    assert result["n_clusters"] == 6
    assert result["n_excluded_neutral"] == 0
    assert result["models"]["R_plus_B"]["cv"] == "leave_one_question_out"
    assert result["models"]["R_plus_B"]["roc_auc"]["estimate"] == 1.0
