from __future__ import annotations

from src.benchmark.generate_causalmembench import TASK_FAMILIES, generate_examples
from src.benchmark.load_dataset import normalize_memory_timestamps
from src.benchmark.schema import BenchmarkExample
from src.benchmark.validate_dataset import deterministic_checks


def test_generated_examples_validate_all_families():
    examples = generate_examples(len(TASK_FAMILIES), seed=1, task_families=TASK_FAMILIES)
    assert {example.task_family for example in examples} == set(TASK_FAMILIES)
    for example in examples:
        parsed = BenchmarkExample(**example.to_dict())
        assert deterministic_checks(parsed) == []
        assert parsed.memory_bank


def test_missing_memory_timestamps_are_assigned_after_existing_memories():
    record = {
        "memory_bank": [
            {"memory_id": "m1", "timestamp": 2},
            {"memory_id": "m2", "timestamp": 4},
            {"memory_id": "harm_00", "timestamp": None},
            {"memory_id": "harm_01", "timestamp": None},
        ]
    }

    normalized = normalize_memory_timestamps(record)

    assert [memory["timestamp"] for memory in normalized["memory_bank"]] == [2, 4, 5, 6]
    assert record["memory_bank"][2]["timestamp"] is None
