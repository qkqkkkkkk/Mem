from __future__ import annotations

from pathlib import Path

from .schema import BenchmarkExample
from src.utils.io import read_jsonl


def normalize_memory_timestamps(record: dict) -> dict:
    """Fill missing memory timestamps without mutating the source dataset.

    Some Causal-LoCoMo harmful distractors are synthetic and have no source
    session, so their exported timestamp is ``null``. The schema requires an
    ordered integer timestamp for retrieval. Assigning these memories timestamps
    after the observed memories preserves their original memory-bank order.
    """
    memories = record.get("memory_bank", [])
    if not any(memory.get("timestamp") is None for memory in memories):
        return record

    normalized = dict(record)
    normalized_memories = [dict(memory) for memory in memories]
    last_timestamp = max(
        (memory["timestamp"] for memory in normalized_memories if isinstance(memory.get("timestamp"), int)),
        default=0,
    )
    for memory in normalized_memories:
        if memory.get("timestamp") is None:
            last_timestamp += 1
            memory["timestamp"] = last_timestamp
    normalized["memory_bank"] = normalized_memories
    return normalized


def load_examples(path: str | Path, max_examples: int | None = None) -> list[BenchmarkExample]:
    records = read_jsonl(path)
    if max_examples is not None:
        records = records[:max_examples]
    return [BenchmarkExample(**normalize_memory_timestamps(record)) for record in records]
