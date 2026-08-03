from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .candidates import MemoryNode, aggregate_structured_memory
from .config import ExperimentConfig
from .types import Candidate


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def _vector(value: dict[str, Any]) -> np.ndarray:
    raw = value.get("vector", value.get("embedding"))
    if raw is None:
        raise ValueError("Each candidate or node requires `vector` or `embedding`")
    return np.asarray(raw, dtype=np.float32)


def candidate_from_dict(value: dict[str, Any]) -> Candidate:
    return Candidate(
        clip_id=str(value["clip_id"]),
        relevance=float(value["relevance"]),
        vector=_vector(value),
        text=str(value.get("text") or ""),
        start=None if value.get("start") is None else float(value["start"]),
        end=None if value.get("end") is None else float(value["end"]),
        metadata=dict(value.get("metadata") or {}),
    )


def memory_node_from_dict(value: dict[str, Any]) -> MemoryNode:
    return MemoryNode(
        node_id=str(value["node_id"]),
        clip_id=str(value["clip_id"]),
        relevance=float(value["relevance"]),
        vector=_vector(value),
        memory_type=str(value.get("memory_type") or value.get("type") or ""),
        text=str(value.get("text") or ""),
        start=None if value.get("start") is None else float(value["start"]),
        end=None if value.get("end") is None else float(value["end"]),
        metadata=dict(value.get("metadata") or {}),
    )


def candidates_from_record(
    record: dict[str, Any], config: ExperimentConfig
) -> list[Candidate]:
    if "candidates" in record:
        return [candidate_from_dict(item) for item in record["candidates"]]
    if "nodes" not in record:
        raise ValueError("Each input row requires either `candidates` or `nodes`")
    structured = config.structured_memory
    if not structured.enabled:
        raise ValueError("Input contains `nodes`, but structured_memory.enabled is false")
    nodes = [memory_node_from_dict(item) for item in record["nodes"]]
    return aggregate_structured_memory(
        nodes,
        top_nodes=structured.top_nodes,
        max_leaders=structured.max_leaders,
        type_window=structured.type_window,
        cluster_threshold=structured.cluster_threshold,
        eta=structured.eta,
        leader_weights=structured.leader_weights,
        type_bonus=structured.type_bonus,
        coverage_weight=structured.coverage_weight,
        coverage_cap=structured.coverage_cap,
    )


def candidate_to_dict(candidate: Candidate, *, include_vector: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "clip_id": candidate.clip_id,
        "relevance": float(candidate.relevance),
        "text": candidate.text,
        "start": candidate.start,
        "end": candidate.end,
        "metadata": candidate.metadata,
    }
    if include_vector:
        value["vector"] = np.asarray(candidate.vector).tolist()
    return value
