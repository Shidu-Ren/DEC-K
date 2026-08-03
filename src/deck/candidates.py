from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import normalize_vector
from .types import Candidate


@dataclass
class MemoryNode:
    node_id: str
    clip_id: str
    relevance: float
    vector: np.ndarray
    memory_type: str = ""
    text: str = ""
    start: float | None = None
    end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Cluster:
    members: list[MemoryNode]
    types: set[str]
    weighted_sum: np.ndarray
    weight_sum: float
    centroid: np.ndarray
    leader: MemoryNode


def caption_candidate(
    *,
    clip_id: str,
    relevance: float,
    vector: np.ndarray,
    text: str,
    start: float | None = None,
    end: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Candidate:
    return Candidate(
        clip_id=str(clip_id),
        relevance=float(relevance),
        vector=normalize_vector(vector),
        text=text,
        start=start,
        end=end,
        metadata=dict(metadata or {}),
    )


def _cluster_nodes(
    nodes: list[MemoryNode],
    *,
    threshold: float,
    eta: float,
) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    for node in nodes:
        vector = normalize_vector(node.vector)
        weight = max(float(node.relevance), eta)
        best_index: int | None = None
        best_similarity = float("-inf")
        for index, cluster in enumerate(clusters):
            similarity = float(np.dot(vector, cluster.centroid))
            if similarity > best_similarity:
                best_index = index
                best_similarity = similarity

        if best_index is None or best_similarity < threshold:
            weighted_sum = vector * weight
            clusters.append(
                _Cluster(
                    members=[node],
                    types={node.memory_type} if node.memory_type else set(),
                    weighted_sum=weighted_sum,
                    weight_sum=weight,
                    centroid=normalize_vector(weighted_sum),
                    leader=node,
                )
            )
            continue

        cluster = clusters[best_index]
        cluster.members.append(node)
        if node.memory_type:
            cluster.types.add(node.memory_type)
        cluster.weighted_sum = cluster.weighted_sum + vector * weight
        cluster.weight_sum += weight
        cluster.centroid = normalize_vector(cluster.weighted_sum / cluster.weight_sum)
        if node.relevance > cluster.leader.relevance:
            cluster.leader = node

    return sorted(clusters, key=lambda item: item.leader.relevance, reverse=True)


def aggregate_structured_memory(
    nodes: Iterable[MemoryNode],
    *,
    top_nodes: int = 8,
    max_leaders: int = 4,
    type_window: int = 3,
    cluster_threshold: float = 0.85,
    eta: float = 1e-3,
    leader_weights: tuple[float, ...] = (1.0, 0.35, 0.15, 0.05),
    type_bonus: float = 0.05,
    coverage_weight: float = 0.03,
    coverage_cap: int = 3,
) -> list[Candidate]:
    """Aggregate M3-Agent-style nodes into one candidate per video clip."""

    by_clip: dict[str, list[MemoryNode]] = {}
    for node in nodes:
        by_clip.setdefault(str(node.clip_id), []).append(node)

    candidates: list[Candidate] = []
    for clip_id, clip_nodes in by_clip.items():
        ordered = sorted(
            clip_nodes,
            key=lambda item: -float(item.relevance),
        )[: max(1, int(top_nodes))]
        clusters = _cluster_nodes(
            ordered,
            threshold=float(cluster_threshold),
            eta=float(eta),
        )
        leaders = clusters[: min(max_leaders, len(clusters), len(leader_weights))]
        if not leaders:
            continue

        relevance = sum(
            leader_weights[index] * float(cluster.leader.relevance)
            for index, cluster in enumerate(leaders)
        )
        leading_types = {
            cluster.leader.memory_type
            for cluster in clusters[:type_window]
            if cluster.leader.memory_type
        }
        if {"episodic", "semantic"}.issubset(leading_types):
            relevance += type_bonus
        relevance += coverage_weight * min(max(len(clusters) - 1, 0), coverage_cap)

        representation = np.zeros_like(leaders[0].centroid, dtype=np.float32)
        for index, cluster in enumerate(leaders):
            representation += (
                leader_weights[index]
                * max(float(cluster.leader.relevance), eta)
                * cluster.centroid
            )

        start_values = [node.start for node in ordered if node.start is not None]
        end_values = [node.end for node in ordered if node.end is not None]
        candidates.append(
            Candidate(
                clip_id=clip_id,
                relevance=float(relevance),
                vector=normalize_vector(representation),
                text="\n".join(node.text for node in ordered if node.text),
                start=min(start_values) if start_values else None,
                end=max(end_values) if end_values else None,
                metadata={
                    "node_ids": [node.node_id for node in ordered],
                    "cluster_count": len(clusters),
                    "cluster_leaders": [cluster.leader.node_id for cluster in leaders],
                    "memory_types": sorted(leading_types),
                },
            )
        )
    return candidates
