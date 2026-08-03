from __future__ import annotations

import numpy as np

from deck import MemoryNode, aggregate_structured_memory


def node(
    node_id: str,
    score: float,
    vector: list[float],
    memory_type: str,
) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        clip_id="7",
        relevance=score,
        vector=np.asarray(vector, dtype=np.float32),
        memory_type=memory_type,
        text=node_id,
    )


def test_structured_memory_clusters_nodes_and_adds_cross_type_bonus() -> None:
    nodes = [
        node("e1", 0.9, [1.0, 0.0], "episodic"),
        node("e2", 0.8, [0.99, 0.01], "episodic"),
        node("s1", 0.7, [0.0, 1.0], "semantic"),
    ]
    with_bonus = aggregate_structured_memory(nodes, type_bonus=0.05)[0]
    without_bonus = aggregate_structured_memory(nodes, type_bonus=0.0)[0]
    assert with_bonus.metadata["cluster_count"] == 2
    assert with_bonus.metadata["cluster_leaders"] == ["e1", "s1"]
    assert with_bonus.relevance == without_bonus.relevance + 0.05
