from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..candidates import aggregate_structured_memory
from ..config import ExperimentConfig
from ..core import select_deck
from ..io import memory_node_from_dict


def select_m3_agent_nodes(
    nodes: Iterable[dict[str, Any]],
    *,
    config: ExperimentConfig | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Aggregate M3-Agent node hits and return selected clip identifiers."""

    config = config or ExperimentConfig.from_mapping(
        {
            "name": "m3_agent_deck",
            "method": "deck",
            "candidate_pool": 200,
            "min_k": 2,
            "max_k": 5,
            "start_k": 1,
            "lambda": 0.85,
            "structured_memory": {"enabled": True},
        }
    )
    structured = config.structured_memory
    candidates = aggregate_structured_memory(
        [memory_node_from_dict(item) for item in nodes],
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
    result = select_deck(
        candidates,
        candidate_pool=config.candidate_pool,
        min_k=config.min_k,
        max_k=config.max_k,
        start_k=config.start_k,
        lambda_=config.lambda_,
        temporal_output=True,
    )
    return [item.clip_id for item in result.selected], result.to_dict()
