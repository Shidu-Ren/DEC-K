from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .core import select_deck
from .io import candidate_to_dict, candidates_from_record
from .types import Candidate


def _temporal(items: list[Candidate], enabled: bool) -> list[Candidate]:
    if not enabled:
        return items
    return sorted(
        items,
        key=lambda item: (
            float("inf") if item.start is None else float(item.start),
            item.clip_id,
        ),
    )


def select_record(record: dict[str, Any], config: ExperimentConfig) -> dict[str, Any]:
    candidates = candidates_from_record(record, config)
    result = select_deck(
        candidates,
        candidate_pool=config.candidate_pool,
        min_k=config.min_k,
        max_k=config.max_k,
        start_k=config.start_k,
        lambda_=config.lambda_,
        temporal_output=config.temporal_output,
    )
    selected = result.selected
    trace = result.to_dict()

    selected = _temporal(list(selected), config.temporal_output)
    passthrough = {
        key: value
        for key, value in record.items()
        if key not in {"candidates", "nodes"}
    }
    return {
        **passthrough,
        "experiment": config.name,
        "method": config.method,
        "selected_k": len(selected),
        "selected_clip_ids": [item.clip_id for item in selected],
        "selected_evidence": [candidate_to_dict(item) for item in selected],
        "trace": trace,
    }
