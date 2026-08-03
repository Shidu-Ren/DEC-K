from __future__ import annotations

from typing import Any

from .baselines import (
    adaptive_k,
    adaptive_rag,
    fixed_top_k,
    mmr_fixed_top_k,
    relevance_calibrated_depth,
)
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
    trace: dict[str, Any]

    if config.method == "deck":
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
    elif config.method == "fixed":
        selected = fixed_top_k(candidates, int(config.fixed_k or 0))
        trace = {"selected_k": len(selected), "selector": "relevance_fixed"}
    elif config.method == "mmr_fixed":
        selected, trace = mmr_fixed_top_k(
            candidates,
            k=int(config.fixed_k or 0),
            candidate_pool=config.candidate_pool,
            lambda_=config.lambda_,
        )
        trace["selector"] = "mmr_fixed"
    elif config.method == "adaptive_k":
        selected, trace = adaptive_k(
            candidates[: config.candidate_pool],
            min_k=config.min_k,
            max_k=config.max_k,
            extra=config.adaptive_extra,
        )
        trace["selector"] = "adaptive_k"
    elif config.method == "adaptive_rag":
        label = record.get(config.adaptive_label_field)
        if label is None:
            raise ValueError(
                f"Missing Adaptive-RAG label field {config.adaptive_label_field!r}"
            )
        selected, trace = adaptive_rag(
            candidates[: config.candidate_pool],
            label=str(label),
            budgets=config.adaptive_budgets,
        )
        trace["selector"] = "adaptive_rag"
    elif config.method == "relevance_deck":
        selected, trace = relevance_calibrated_depth(
            candidates,
            candidate_pool=config.candidate_pool,
            min_k=config.min_k,
            max_k=config.max_k,
            start_k=config.start_k,
        )
        trace["selector"] = "relevance_deck"
    else:
        raise AssertionError(f"Unhandled method: {config.method}")

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
