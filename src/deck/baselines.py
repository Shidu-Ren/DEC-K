from __future__ import annotations

from collections.abc import Mapping, Sequence

from .core import calibrated_depth, minmax_scale, sequential_mmr
from .types import Candidate


def relevance_order(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -float(item.relevance),
            float("inf") if item.start is None else float(item.start),
            item.clip_id,
        ),
    )


def fixed_top_k(candidates: Sequence[Candidate], k: int) -> list[Candidate]:
    return relevance_order(candidates)[: max(0, int(k))]


def mmr_fixed_top_k(
    candidates: Sequence[Candidate],
    *,
    k: int,
    candidate_pool: int = 200,
    lambda_: float = 0.85,
) -> tuple[list[Candidate], dict[str, object]]:
    pool = relevance_order(candidates)[: max(0, int(candidate_pool))]
    selected, steps = sequential_mmr(pool, steps=int(k), lambda_=lambda_)
    return selected, {
        "selected_k": len(selected),
        "selected_clip_ids": [item.clip_id for item in selected],
        "mmr_steps": [step.__dict__ for step in steps],
    }


def relevance_calibrated_depth(
    candidates: Sequence[Candidate],
    *,
    candidate_pool: int = 200,
    min_k: int = 2,
    max_k: int = 8,
    start_k: int = 2,
) -> tuple[list[Candidate], dict[str, object]]:
    """Ablation: apply the DEC-K depth rule to relevance order only."""

    pool = relevance_order(candidates)[: max(0, int(candidate_pool))]
    observed = pool[: min(len(pool), int(max_k) + 1)]
    scores = minmax_scale([item.relevance for item in observed]).tolist()
    selected_k, depth_scores, reason = calibrated_depth(
        scores,
        min_k=min_k,
        max_k=max_k,
        start_k=start_k,
    )
    return observed[:selected_k], {
        "selected_k": selected_k,
        "selected_clip_ids": [item.clip_id for item in observed[:selected_k]],
        "observed_clip_ids": [item.clip_id for item in observed],
        "normalized_relevance": scores,
        "depth_scores": [item.__dict__ for item in depth_scores],
        "stop_reason": reason,
    }


def adaptive_k(
    candidates: Sequence[Candidate],
    *,
    min_k: int,
    max_k: int,
    extra: int,
) -> tuple[list[Candidate], dict[str, int | float]]:
    """Official largest-gap Adaptive-k followed by the paper's budget offset."""

    ranked = relevance_order(candidates)
    if not ranked:
        return [], {"base_k": 0, "selected_k": 0, "extra": int(extra)}
    decision_limit = len(ranked) - 1
    if decision_limit <= 0:
        selected_k = min(len(ranked), max(1, int(min_k)))
        return ranked[:selected_k], {
            "base_k": selected_k,
            "selected_k": selected_k,
            "extra": int(extra),
        }

    normalized_scores = minmax_scale([item.relevance for item in ranked]).tolist()
    gaps = [
        float(normalized_scores[index] - normalized_scores[index + 1])
        for index in range(decision_limit)
    ]
    base_k = max(range(1, decision_limit + 1), key=lambda k: (gaps[k - 1], -k))
    selected_k = min(len(ranked), int(max_k), max(int(min_k), base_k + int(extra)))
    return ranked[:selected_k], {
        "base_k": int(base_k),
        "selected_k": int(selected_k),
        "extra": int(extra),
        "largest_gap": float(gaps[base_k - 1]),
    }


def adaptive_rag(
    candidates: Sequence[Candidate],
    *,
    label: str,
    budgets: Mapping[str, int],
) -> tuple[list[Candidate], dict[str, str | int]]:
    normalized_label = str(label).strip().upper()
    if normalized_label not in budgets:
        raise ValueError(f"Unknown Adaptive-RAG label: {label}")
    selected_k = int(budgets[normalized_label])
    selected = fixed_top_k(candidates, selected_k)
    return selected, {
        "label": normalized_label,
        "requested_k": selected_k,
        "selected_k": len(selected),
    }
