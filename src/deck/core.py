from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import Candidate, DepthScore, MMRStep, SelectionResult


def normalize_vector(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array
    return array / norm


def minmax_scale(values: Sequence[float]) -> np.ndarray:
    """Scale relevance to [0, 1] before combining it with cosine redundancy."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    lower = float(array.min())
    upper = float(array.max())
    denominator = max(upper - lower, 1e-12)
    return (array - lower) / denominator


def sequential_mmr(
    candidates: Sequence[Candidate],
    *,
    steps: int,
    lambda_: float,
) -> tuple[list[Candidate], list[MMRStep]]:
    """Construct a prefix with the set-aware MMR objective used by DEC-K."""

    if not candidates or steps <= 0:
        return [], []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    count = min(int(steps), len(candidates))
    relevance = minmax_scale([item.relevance for item in candidates])
    vectors = [normalize_vector(item.vector) for item in candidates]
    selected_indices: list[int] = []
    selected: list[Candidate] = []
    trace: list[MMRStep] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < count:
        def score(index: int) -> tuple[float, float, int, float]:
            if not selected_indices:
                redundancy = 0.0
                marginal = float(relevance[index])
            else:
                redundancy = max(
                    float(np.dot(vectors[index], vectors[chosen]))
                    for chosen in selected_indices
                )
                marginal = float(
                    lambda_ * relevance[index] - (1.0 - lambda_) * redundancy
                )
            return marginal, float(relevance[index]), -index, redundancy

        chosen = max(remaining, key=lambda index: score(index)[:3])
        marginal, rel, _, redundancy = score(chosen)
        selected_indices.append(chosen)
        selected.append(candidates[chosen])
        remaining.remove(chosen)
        trace.append(
            MMRStep(
                rank=len(selected),
                clip_id=candidates[chosen].clip_id,
                relevance=rel,
                redundancy=redundancy,
                marginal_score=marginal,
            )
        )
    return selected, trace


def calibrated_depth(
    marginal_scores: Sequence[float],
    *,
    min_k: int,
    max_k: int,
    start_k: int,
) -> tuple[int, list[DepthScore], str]:
    """Choose K from length-calibrated adjacent drops in an MMR trajectory."""

    scores = [float(value) for value in marginal_scores]
    if not scores:
        return 0, [], "empty_candidates"

    max_return = min(max(1, int(max_k)), len(scores))
    min_return = min(max_return, max(1, int(min_k)))
    max_boundary = min(max_return, len(scores) - 1)
    first_boundary = max(1, int(start_k))
    g1 = scores[0]

    if g1 <= 0.0 or max_boundary < first_boundary:
        return max_return, [], "no_valid_boundary_return_max"

    depth_scores: list[DepthScore] = []
    for k in range(first_boundary, max_boundary + 1):
        current = scores[k - 1]
        next_score = scores[k]
        drop = current - next_score
        if drop <= 0.0:
            continue
        normalized_drop = drop / g1
        score = normalized_drop ** (1.0 / float(k))
        depth_scores.append(
            DepthScore(
                k=k,
                current_score=current,
                next_score=next_score,
                drop=drop,
                normalized_drop=normalized_drop,
                score=score,
            )
        )

    if not depth_scores:
        return max_return, [], "no_positive_drop_return_max"

    best = max(
        depth_scores,
        key=lambda item: (item.score, item.normalized_drop, -item.k),
    )
    selected_k = min(max_return, max(min_return, best.k))
    return selected_k, depth_scores, "calibrated_adjacent_drop"


def select_deck(
    candidates: Sequence[Candidate],
    *,
    candidate_pool: int = 200,
    min_k: int = 2,
    max_k: int = 8,
    start_k: int = 2,
    lambda_: float = 0.85,
    temporal_output: bool = True,
) -> SelectionResult:
    """Run the complete DEC-K selector for one query."""

    pool = sorted(
        candidates,
        key=lambda item: (
            -float(item.relevance),
            float("inf") if item.start is None else float(item.start),
            item.clip_id,
        ),
    )[: max(0, int(candidate_pool))]

    if not pool:
        return SelectionResult([], [], [], [], [], 0, "empty_candidates", {})

    # One extra MMR step is observed to score the K_max boundary.
    observed_steps = min(len(pool), max(1, int(max_k)) + 1)
    observed, mmr_steps = sequential_mmr(pool, steps=observed_steps, lambda_=lambda_)
    selected_k, depth_scores, stop_reason = calibrated_depth(
        [step.marginal_score for step in mmr_steps],
        min_k=min_k,
        max_k=max_k,
        start_k=start_k,
    )
    selected_mmr = observed[:selected_k]
    selected = list(selected_mmr)
    if temporal_output:
        selected.sort(
            key=lambda item: (
                float("inf") if item.start is None else float(item.start),
                item.clip_id,
            )
        )

    config = {
        "candidate_pool": int(candidate_pool),
        "min_k": int(min_k),
        "max_k": int(max_k),
        "start_k": int(start_k),
        "lambda": float(lambda_),
        "temporal_output": bool(temporal_output),
        "lookahead_steps": int(observed_steps),
    }
    return SelectionResult(
        selected=selected,
        selected_in_mmr_order=selected_mmr,
        observed=observed,
        mmr_steps=mmr_steps,
        depth_scores=depth_scores,
        selected_k=selected_k,
        stop_reason=stop_reason,
        config=config,
    )
