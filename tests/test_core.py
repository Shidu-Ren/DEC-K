from __future__ import annotations

import numpy as np

from deck import Candidate, calibrated_depth, select_deck, sequential_mmr


def candidate(clip_id: str, relevance: float, vector: list[float]) -> Candidate:
    return Candidate(clip_id, relevance, np.asarray(vector, dtype=np.float32))


def test_mmr_prefers_complementary_candidate() -> None:
    candidates = [
        candidate("a", 1.00, [1.0, 0.0]),
        candidate("b", 0.99, [1.0, 0.0]),
        candidate("c", 0.80, [0.0, 1.0]),
    ]
    selected, trace = sequential_mmr(candidates, steps=2, lambda_=0.40)
    assert [item.clip_id for item in selected] == ["a", "c"]
    assert trace[1].redundancy == 0.0


def test_calibrated_depth_selects_largest_length_calibrated_boundary() -> None:
    selected_k, scores, reason = calibrated_depth(
        [1.0, 0.90, 0.89, 0.30, 0.29],
        min_k=2,
        max_k=4,
        start_k=1,
    )
    assert selected_k == 3
    assert max(scores, key=lambda item: item.score).k == 3
    assert reason == "calibrated_adjacent_drop"


def test_lookahead_is_observed_but_never_returned() -> None:
    candidates = [
        candidate(str(index), 1.0 - index * 0.1, [1.0, float(index), 0.5])
        for index in range(6)
    ]
    result = select_deck(
        candidates,
        candidate_pool=6,
        min_k=2,
        max_k=3,
        start_k=1,
        lambda_=0.85,
    )
    assert len(result.observed) == 4
    assert len(result.selected) <= 3
    assert result.observed[-1].clip_id not in {
        item.clip_id for item in result.selected_in_mmr_order
    }


def test_equal_relevance_returns_maximum_feasible_depth() -> None:
    candidates = [
        candidate(str(index), 0.5, [1.0, float(index)]) for index in range(5)
    ]
    result = select_deck(candidates, min_k=2, max_k=4, start_k=2)
    assert result.selected_k == 4
    assert result.stop_reason == "no_valid_boundary_return_max"
