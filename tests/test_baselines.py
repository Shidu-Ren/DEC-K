from __future__ import annotations

import numpy as np

from deck import Candidate, adaptive_k


def test_adaptive_k_finds_gap_before_applying_buffer() -> None:
    candidates = [
        Candidate(str(index), score, np.asarray([1.0, index], dtype=np.float32))
        for index, score in enumerate([1.0, 0.9, 0.2, 0.19, 0.18, 0.17])
    ]
    selected, trace = adaptive_k(candidates, min_k=2, max_k=5, extra=2)
    assert trace["base_k"] == 2
    assert trace["selected_k"] == 4
    assert len(selected) == 4
