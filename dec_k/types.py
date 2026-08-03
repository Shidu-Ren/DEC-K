from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Candidate:
    """A clip-level retrieval candidate exposed by a host memory system."""

    clip_id: str
    relevance: float
    vector: np.ndarray
    text: str = ""
    start: float | None = None
    end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MMRStep:
    rank: int
    clip_id: str
    relevance: float
    redundancy: float
    marginal_score: float


@dataclass
class DepthScore:
    k: int
    current_score: float
    next_score: float
    drop: float
    normalized_drop: float
    score: float


@dataclass
class SelectionResult:
    selected: list[Candidate]
    selected_in_mmr_order: list[Candidate]
    observed: list[Candidate]
    mmr_steps: list[MMRStep]
    depth_scores: list[DepthScore]
    selected_k: int
    stop_reason: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_k": self.selected_k,
            "stop_reason": self.stop_reason,
            "selected_clip_ids": [item.clip_id for item in self.selected],
            "selected_mmr_order": [item.clip_id for item in self.selected_in_mmr_order],
            "observed_clip_ids": [item.clip_id for item in self.observed],
            "mmr_steps": [step.__dict__ for step in self.mmr_steps],
            "depth_scores": [score.__dict__ for score in self.depth_scores],
            "config": self.config,
        }
