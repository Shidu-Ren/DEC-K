from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core import select_deck
from ..types import Candidate


def select_silvr_documents(
    docs: Sequence[dict[str, Any]],
    scores: Sequence[float],
    vectors: Sequence[np.ndarray],
    *,
    candidate_pool: int = 200,
    min_k: int = 2,
    max_k: int = 8,
    start_k: int = 2,
    lambda_: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply DEC-K to a SiLVR-style document, score, and vector triple."""

    if not (len(docs) == len(scores) == len(vectors)):
        raise ValueError("docs, scores, and vectors must have equal length")
    candidates = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, (doc, score, vector) in enumerate(
        zip(docs, scores, vectors, strict=True)
    ):
        clip_id = str(doc.get("doc_id", doc.get("clip_id", index)))
        by_id[clip_id] = doc
        candidates.append(
            Candidate(
                clip_id=clip_id,
                relevance=float(score),
                vector=np.asarray(vector, dtype=np.float32),
                text=str(doc.get("text") or ""),
                start=doc.get("start"),
                end=doc.get("end"),
                metadata={"source_index": index},
            )
        )
    result = select_deck(
        candidates,
        candidate_pool=candidate_pool,
        min_k=min_k,
        max_k=max_k,
        start_k=start_k,
        lambda_=lambda_,
        temporal_output=True,
    )
    return [by_id[item.clip_id] for item in result.selected], result.to_dict()
