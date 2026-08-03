from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StructuredMemoryConfig:
    enabled: bool = False
    top_nodes: int = 8
    max_leaders: int = 4
    type_window: int = 3
    cluster_threshold: float = 0.85
    eta: float = 1e-3
    leader_weights: tuple[float, ...] = (1.0, 0.35, 0.15, 0.05)
    type_bonus: float = 0.05
    coverage_weight: float = 0.03
    coverage_cap: int = 3

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> StructuredMemoryConfig:
        data = dict(value or {})
        if "leader_weights" in data:
            data["leader_weights"] = tuple(float(item) for item in data["leader_weights"])
        return cls(**data)


@dataclass
class ExperimentConfig:
    name: str = "deck"
    method: str = "deck"
    candidate_pool: int = 200
    min_k: int = 2
    max_k: int = 8
    start_k: int = 2
    lambda_: float = 0.85
    temporal_output: bool = True
    fixed_k: int | None = None
    adaptive_extra: int = 0
    adaptive_budgets: dict[str, int] = field(
        default_factory=lambda: {"A": 2, "B": 7, "C": 8}
    )
    adaptive_label_field: str = "adaptive_rag_label"
    structured_memory: StructuredMemoryConfig = field(
        default_factory=StructuredMemoryConfig
    )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ExperimentConfig:
        data = dict(value)
        if "lambda" in data:
            data["lambda_"] = data.pop("lambda")
        data["structured_memory"] = StructuredMemoryConfig.from_mapping(
            data.get("structured_memory")
        )
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration must be a mapping: {path}")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        allowed = {
            "deck",
            "fixed",
            "mmr_fixed",
            "adaptive_k",
            "adaptive_rag",
            "relevance_deck",
        }
        if self.method not in allowed:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {sorted(allowed)}")
        if self.candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if self.min_k <= 0 or self.max_k < self.min_k:
            raise ValueError("Expected 0 < min_k <= max_k")
        if not 1 <= self.start_k <= self.max_k:
            raise ValueError("start_k must be between 1 and max_k")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda must be in [0, 1]")
        if self.method in {"fixed", "mmr_fixed"} and self.fixed_k is None:
            raise ValueError(f"fixed_k is required for method={self.method}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lambda"] = data.pop("lambda_")
        return data
