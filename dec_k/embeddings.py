from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .api import _endpoint, post_json
from .core import normalize_vector


class EmbeddingClient(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class OpenAICompatibleEmbeddingClient:
    model: str
    base_url: str
    api_key: str = ""
    timeout: float = 120.0
    retries: int = 3

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        result = post_json(
            _endpoint(self.base_url, "embeddings"),
            {"model": self.model, "input": list(texts)},
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
        )
        try:
            ordered = sorted(result["data"], key=lambda item: int(item["index"]))
            vectors = [normalize_vector(item["embedding"]) for item in ordered]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embedding response: {result}") from exc
        return np.stack(vectors, axis=0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEmbeddingClient:
    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        query_prompt_name: str | None = "query",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'Install model dependencies with `pip install -e ".[models]"`'
            ) from exc
        self.model = SentenceTransformer(
            model,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self.query_prompt_name = query_prompt_name

    def _encode(self, texts: Sequence[str], *, query: bool) -> np.ndarray:
        kwargs = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if query and self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        try:
            value = self.model.encode(list(texts), **kwargs)
        except (KeyError, ValueError):
            kwargs.pop("prompt_name", None)
            value = self.model.encode(list(texts), **kwargs)
        return np.asarray(value, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=True)


def cosine_relevance(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    query_vector = normalize_vector(query)
    document_vectors = np.asarray(
        [normalize_vector(item) for item in np.asarray(documents)], dtype=np.float32
    )
    return document_vectors @ query_vector
