from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .api import ChatClient
from .embeddings import EmbeddingClient, cosine_relevance

DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the retrieved video-memory evidence. "
    "If options are provided, return the option letter and a concise answer."
)


JUDGE_PROMPT = """You are provided with a question, a ground truth answer, and an answer
from an agent model. Determine whether the ground truth answer can be logically inferred
from the agent answer in the context of the question. Semantic equivalence is sufficient.
Return only Yes or No.

Question: {question}
Ground truth answer: {ground_truth}
Agent answer: {prediction}

Output:"""


def embed_record(
    record: dict[str, Any],
    client: EmbeddingClient,
    *,
    query_field: str = "question",
    documents_field: str = "documents",
    batch_size: int = 32,
) -> dict[str, Any]:
    question = str(record.get(query_field) or record.get("query") or "")
    documents = list(record.get(documents_field) or [])
    if not question:
        raise ValueError(f"Missing query field {query_field!r}")
    if not documents:
        raise ValueError(f"Missing documents field {documents_field!r}")

    query_vector = client.encode_queries([question])[0]
    vectors: list[np.ndarray] = []
    size = max(1, int(batch_size))
    for start in range(0, len(documents), size):
        texts = [str(item.get("text") or "") for item in documents[start : start + size]]
        vectors.extend(client.encode_documents(texts))
    matrix = np.asarray(vectors, dtype=np.float32)
    scores = cosine_relevance(query_vector, matrix)
    candidates = []
    for document, vector, score in zip(documents, matrix, scores, strict=True):
        value = dict(document)
        value["clip_id"] = str(value.get("clip_id", len(candidates)))
        value["relevance"] = float(score)
        value["vector"] = vector.tolist()
        candidates.append(value)
    return {
        **{key: value for key, value in record.items() if key != documents_field},
        "candidates": candidates,
    }


def format_evidence(record: dict[str, Any]) -> str:
    blocks = []
    for item in record.get("selected_evidence") or []:
        interval = ""
        if item.get("start") is not None:
            interval = f" | {float(item['start']):.1f}s"
            if item.get("end") is not None:
                interval += f"-{float(item['end']):.1f}s"
        blocks.append(f"[Clip {item.get('clip_id')}{interval}]\n{item.get('text', '')}")
    return "\n\n".join(blocks)


def answer_record(
    record: dict[str, Any],
    client: ChatClient,
    *,
    system_prompt: str = DEFAULT_ANSWER_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict[str, Any]:
    question = str(record.get("question") or record.get("query") or "")
    options = record.get("options")
    option_text = ""
    if isinstance(options, dict):
        option_text = "\n" + "\n".join(f"{key}. {value}" for key, value in options.items())
    elif isinstance(options, list):
        option_text = "\n" + "\n".join(
            f"{chr(65 + index)}. {value}" for index, value in enumerate(options)
        )
    user_prompt = (
        f"Question: {question}{option_text}\n\n"
        f"Retrieved evidence:\n{format_evidence(record)}\n\nAnswer:"
    )
    prediction, usage = client.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {**record, "prediction": prediction.strip(), "answer_usage": usage}


def normalize_answer(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def extract_option(value: Any) -> str:
    import re

    text = str(value or "").strip().upper()
    patterns = [r"^\s*\(?([A-D])\)?(?:[.\s:]|$)", r"(?:ANSWER|OPTION)\s*[:=]?\s*\(?([A-D])\)?"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text if text in {"A", "B", "C", "D"} else ""


def evaluate_records(
    records: Iterable[dict[str, Any]],
    *,
    mode: str,
    judge_client: ChatClient | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated = []
    correct = 0
    for record in records:
        prediction = record.get("prediction", record.get("answer"))
        ground_truth = record.get("ground_truth", record.get("ground_truth_answer"))
        if mode == "multiple_choice":
            is_correct = extract_option(prediction) == extract_option(ground_truth)
        elif mode == "exact":
            is_correct = normalize_answer(prediction) == normalize_answer(ground_truth)
        elif mode == "judge":
            if judge_client is None:
                raise ValueError("judge_client is required for mode=judge")
            response, usage = judge_client.complete(
                [
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=record.get("question", ""),
                            ground_truth=ground_truth,
                            prediction=prediction,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=4,
            )
            is_correct = response.strip().lower().startswith("yes")
            record = {**record, "judge_response": response.strip(), "judge_usage": usage}
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        correct += int(is_correct)
        evaluated.append({**record, "correct": bool(is_correct)})

    total = len(evaluated)
    selected_counts = [int(item.get("selected_k", 0)) for item in evaluated]
    metrics = {
        "mode": mode,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "average_selected_clips": (
            sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
        ),
    }
    return evaluated, metrics
