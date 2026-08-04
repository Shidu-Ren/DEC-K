from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedding_backend import load_or_create_doc_embeddings, get_embedding_client


TIME_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}(?:[,.]\d+)?|\d+(?:\.\d+)?)"
    r"\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:[,.]\d+)?|\d+(?:\.\d+)?)"
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "best",
    "by",
    "correct",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "option",
    "question",
    "select",
    "the",
    "this",
    "to",
    "video",
    "what",
    "which",
    "who",
    "with",
}
ADAPTIVERAG_LABELS = {"A", "B", "C"}
_ADAPTIVERAG_FILE_LABEL_CACHE: dict[str, dict[str, str]] = {}
_ADAPTIVERAG_HF_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


@dataclass
class TimedBlock:
    source: str
    index: int
    text: str
    start: float | None = None
    end: float | None = None
    time_label: str = ""


def parse_time_seconds(value: str) -> float | None:
    value = str(value or "").strip().replace(",", ".")
    if not value:
        return None
    try:
        if ":" not in value:
            return float(value)
        parts = value.split(":")
        if len(parts) != 3:
            return None
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_time_range(start: float | None, end: float | None, fallback: str = "") -> str:
    if start is None or end is None:
        return fallback
    return f"{format_seconds(start)} --> {format_seconds(end)}"


def parse_timed_blocks(text: str, source: str) -> list[TimedBlock]:
    blocks: list[TimedBlock] = []
    current_start: float | None = None
    current_end: float | None = None
    current_label = ""
    current_lines: list[str] = []
    untimed_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_label, current_lines
        body_lines = [line.strip() for line in current_lines if line.strip() and not line.strip().isdigit()]
        body = "\n".join(body_lines).strip()
        if body:
            blocks.append(
                TimedBlock(
                    source=source,
                    index=len(blocks),
                    text=body,
                    start=current_start,
                    end=current_end,
                    time_label=current_label,
                )
            )
        current_start = None
        current_end = None
        current_label = ""
        current_lines = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        match = TIME_RE.match(line)
        if match:
            flush()
            current_start = parse_time_seconds(match.group("start"))
            current_end = parse_time_seconds(match.group("end"))
            current_label = format_time_range(current_start, current_end, line)
            continue
        if current_label:
            current_lines.append(line)
        elif line and not line.isdigit():
            untimed_lines.append(line)
    flush()

    if blocks:
        return blocks
    body = "\n".join(line for line in untimed_lines if line).strip()
    if body:
        return [TimedBlock(source=source, index=0, text=body)]
    return []


def overlaps(left: TimedBlock, right: TimedBlock) -> bool:
    if left.start is None or left.end is None or right.start is None or right.end is None:
        return False
    return left.start < right.end and right.start < left.end


def render_block(block: TimedBlock) -> str:
    label = block.time_label or format_time_range(block.start, block.end)
    if label:
        return f"{label}\n{block.text}"
    return block.text


def build_query(item: dict[str, Any]) -> str:
    options = item.get("options") or []
    option_lines = []
    for idx, option in enumerate(options):
        option_lines.append(f"{chr(ord('A') + idx)}. {option}")
    return "\n".join([str(item.get("question") or "").strip(), *option_lines]).strip()


def build_adaptiverag_query(item: dict[str, Any], query_field: str) -> str:
    if str(query_field or "question") == "question_with_options":
        return build_query(item)
    return str(item.get("question") or "").strip()


def normalize_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def normalize_adaptiverag_label(value: Any, fallback: str | None = "B") -> str:
    if isinstance(value, dict):
        for key in ("prediction", "label", "option", "route_label", "class"):
            if key in value:
                return normalize_adaptiverag_label(value[key], fallback=fallback)

    text = str(value or "").strip()
    upper = text.upper()
    if upper in ADAPTIVERAG_LABELS:
        return upper
    match = re.search(r"\b([ABC])\b", upper)
    if match:
        return match.group(1)

    lowered = upper.lower()
    if any(marker in lowered for marker in ("zero", "no retrieval", "no-retrieval", "none")):
        return "A"
    if any(marker in lowered for marker in ("single", "one-step", "1-step", "one hop", "single-step")):
        return "B"
    if any(marker in lowered for marker in ("multi", "complex", "multi-step", "multihop", "multi-hop")):
        return "C"

    if fallback is None:
        return ""
    return normalize_adaptiverag_label(str(fallback or "B").strip()[:1], fallback="B")


def resolve_model_name_or_path(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return value
    expanded = os.path.expanduser(value)
    if os.path.exists(expanded) or expanded.startswith(("/", ".")):
        return os.path.abspath(expanded)
    return value


def adaptiverag_label_description(label: str) -> str:
    return {
        "A": "zero/no-retrieval",
        "B": "single-step retrieval",
        "C": "multi-step retrieval",
    }.get(normalize_adaptiverag_label(label), "single-step retrieval")


def register_adaptiverag_label(record: dict[str, Any], label_map: dict[str, str], outer_id: str | None = None) -> None:
    label = normalize_adaptiverag_label(record, fallback=None)
    if label not in ADAPTIVERAG_LABELS:
        return
    if outer_id is not None:
        label_map[f"id:{str(outer_id)}"] = label
    for key in ("id", "qid", "question_id", "global_idx", "index"):
        if record.get(key) is not None:
            label_map[f"id:{str(record[key])}"] = label
    question = record.get("question") or record.get("query")
    if question:
        label_map[f"question:{normalize_text_key(question)}"] = label


def load_adaptiverag_label_map(path: str) -> dict[str, str]:
    path = os.path.abspath(os.path.expanduser(str(path or "")))
    if not path:
        return {}
    cached = _ADAPTIVERAG_FILE_LABEL_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adaptive-RAG classifier label file not found: {path}")

    label_map: dict[str, str] = {}
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        register_adaptiverag_label(record, label_map)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    register_adaptiverag_label(record, label_map)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    register_adaptiverag_label(value, label_map, outer_id=str(key))
                else:
                    label = normalize_adaptiverag_label(value, fallback=None)
                    if label in ADAPTIVERAG_LABELS:
                        label_map[f"id:{str(key)}"] = label
                        label_map[f"question:{normalize_text_key(key)}"] = label

    _ADAPTIVERAG_FILE_LABEL_CACHE[path] = label_map
    return label_map


def item_adaptiverag_label_keys(item: dict[str, Any], route_query: str, full_query: str) -> list[str]:
    keys = []
    for key in ("question_id", "qid", "id", "global_idx"):
        if item.get(key) is not None:
            keys.append(f"id:{str(item[key])}")
    keys.append(f"question:{normalize_text_key(route_query)}")
    keys.append(f"question:{normalize_text_key(full_query)}")
    return keys


def load_adaptiverag_hf_classifier(model_path: str, device_arg: str) -> tuple[Any, Any, Any]:
    if not model_path:
        raise ValueError(
            "evidence_selector=adaptiverag with route_source=hf requires --adaptiverag_classifier_path "
            "to point at a trained Adaptive-RAG classifier checkpoint."
        )

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device_arg or "cpu")
    model_path = resolve_model_name_or_path(model_path)
    cache_key = (model_path, device)
    cached = _ADAPTIVERAG_HF_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    cached = (tokenizer, model, torch.device(device))
    _ADAPTIVERAG_HF_MODEL_CACHE[cache_key] = cached
    return cached


def predict_adaptiverag_hf_label(
    route_query: str,
    classifier_path: str,
    classifier_device: str,
    max_length: int,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    tokenizer, model, device = load_adaptiverag_hf_classifier(classifier_path, classifier_device)
    import torch

    inputs = tokenizer(
        route_query,
        return_tensors="pt",
        truncation=True,
        max_length=max(8, int(max_length)),
    ).to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=False,
        )
    scores = generated.scores[0]
    label_token_ids = [tokenizer(label).input_ids[0] for label in ("A", "B", "C")]
    label_scores = torch.stack([scores[:, token_id] for token_id in label_token_ids], dim=0)
    probs = torch.nn.functional.softmax(label_scores, dim=0).detach().cpu()
    pred_idx = int(torch.argmax(probs[:, 0]).item())
    label = ("A", "B", "C")[pred_idx]
    return label, {
        "source": "hf",
        "classifier_path": resolve_model_name_or_path(classifier_path),
        "classifier_device": str(device),
        "label_token_ids": {label: int(token_id) for label, token_id in zip(("A", "B", "C"), label_token_ids)},
        "label_probs": {label: float(probs[idx, 0].item()) for idx, label in enumerate(("A", "B", "C"))},
    }


def build_memory_docs(item: dict[str, Any], memory: str) -> list[dict[str, Any]]:
    captions = parse_timed_blocks(str(item.get("caption") or ""), "caption")
    subtitles = parse_timed_blocks(str(item.get("subtitle") or ""), "subtitle")
    docs: list[dict[str, Any]] = []

    def add_doc(
        doc_id: str,
        clip_idx: int | None,
        start: float | None,
        end: float | None,
        caption_text: str,
        subtitle_text: str,
        text: str,
    ) -> None:
        docs.append(
            {
                "doc_id": doc_id,
                "clip_idx": clip_idx,
                "start": start,
                "end": end,
                "time": format_time_range(start, end),
                "caption": caption_text.strip(),
                "subtitle": subtitle_text.strip(),
                "text": text.strip(),
            }
        )

    if memory == "caption":
        for block in captions:
            add_doc(
                f"caption:{block.index}",
                block.index,
                block.start,
                block.end,
                render_block(block),
                "",
                f"Caption:\n{block.text}",
            )
        return docs

    if memory == "subtitle":
        for block in subtitles:
            add_doc(
                f"subtitle:{block.index}",
                block.index,
                block.start,
                block.end,
                "",
                render_block(block),
                f"Subtitle:\n{block.text}",
            )
        return docs

    if memory != "caption_subtitle":
        raise ValueError(f"Unknown selector memory: {memory}")

    if captions:
        for block in captions:
            matched_subtitles = [sub for sub in subtitles if overlaps(block, sub)]
            subtitle_text = "\n\n".join(render_block(sub) for sub in matched_subtitles)
            text_parts = [f"Caption:\n{block.text}"]
            if subtitle_text:
                text_parts.append(f"Subtitle:\n{subtitle_text}")
            add_doc(
                f"caption:{block.index}",
                block.index,
                block.start,
                block.end,
                render_block(block),
                subtitle_text,
                "\n\n".join(text_parts),
            )
        return docs

    for block in subtitles:
        add_doc(
            f"subtitle:{block.index}",
            block.index,
            block.start,
            block.end,
            "",
            render_block(block),
            f"Subtitle:\n{block.text}",
        )
    return docs


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(str(text).lower()) if tok not in STOPWORDS and len(tok) > 1]


def _normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vec.values()))
    if norm <= 1e-12:
        return vec
    return {key: value / norm for key, value in vec.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def build_tfidf_vectors(texts: list[str], query: str) -> tuple[list[dict[str, float]], dict[str, float]]:
    doc_counts = [Counter(tokenize(text)) for text in texts]
    df: Counter[str] = Counter()
    for counts in doc_counts:
        df.update(counts.keys())
    n_docs = max(1, len(doc_counts))

    def idf(token: str) -> float:
        return math.log((n_docs + 1) / (df.get(token, 0) + 1)) + 1.0

    doc_vectors = []
    for counts in doc_counts:
        total = max(1, sum(counts.values()))
        doc_vectors.append(_normalize({tok: (count / total) * idf(tok) for tok, count in counts.items()}))

    query_counts = Counter(tokenize(query))
    query_total = max(1, sum(query_counts.values()))
    query_vector = _normalize({tok: (count / query_total) * idf(tok) for tok, count in query_counts.items()})
    return doc_vectors, query_vector


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    denom = max(abs(hi - lo), 1e-12)
    return [(float(value) - lo) / denom for value in values]


def _mmr_relevance_scores(scores: list[float]) -> tuple[str, list[float]]:
    mode = os.getenv("SILVR_MMR_RELEVANCE_MODE", "minmax").strip().lower()
    if mode in {"raw", "cosine", "identity", "none", "no_norm", "unnormalized"}:
        return "raw", [float(score) for score in scores]
    return "minmax", minmax(scores)


def relevance_rank(
    docs: list[dict[str, Any]],
    query: str,
) -> tuple[list[dict[str, Any]], list[float], list[dict[str, float]]]:
    vectors, query_vector = build_tfidf_vectors([doc["text"] for doc in docs], query)
    scores = [_dot(vector, query_vector) for vector in vectors]
    order = sorted(range(len(docs)), key=lambda idx: (-scores[idx], docs[idx].get("start") or 0.0, idx))
    ranked_docs = [docs[idx] for idx in order]
    ranked_scores = [float(scores[idx]) for idx in order]
    ranked_vectors = [vectors[idx] for idx in order]
    return ranked_docs, ranked_scores, ranked_vectors


def dense_relevance_rank(
    docs: list[dict[str, Any]],
    query: str,
    item: dict[str, Any],
    args: Any,
) -> tuple[list[dict[str, Any]], list[float], list[np.ndarray], dict[str, Any]]:
    doc_embeddings, cache_meta = load_or_create_doc_embeddings(docs, item, args)
    query_embedding = get_embedding_client(args).embed_texts([query])[0]
    scores_array = np.dot(doc_embeddings, query_embedding.T)
    scores = [float(score) for score in np.atleast_1d(np.squeeze(scores_array)).tolist()]
    order = sorted(range(len(docs)), key=lambda idx: (-scores[idx], docs[idx].get("start") or 0.0, idx))
    ranked_docs = [docs[idx] for idx in order]
    ranked_scores = [scores[idx] for idx in order]
    ranked_vectors = [doc_embeddings[idx] for idx in order]
    return ranked_docs, ranked_scores, ranked_vectors, {
        "backend": "dense",
        "query": query,
        "embedding": cache_meta,
    }


def hybrid_relevance_rank(
    docs: list[dict[str, Any]],
    query: str,
    item: dict[str, Any],
    args: Any,
) -> tuple[list[dict[str, Any]], list[float], list[np.ndarray], dict[str, Any]]:
    dense_docs, dense_scores, dense_vectors, dense_trace = dense_relevance_rank(docs, query, item, args)
    dense_by_id = {doc["doc_id"]: score for doc, score in zip(dense_docs, dense_scores)}
    dense_vec_by_id = {doc["doc_id"]: vec for doc, vec in zip(dense_docs, dense_vectors)}
    tfidf_docs, tfidf_scores, _ = relevance_rank(docs, query)
    tfidf_by_id = {doc["doc_id"]: score for doc, score in zip(tfidf_docs, tfidf_scores)}

    dense_norm = {
        doc_id: score
        for doc_id, score in zip(dense_by_id.keys(), minmax(list(dense_by_id.values())))
    }
    tfidf_norm = {
        doc_id: score
        for doc_id, score in zip(tfidf_by_id.keys(), minmax(list(tfidf_by_id.values())))
    }
    dense_weight = min(1.0, max(0.0, float(getattr(args, "hybrid_dense_weight", 0.7))))
    scores_by_id = {
        doc["doc_id"]: dense_weight * dense_norm.get(doc["doc_id"], 0.0)
        + (1.0 - dense_weight) * tfidf_norm.get(doc["doc_id"], 0.0)
        for doc in docs
    }
    order = sorted(range(len(docs)), key=lambda idx: (-scores_by_id[docs[idx]["doc_id"]], docs[idx].get("start") or 0.0, idx))
    ranked_docs = [docs[idx] for idx in order]
    ranked_scores = [float(scores_by_id[docs[idx]["doc_id"]]) for idx in order]
    ranked_vectors = [dense_vec_by_id[docs[idx]["doc_id"]] for idx in order]
    return ranked_docs, ranked_scores, ranked_vectors, {
        "backend": "hybrid",
        "dense_weight": dense_weight,
        "dense_trace": dense_trace,
    }


def mmr_rerank(
    docs: list[dict[str, Any]],
    scores: list[float],
    vectors: list[Any],
    top_k: int,
    lambda_mult: float,
) -> tuple[list[dict[str, Any]], list[float], list[Any], list[int]]:
    if not docs:
        return [], [], [], []
    top_k = max(1, min(int(top_k), len(docs)))
    lambda_mult = min(1.0, max(0.0, float(lambda_mult)))
    _, relevance_scores = _mmr_relevance_scores(scores)
    selected: list[int] = []
    selected_scores: list[float] = []
    remaining = list(range(len(docs)))
    while remaining and len(selected) < top_k:
        def objective(idx: int) -> float:
            if not selected:
                return float(relevance_scores[idx])
            redundancy = max(_dot(vectors[idx], vectors[sel]) for sel in selected)
            return float(lambda_mult * relevance_scores[idx] - (1.0 - lambda_mult) * redundancy)

        if not selected:
            best = max(remaining, key=lambda idx: (relevance_scores[idx], -idx))
        else:
            best = max(
                remaining,
                key=lambda idx: (
                    objective(idx),
                    relevance_scores[idx],
                    -idx,
                ),
            )
        selected_scores.append(objective(best))
        selected.append(best)
        remaining.remove(best)
    return (
        [docs[idx] for idx in selected],
        selected_scores,
        [vectors[idx] for idx in selected],
        selected,
    )


def select_topk(docs: list[dict[str, Any]], scores: list[float], k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = docs[: max(0, min(int(k), len(docs)))]
    return selected, {"selector": "fixed_topk", "selected_count": len(selected), "qa_top_k": int(k)}


def select_relative_retention(
    docs: list[dict[str, Any]],
    scores: list[float],
    pool_size: int,
    min_chunks: int,
    max_chunks: int,
    selector_name: str = "relret",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_docs = list(docs[:pool_size])
    candidate_scores = [float(score) for score in scores[:pool_size]]
    if not candidate_docs:
        return [], {"selector": selector_name, "selected_count": 0, "reason": "empty_candidates"}
    max_chunks = max(1, min(int(max_chunks), len(candidate_docs), int(pool_size)))
    min_chunks = max(1, min(int(min_chunks), max_chunks))
    eps = float(os.getenv("SILVR_RELRET_EPS", "1e-12"))
    eps = max(eps, 1e-300)
    policy = os.getenv("SILVR_RELRET_POLICY", "relative_retention_survival").lower()

    if policy in {
        "raw_mmr_delta_root",
        "raw_mmr_delta_root_g1",
        "raw_mmr_delta_root_legacy",
        "raw_delta_root",
        "mmr_delta_root",
        "adjacent_delta_root",
    }:
        raw_scores = list(candidate_scores)
        use_g1_normalization = policy in {
            "raw_mmr_delta_root",
            "raw_mmr_delta_root_g1",
        }
        delta_normalizer = "g1" if use_g1_normalization else "none"
        delta_denominator = float(raw_scores[0]) if use_g1_normalization else 1.0
        positive_g1 = delta_denominator > 0.0
        max_decision_k = min(max_chunks, len(candidate_docs) - 1)
        min_decision_k = max(
            1,
            int(os.getenv("SILVR_RELRET_MIN_DECISION_K", "1")),
        )
        delta_root_scores: list[dict[str, Any]] = []
        if len(candidate_docs) <= 1:
            selected_count = len(candidate_docs)
            stop_reason = "single_candidate"
        elif max_decision_k < min_decision_k:
            selected_count = min(max_chunks, len(candidate_docs), max(1, min_chunks))
            stop_reason = "insufficient_boundary_score"
        else:
            for k in range(min_decision_k, max_decision_k + 1):
                current = float(raw_scores[k - 1])
                next_score = float(raw_scores[k])
                raw_delta = current - next_score
                if use_g1_normalization:
                    relative_drop = raw_delta / delta_denominator if positive_g1 else None
                    eligible = relative_drop is not None and relative_drop > 0.0
                    root_score = (
                        float(relative_drop ** (1.0 / float(k))) if eligible else None
                    )
                    delta = raw_delta
                    normalized_delta = relative_drop
                else:
                    delta = max(raw_delta, eps)
                    normalized_delta = delta
                    eligible = True
                    root_score = float(delta ** (1.0 / float(k)))
                delta_root_scores.append(
                    {
                        "k": int(k),
                        "current_score": current,
                        "next_score": next_score,
                        "raw_delta": float(raw_delta),
                        "delta": float(delta),
                        "normalized_delta": normalized_delta,
                        "eligible": bool(eligible),
                        "score": root_score,
                    }
                )
            eligible_scores = [item for item in delta_root_scores if item["eligible"]]
            if eligible_scores:
                best = max(
                    eligible_scores,
                    key=lambda item: (
                        float(item["score"]),
                        float(item["normalized_delta"]),
                        -int(item["k"]),
                    ),
                )
                selected_count = int(best["k"])
                stop_reason = (
                    "raw_mmr_relative_drop_root_map"
                    if use_g1_normalization
                    else "raw_mmr_adjacent_delta_root_map_legacy"
                )
            else:
                selected_count = max_chunks
                stop_reason = (
                    "nonpositive_g1_return_max"
                    if not positive_g1
                    else "no_positive_relative_drop_return_max"
                )
        selected_count = min(max_chunks, len(candidate_docs), max(min_chunks, selected_count))
        return candidate_docs[:selected_count], {
            "selector": selector_name,
            "policy": policy,
            "pool_size": int(pool_size),
            "min_chunks": int(min_chunks),
            "max_chunks": int(max_chunks),
            "min_decision_k": int(min_decision_k),
            "boundary_required": True,
            "max_decision_k": int(max_decision_k),
            "delta_normalizer": delta_normalizer,
            "delta_denominator": float(delta_denominator),
            "positive_g1": bool(positive_g1),
            "selected_count": int(selected_count),
            "stop_reason": stop_reason,
            "raw_scores": raw_scores,
            "delta_root_scores": delta_root_scores,
        }

    smoothed_scores = list(candidate_scores)
    for idx in range(1, len(smoothed_scores)):
        smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])
    normalized_scores = minmax(smoothed_scores)
    n_scores = len(normalized_scores)
    relative_drops: list[dict[str, float | int]] = []
    continue_probs: list[float] = []
    k_distribution: list[float] = []
    length_normalized_scores: list[float] = []
    length_normalized_distribution: list[float] = []
    expected_k = 0.0
    expected_decision: str | None = None

    if n_scores <= 1:
        selected_count = n_scores
        stop_reason = "single_candidate"
    else:
        for idx in range(0, len(normalized_scores) - 1):
            current = float(normalized_scores[idx])
            next_score = float(normalized_scores[idx + 1])
            drop = max(0.0, current - next_score)
            relative_drop = drop / max(abs(current), eps)
            continue_prob = min(1.0, max(0.0, next_score / max(abs(current), eps)))
            continue_probs.append(float(continue_prob))
            relative_drops.append(
                {
                    "after_rank": int(idx + 1),
                    "next_rank": int(idx + 2),
                    "gap": float(drop),
                    "relative_drop": float(relative_drop),
                }
            )

        survival_log = 0.0
        for rank in range(1, n_scores + 1):
            if rank < n_scores:
                stop_prob = 1.0 - float(continue_probs[rank - 1])
                raw_log_prob = survival_log + math.log(max(eps, stop_prob))
            else:
                raw_log_prob = survival_log
            k_distribution.append(float(math.exp(raw_log_prob)))
            length_normalized_scores.append(float(math.exp(raw_log_prob / float(rank))))
            if rank < n_scores:
                survival_log += math.log(max(eps, float(continue_probs[rank - 1])))

        if policy in {
            "relative_retention_ln_expectation",
            "adaptive_relative_retention_ln_expectation",
            "retention_ln_expectation",
        }:
            normalizer = max(float(sum(length_normalized_scores)), eps)
            length_normalized_distribution = [
                float(score / normalizer) for score in length_normalized_scores
            ]
            expected_k = float(
                sum((idx + 1) * prob for idx, prob in enumerate(length_normalized_distribution))
            )
            expected_decision = os.getenv(
                "SILVR_RELRET_LN_EXPECT_DECISION",
                os.getenv("DMMR_RELRET_LN_EXPECT_DECISION", "ceil"),
            ).lower()
            if expected_decision == "round":
                selected_count = int(math.floor(expected_k + 0.5))
            elif expected_decision == "floor":
                selected_count = int(math.floor(expected_k))
            else:
                expected_decision = "ceil"
                selected_count = int(math.ceil(expected_k))
            stop_reason = "relative_retention_length_normalized_expectation"
        else:
            policy = "relative_retention_survival"
            selected_count = int(
                max(
                    range(len(length_normalized_scores)),
                    key=lambda idx: length_normalized_scores[idx],
                )
                + 1
            )
            stop_reason = "relative_retention_length_normalized_map"
    selected_count = min(max_chunks, len(candidate_docs), max(min_chunks, selected_count))
    return candidate_docs[:selected_count], {
        "selector": selector_name,
        "policy": policy,
        "pool_size": int(pool_size),
        "min_chunks": int(min_chunks),
        "max_chunks": int(max_chunks),
        "selected_count": int(selected_count),
        "stop_reason": stop_reason,
        "smoothed_scores": smoothed_scores,
        "normalized_scores": normalized_scores,
        "relative_drops": relative_drops,
        "continue_probs": continue_probs,
        "k_distribution": [
            {"k": int(idx + 1), "prob": float(prob)}
            for idx, prob in enumerate(k_distribution)
        ],
        "expected_k": float(expected_k),
        "expected_decision": expected_decision,
        "length_normalized_distribution": [
            {"k": int(idx + 1), "prob": float(prob)}
            for idx, prob in enumerate(length_normalized_distribution)
        ],
        "length_normalized_scores": [
            {"k": int(idx + 1), "score": float(score)}
            for idx, score in enumerate(length_normalized_scores)
        ],
    }


def select_adaptivek(
    docs: list[dict[str, Any]],
    scores: list[float],
    pool_size: int,
    min_chunks: int,
    max_chunks: int,
    retrieve_more: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_docs = list(docs[:pool_size])
    candidate_scores = [float(score) for score in scores[:pool_size]]
    if not candidate_docs:
        return [], {"selector": "adaptivek", "selected_count": 0, "reason": "empty_candidates"}
    max_chunks = max(1, min(int(max_chunks), len(candidate_docs), int(pool_size)))
    min_chunks = max(0, min(int(min_chunks), max_chunks))
    smoothed_scores = list(candidate_scores)
    for idx in range(1, len(smoothed_scores)):
        smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])
    normalized_scores = minmax(smoothed_scores)
    if len(normalized_scores) <= 1:
        selected_count = len(normalized_scores)
        best_gap = None
    else:
        gaps = [
            {
                "after_rank": idx + 1,
                "next_rank": idx + 2,
                "gap": max(0.0, float(normalized_scores[idx]) - float(normalized_scores[idx + 1])),
            }
            for idx in range(len(normalized_scores) - 1)
        ]
        best_gap = max(gaps, key=lambda item: item["gap"])
        selected_count = int(best_gap["after_rank"]) + max(0, int(retrieve_more))
    selected_count = min(max_chunks, len(candidate_docs), max(min_chunks, selected_count))
    return candidate_docs[:selected_count], {
        "selector": "adaptivek",
        "pool_size": int(pool_size),
        "min_chunks": int(min_chunks),
        "max_chunks": int(max_chunks),
        "retrieve_more": int(retrieve_more),
        "selected_count": int(selected_count),
        "smoothed_scores": smoothed_scores,
        "normalized_scores": normalized_scores,
        "best_gap": best_gap,
    }


def adaptive_rag_label(query: str) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip().lower()
    complex_markers = [
        "after",
        "all ",
        "before",
        "both",
        "compare",
        "count",
        "difference",
        "differences",
        "each",
        "final",
        "first",
        "how many",
        "last",
        "multiple",
        "number of",
        "relationship",
        "sequence",
        "several",
        "then",
        "three",
        "why",
    ]
    return "C" if any(marker in text for marker in complex_markers) else "B"


def predict_adaptiverag_label(item: dict[str, Any], full_query: str, args: Any, route_source: str) -> tuple[str, dict[str, Any]]:
    route_source = str(route_source or "hf").strip().lower()
    route_query = build_adaptiverag_query(item, args.adaptiverag_query_field)
    fallback_label = normalize_adaptiverag_label(args.adaptiverag_fallback_label)

    if route_source in {"file", "classifier_file", "precomputed", "official_file", "official_predictions"}:
        if not args.adaptiverag_classifier_path:
            raise ValueError(
                "evidence_selector=adaptiverag with route_source=file requires "
                "--adaptiverag_classifier_path pointing to official Adaptive-RAG A/B/C predictions."
            )
        label_map = load_adaptiverag_label_map(args.adaptiverag_classifier_path)
        for key in item_adaptiverag_label_keys(item, route_query, full_query):
            if key in label_map:
                return label_map[key], {
                    "source": "official_file" if route_source.startswith("official") else route_source,
                    "classifier_path": os.path.abspath(os.path.expanduser(str(args.adaptiverag_classifier_path))),
                    "matched_key": key,
                    "query_field": args.adaptiverag_query_field,
                }
        return fallback_label, {
            "source": "official_file" if route_source.startswith("official") else route_source,
            "classifier_path": os.path.abspath(os.path.expanduser(str(args.adaptiverag_classifier_path))),
            "matched_key": None,
            "query_field": args.adaptiverag_query_field,
            "reason": "classifier_file_miss",
            "fallback_label": fallback_label,
        }

    if route_source in {"hf", "checkpoint", "model", "classifier", "official_hf", "official_checkpoint"}:
        label, meta = predict_adaptiverag_hf_label(
            route_query,
            args.adaptiverag_classifier_path,
            args.adaptiverag_classifier_device,
            args.adaptiverag_classifier_max_length,
            args.adaptiverag_classifier_max_new_tokens,
        )
        meta["query_field"] = args.adaptiverag_query_field
        if route_source.startswith("official"):
            meta["source"] = "official_hf"
        return label, meta

    if route_source in {"constant", "fixed"}:
        return fallback_label, {
            "source": route_source,
            "raw_output": fallback_label,
            "query_field": args.adaptiverag_query_field,
        }

    if route_source != "heuristic":
        raise ValueError(
            f"Unknown Adaptive-RAG route source: {route_source}. "
            "Use hf, file, precomputed, heuristic, or constant."
        )
    label = adaptive_rag_label(route_query)
    return label, {
        "source": "heuristic",
        "raw_output": label,
        "query_field": args.adaptiverag_query_field,
    }


def select_adaptiverag(
    docs: list[dict[str, Any]],
    scores: list[float],
    item: dict[str, Any],
    query: str,
    pool_size: int,
    args: Any,
    force_route_source: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_docs = list(docs[:pool_size])
    route_source = force_route_source or args.adaptiverag_route_source
    label, route_meta = predict_adaptiverag_label(item, query, args, route_source)
    label_targets = {
        "A": int(getattr(args, "adaptiverag_a_chunks", None) if getattr(args, "adaptiverag_a_chunks", None) is not None else 0),
        "B": int(
            getattr(args, "adaptiverag_b_chunks", None)
            if getattr(args, "adaptiverag_b_chunks", None) is not None
            else args.adaptiverag_single_chunks
        ),
        "C": int(
            getattr(args, "adaptiverag_c_chunks", None)
            if getattr(args, "adaptiverag_c_chunks", None) is not None
            else args.adaptiverag_multi_chunks
        ),
    }
    if label == "A":
        target = label_targets["A"]
    elif label == "C":
        target = label_targets["C"]
    else:
        target = label_targets["B"]
    selected_count = max(0, min(target, int(args.adaptiverag_max_chunks), len(candidate_docs)))
    selector_name = "adaptiverag_heuristic" if route_meta.get("source") == "heuristic" else "adaptiverag_official"
    return candidate_docs[:selected_count], {
        "selector": selector_name,
        "route_label": label,
        "route_description": adaptiverag_label_description(label),
        "route_source": route_meta.get("source"),
        "route_meta": route_meta,
        "pool_size": int(pool_size),
        "single_chunks": int(args.adaptiverag_single_chunks),
        "multi_chunks": int(args.adaptiverag_multi_chunks),
        "max_chunks": int(args.adaptiverag_max_chunks),
        "label_chunk_targets": label_targets,
        "selected_count": int(selected_count),
        "source_repo": "https://github.com/starsuzi/Adaptive-RAG",
    }


def select_evidence(item: dict[str, Any], args: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs = build_memory_docs(item, args.selector_memory)
    query = build_query(item)
    backend = str(getattr(args, "retrieval_backend", "tfidf") or "tfidf").strip().lower()
    if not docs:
        return [], {
            "selector": args.evidence_selector,
            "memory": args.selector_memory,
            "backend": backend,
            "selected_count": 0,
            "reason": "empty_memory",
        }

    backend_trace: dict[str, Any] = {"backend": "local_tfidf"}
    if backend == "tfidf":
        ranked_docs, ranked_scores, ranked_vectors = relevance_rank(docs, query)
    elif backend == "dense":
        ranked_docs, ranked_scores, ranked_vectors, backend_trace = dense_relevance_rank(docs, query, item, args)
    elif backend == "hybrid":
        ranked_docs, ranked_scores, ranked_vectors, backend_trace = hybrid_relevance_rank(docs, query, item, args)
    else:
        raise ValueError(f"Unknown retrieval backend: {backend}")

    retrieval_top_k = max(1, min(int(args.retrieval_top_k), len(ranked_docs)))
    ranked_docs = ranked_docs[:retrieval_top_k]
    ranked_scores = ranked_scores[:retrieval_top_k]
    ranked_vectors = ranked_vectors[:retrieval_top_k]
    selector = args.evidence_selector

    mmr_trace: dict[str, Any] = {}
    if selector == "mmr_relret":
        mmr_relevance_mode, _ = _mmr_relevance_scores(ranked_scores)
        ranked_docs, ranked_scores, ranked_vectors, mmr_indices = mmr_rerank(
            ranked_docs,
            ranked_scores,
            ranked_vectors,
            args.selector_pool_size,
            args.mmr_lambda,
        )
        mmr_trace = {
            "mmr_lambda": float(args.mmr_lambda),
            "mmr_relevance_mode": mmr_relevance_mode,
            "mmr_selected_from_retrieved_indices": [int(idx) for idx in mmr_indices],
        }
        selected, selector_trace = select_relative_retention(
            ranked_docs,
            ranked_scores,
            args.selector_pool_size,
            args.relret_min_chunks,
            args.relret_max_chunks,
            selector_name="mmr_relret",
        )
    elif selector == "mmr_fixed_topk":
        mmr_relevance_mode, _ = _mmr_relevance_scores(ranked_scores)
        ranked_docs, ranked_scores, ranked_vectors, mmr_indices = mmr_rerank(
            ranked_docs,
            ranked_scores,
            ranked_vectors,
            args.selector_pool_size,
            args.mmr_lambda,
        )
        mmr_trace = {
            "mmr_lambda": float(args.mmr_lambda),
            "mmr_relevance_mode": mmr_relevance_mode,
            "mmr_selected_from_retrieved_indices": [int(idx) for idx in mmr_indices],
        }
        selected, selector_trace = select_topk(ranked_docs, ranked_scores, args.qa_top_k)
        selector_trace["selector"] = "mmr_fixed_topk"
    elif selector == "relret":
        selected, selector_trace = select_relative_retention(
            ranked_docs,
            ranked_scores,
            args.selector_pool_size,
            args.relret_min_chunks,
            args.relret_max_chunks,
            selector_name="relret",
        )
    elif selector == "fixed_topk":
        selected, selector_trace = select_topk(ranked_docs, ranked_scores, args.qa_top_k)
    elif selector == "adaptivek":
        selected, selector_trace = select_adaptivek(
            ranked_docs,
            ranked_scores,
            args.selector_pool_size,
            args.adaptivek_min_chunks,
            args.adaptivek_max_chunks,
            args.adaptivek_retrieve_more,
        )
    elif selector == "adaptiverag":
        selected, selector_trace = select_adaptiverag(
            ranked_docs,
            ranked_scores,
            item,
            query,
            args.selector_pool_size,
            args,
        )
    elif selector == "adaptiverag_heuristic":
        selected, selector_trace = select_adaptiverag(
            ranked_docs,
            ranked_scores,
            item,
            query,
            args.selector_pool_size,
            args,
            force_route_source="heuristic",
        )
    else:
        raise ValueError(f"Unknown evidence selector: {selector}")

    score_by_id = {doc["doc_id"]: score for doc, score in zip(ranked_docs, ranked_scores)}
    for rank, doc in enumerate(selected, start=1):
        doc["rank"] = rank
        doc["score"] = float(score_by_id.get(doc["doc_id"], 0.0))
    trace = {
        "selector": selector,
        "memory": args.selector_memory,
        "backend": backend_trace.get("backend", backend),
        "backend_trace": backend_trace,
        "total_docs": len(docs),
        "retrieval_top_k": int(retrieval_top_k),
        "pool_size": int(args.selector_pool_size),
        "selected_count": len(selected),
        "selected_doc_ids": [doc["doc_id"] for doc in selected],
        "selected_clip_indices": [doc.get("clip_idx") for doc in selected],
        "selector_trace": selector_trace,
    }
    if mmr_trace:
        trace.update(mmr_trace)
    return selected, trace


def _dedupe_blocks(texts: list[str]) -> list[str]:
    seen = set()
    result = []
    for text in texts:
        cleaned = str(text or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def apply_evidence_selector(item: dict[str, Any], args: Any) -> dict[str, Any]:
    if args.evidence_selector == "none":
        return item

    selected, trace = select_evidence(item, args)
    updated = copy.deepcopy(item)
    selected_captions = _dedupe_blocks([doc.get("caption", "") for doc in selected])
    selected_subtitles = _dedupe_blocks([doc.get("subtitle", "") for doc in selected])

    if args.selector_memory in {"caption", "caption_subtitle"}:
        updated["caption"] = "\n\n".join(selected_captions)
    elif not args.selector_keep_full_caption:
        updated["caption"] = ""

    if args.selector_memory in {"subtitle", "caption_subtitle"}:
        updated["subtitle"] = "\n\n".join(selected_subtitles)
    elif not args.selector_keep_full_subtitle:
        updated["subtitle"] = ""

    updated["evidence_selector"] = args.evidence_selector
    updated["selector_memory"] = args.selector_memory
    updated["retrieval_trace"] = trace
    updated["selected_evidence"] = [
        {
            "rank": doc.get("rank"),
            "doc_id": doc.get("doc_id"),
            "clip_idx": doc.get("clip_idx"),
            "time": doc.get("time"),
            "score": doc.get("score"),
            "text": doc.get("text"),
        }
        for doc in selected
    ]
    return updated
