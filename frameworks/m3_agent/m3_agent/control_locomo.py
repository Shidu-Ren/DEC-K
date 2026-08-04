# Copyright (2026)

import argparse
import json
import math
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Set

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "models/M3-Agent-Control"
TOKENIZER = None

SYSTEM_PROMPT = (
    "You are given a question and some relevant knowledge. Your task is to reason "
    "about whether the provided knowledge is sufficient to answer the question. "
    "If it is sufficient, output [Answer] followed by the answer. "
    "If it is not sufficient, output [Search] and generate a query that will help "
    "retrieve additional information from a memory bank.\n\n"
    "Question: {question}"
)

INSTRUCTION = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query
that will help retrieve the missing information. The search {content} needs to be
different from the previous.
If the answer can be derived from the provided knowledge, the {content} should be
the concise final answer only.
"""

PATTERN = r"Action: \[(.*)\].*Content: (.*)"


def normalize_answer(text: str) -> str:
    text = str(text).replace(",", "")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_answer(text))


def f1_score_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_score_multi(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in str(prediction).split(",") if p.strip()]
    ground_truths = [g.strip() for g in str(ground_truth).split(",") if g.strip()]
    if not predictions or not ground_truths:
        return 0.0
    return sum(
        max(f1_score_single(pred, gold) for pred in predictions)
        for gold in ground_truths
    ) / len(ground_truths)


def evaluate_locomo_answer(prediction: str, answer: str, category: int) -> float:
    prediction = str(prediction or "").strip()
    answer = str(answer or "").strip()
    if not prediction:
        return 0.0

    if category in [2, 3, 4]:
        return f1_score_single(prediction, answer)
    if category == 1:
        return f1_score_multi(prediction, answer)
    if category == 5:
        lowered = prediction.lower()
        return 1.0 if ("no information available" in lowered or "not mentioned" in lowered) else 0.0
    return f1_score_single(prediction, answer)


def session_numbers(conversation: dict) -> list[int]:
    nums = []
    for key in conversation:
        match = re.fullmatch(r"session_(\d+)", key)
        if match:
            nums.append(int(match.group(1)))
    return sorted(nums)


def build_memory_bank(sample: dict, include_dialogue: bool = True, include_observations: bool = True, include_summaries: bool = True) -> list[dict]:
    conv = sample["conversation"]
    docs = []

    for sess in session_numbers(conv):
        session_key = f"session_{sess}"
        date_key = f"session_{sess}_date_time"
        date_time = conv.get(date_key, "")

        if include_dialogue:
            for turn in conv.get(session_key, []):
                text = f"[{date_time}] {turn['speaker']}: {turn['text']}"
                if turn.get("blip_caption"):
                    text += f" [Image: {turn['blip_caption']}]"
                docs.append(
                    {
                        "doc_id": turn["dia_id"],
                        "text": text,
                        "source": "dialogue",
                        "session": sess,
                    }
                )

        if include_observations:
            obs_key = f"session_{sess}_observation"
            obs = sample.get("observation", {}).get(obs_key, {})
            if isinstance(obs, dict):
                for speaker, items in obs.items():
                    if not isinstance(items, list):
                        continue
                    for idx, item in enumerate(items, start=1):
                        if not item:
                            continue
                        obs_text = item[0] if isinstance(item, list) and item else str(item)
                        evidence = item[1] if isinstance(item, list) and len(item) > 1 else ""
                        docs.append(
                            {
                                "doc_id": f"O{sess}:{speaker}:{idx}",
                                "text": f"[{date_time}] Observation about {speaker}: {obs_text}",
                                "source": "observation",
                                "session": sess,
                                "evidence": evidence,
                            }
                        )

        if include_summaries:
            sum_key = f"session_{sess}_summary"
            summary = sample.get("session_summary", {}).get(sum_key)
            if summary:
                docs.append(
                    {
                        "doc_id": f"S{sess}",
                        "text": f"[{date_time}] Session summary: {summary}",
                        "source": "summary",
                        "session": sess,
                    }
                )

    return docs


class BM25Index:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.doc_tokens = []
        self.doc_freq = defaultdict(int)
        self.avgdl = 0.0

        total_len = 0
        for doc in docs:
            tokens = tokenize(doc["text"])
            counts = Counter(tokens)
            self.doc_tokens.append(counts)
            total_len += sum(counts.values())
            for token in counts:
                self.doc_freq[token] += 1
        self.avgdl = (total_len / len(docs)) if docs else 0.0

    def score(self, query: str, doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        query_terms = tokenize(query)
        if not query_terms or not self.docs:
            return 0.0
        counts = self.doc_tokens[doc_idx]
        dl = sum(counts.values()) or 1
        score = 0.0
        n_docs = len(self.docs)
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * dl / (self.avgdl or 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def _candidate_embeddings(self, candidate_indices: list[int]) -> np.ndarray:
        vocab = {}
        for idx in candidate_indices:
            for token in self.doc_tokens[idx]:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if not vocab:
            return np.zeros((len(candidate_indices), 1), dtype=float)

        matrix = np.zeros((len(candidate_indices), len(vocab)), dtype=float)
        for row_idx, idx in enumerate(candidate_indices):
            for token, value in self.doc_tokens[idx].items():
                if token in vocab:
                    matrix[row_idx, vocab[token]] = float(value)
        return matrix

    def _mmr_select_indices(self, relevance_scores: list[float], embeddings: np.ndarray, top_k: int, mmr_lambda: float) -> list[int]:
        if top_k <= 0 or not relevance_scores:
            return []

        scores = np.asarray(relevance_scores, dtype=float)
        vectors = np.asarray(embeddings, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        if np.max(scores) > np.min(scores):
            rel = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            rel = np.ones_like(scores)

        selected = [int(np.argmax(rel))]
        remaining = set(range(len(relevance_scores))) - set(selected)

        while remaining and len(selected) < min(top_k, len(relevance_scores)):
            best_idx = None
            best_score = -float("inf")
            selected_vecs = vectors[selected]
            for idx in remaining:
                redundancy = float(np.max(np.dot(selected_vecs, vectors[idx])))
                score = mmr_lambda * float(rel[idx]) - (1.0 - mmr_lambda) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def search(
        self,
        query: str,
        topk: int = 5,
        seen_doc_ids: Optional[Set[str]] = None,
        diverse_retrieval: bool = False,
        diverse_pool_size: int = 20,
        diverse_mmr_lambda: float = 0.75,
    ) -> list[dict]:
        seen_doc_ids = seen_doc_ids or set()
        scored = []
        for idx, doc in enumerate(self.docs):
            if doc["doc_id"] in seen_doc_ids:
                continue
            value = self.score(query, idx)
            if value > 0:
                scored.append((value, idx))
        scored.sort(reverse=True)
        if not diverse_retrieval or len(scored) <= topk:
            return [self.docs[idx] for _, idx in scored[:topk]]

        pool = scored[: max(topk, diverse_pool_size)]
        candidate_indices = [idx for _, idx in pool]
        relevance_scores = [score for score, _ in pool]
        embeddings = self._candidate_embeddings(candidate_indices)
        selected = self._mmr_select_indices(
            relevance_scores=relevance_scores,
            embeddings=embeddings,
            top_k=topk,
            mmr_lambda=diverse_mmr_lambda,
        )
        return [self.docs[candidate_indices[i]] for i in selected]


def parse_action_and_content(response: str) -> tuple[str, str | None]:
    match = re.search(PATTERN, response.split("</think>")[-1], re.DOTALL)
    if not match:
        return "Search", None
    return match.group(1), match.group(2).strip()


def format_search_result(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "Searched knowledge: {}\n(The search result is empty. Please try searching from another perspective.)"

    payload = {
        doc["doc_id"]: {
            "source": doc["source"],
            "session": doc["session"],
            "text": doc["text"],
        }
        for doc in retrieved_docs
    }
    return "Searched knowledge: " + json.dumps(payload, ensure_ascii=False)


def evaluate_prediction_set(prediction_key: str, output_samples: list[dict]) -> dict:
    category_total = defaultdict(int)
    category_score = defaultdict(float)
    total = 0
    score_sum = 0.0

    for sample in output_samples:
        for qa in sample["qa"]:
            category = qa["category"]
            value = float(qa.get(f"{prediction_key}_f1", 0.0))
            category_total[category] += 1
            category_score[category] += value
            total += 1
            score_sum += value

    return {
        "overall": (score_sum / total) if total else 0.0,
        "category_total": dict(category_total),
        "category_score": dict(category_score),
        "category_avg": {
            str(cat): (category_score[cat] / category_total[cat]) if category_total[cat] else 0.0
            for cat in sorted(category_total)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M3-Agent-Control on LoCoMo using text-memory retrieval.")
    parser.add_argument("--data-file", default="third_party/locomo_official/data/locomo10.json")
    parser.add_argument("--out-file", default="data/results/locomo_m3_agent_control.json")
    parser.add_argument("--stats-file", default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--total-rounds", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--include-summaries", action="store_true")
    parser.add_argument("--diverse-retrieval", action="store_true")
    parser.add_argument("--diverse-pool-size", type=int, default=20)
    parser.add_argument("--diverse-mmr-lambda", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=1024,
    )

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.sample_limit is not None:
        samples = samples[: args.sample_limit]

    mem_banks = {}
    sample_map = {}
    for sample in samples:
        docs = build_memory_bank(
            sample,
            include_dialogue=True,
            include_observations=args.include_observations,
            include_summaries=args.include_summaries,
        )
        mem_banks[sample["sample_id"]] = BM25Index(docs)
        sample_map[sample["sample_id"]] = sample

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists() and not args.overwrite:
        with out_path.open("r", encoding="utf-8") as f:
            for sample in json.load(f):
                existing[sample["sample_id"]] = sample

    flat = []
    for sample in samples:
        out_sample = existing.get(sample["sample_id"])
        if out_sample:
            done_questions = {qa["question"] for qa in out_sample.get("qa", []) if f"{args.model}_prediction" in qa}
        else:
            done_questions = set()
        for qa in sample["qa"]:
            if args.question_limit is not None and len(flat) >= args.question_limit:
                break
            if qa["question"] in done_questions:
                continue
            flat.append(
                {
                    "sample_id": sample["sample_id"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "category": qa["category"],
                    "evidence": qa.get("evidence", []),
                    "conversations": [
                        {"role": "system", "content": SYSTEM_PROMPT.format(question=qa["question"])},
                        {"role": "user", "content": "Searched knowledge: {}"},
                    ],
                    "finish": False,
                    "retrieved_doc_ids": [],
                    "prediction": "",
                }
            )

    model = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)

    batched = []
    chunk = []
    for item in flat:
        chunk.append(item)
        if len(chunk) >= args.batch_size:
            batched.append(chunk)
            chunk = []
    if chunk:
        batched.append(chunk)

    for batch in batched:
        for round_idx in range(args.total_rounds):
            vllm_inputs = []
            live_items = []
            for item in batch:
                if item["finish"]:
                    continue
                item["conversations"][-1]["content"] += INSTRUCTION
                if round_idx == args.total_rounds - 1:
                    item["conversations"][-1]["content"] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                prompt_ids = TOKENIZER.apply_chat_template(
                    item["conversations"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                vllm_inputs.append({"prompt_token_ids": prompt_ids})
                live_items.append(item)

            if not vllm_inputs:
                break

            outputs = model.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
            for item, output in zip(live_items, outputs):
                text = output.outputs[0].text
                item["conversations"].append({"role": "assistant", "content": text})
                action, content = parse_action_and_content(text)
                if action == "Answer":
                    item["prediction"] = content or ""
                    item["finish"] = True
                    continue

                query = content or item["question"]
                index = mem_banks[item["sample_id"]]
                seen = set(item["retrieved_doc_ids"])
                retrieved = index.search(
                    query,
                    topk=args.topk,
                    seen_doc_ids=seen,
                    diverse_retrieval=args.diverse_retrieval,
                    diverse_pool_size=args.diverse_pool_size,
                    diverse_mmr_lambda=args.diverse_mmr_lambda,
                )
                item["retrieved_doc_ids"].extend(doc["doc_id"] for doc in retrieved)
                item["conversations"].append({"role": "user", "content": format_search_result(retrieved)})

        for item in batch:
            sample_id = item["sample_id"]
            sample_out = existing.setdefault(sample_id, {"sample_id": sample_id, "qa": []})
            qa_entry = next(
                (qa for qa in sample_out["qa"] if qa.get("question") == item["question"]),
                None,
            )
            if qa_entry is None:
                qa_entry = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "category": item["category"],
                    "evidence": item["evidence"],
                }
                sample_out["qa"].append(qa_entry)

            pred_key = f"{args.model}_prediction"
            ctx_key = f"{args.model}_context"
            score_key = f"{args.model}_f1"
            qa_entry[pred_key] = item["prediction"]
            qa_entry[ctx_key] = item["retrieved_doc_ids"]
            qa_entry[score_key] = round(
                evaluate_locomo_answer(item["prediction"], item["answer"], item["category"]),
                3,
            )

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(list(existing.values()), f, ensure_ascii=False, indent=2)

    stats = evaluate_prediction_set(args.model, list(existing.values()))
    stats["diverse_retrieval"] = args.diverse_retrieval
    stats["diverse_pool_size"] = args.diverse_pool_size if args.diverse_retrieval else None
    stats["diverse_mmr_lambda"] = args.diverse_mmr_lambda if args.diverse_retrieval else None
    stats_path = Path(args.stats_file) if args.stats_file else out_path.with_name(out_path.stem + "_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to {out_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Overall score: {stats['overall']:.4f}")


if __name__ == "__main__":
    main()
