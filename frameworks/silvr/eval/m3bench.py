from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


PROMPT_AGENT_VERIFY_ANSWER_REFERENCING = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
- Do not require exact wording or matching structure.
- Semantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
- Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
- question: the question asked
- ground_truth_answer: the correct answer
- agent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
- question: {question}
- ground_truth_answer: {ground_truth_answer}
- agent_answer: {agent_answer}

Output ('Yes' or 'No'):"""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _normalize_for_exact(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'")


def _normalize_task_types(value: Any) -> list[str]:
    if value is None:
        return ["(none)"]
    if isinstance(value, list):
        return [str(item) for item in value] or ["(none)"]
    return [str(value)]


def _load_api_config() -> dict[str, Any]:
    config_path = os.environ.get(
        "M3BENCH_EVAL_API_CONFIG",
        str(Path(__file__).resolve().parents[2] / "m3_agent" / "configs" / "api_config.json"),
    )
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _api_key_for_model(model: str, api_config: dict[str, Any]) -> str:
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("M3BENCH_EVAL_OPENAI_API_KEY")
    if env_key:
        return env_key
    candidates = [model]
    if model.startswith("models/"):
        candidates.append(model[len("models/") :])
    else:
        candidates.append(f"models/{model}")
    for key in candidates:
        value = api_config.get(key)
        if isinstance(value, dict) and value.get("api_key"):
            return str(value["api_key"])
    return ""


def _extract_response_text(response_json: dict[str, Any]) -> str:
    if response_json.get("output_text"):
        return str(response_json["output_text"])
    texts: list[str] = []
    for item in response_json.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") in {"output_text", "text"} and part.get("text") is not None:
                texts.append(str(part["text"]))
    return "".join(texts)


def _judge_with_openai(prompt: str, model: str, api_key: str, timeout: float) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model[len("models/") :] if model.startswith("models/") else model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "temperature": 0,
        "max_output_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        os.environ.get("M3BENCH_EVAL_OPENAI_URL", "https://api.openai.com/v1/responses"),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI judge HTTP {response.status_code}: {response.text[:500]}")
    response_json = response.json()
    return _extract_response_text(response_json), response_json.get("usage", {}) or {}


def _judge_answer(
    question: str,
    prediction: str,
    answer: str,
    model: str,
    api_key: str,
    timeout: float,
    max_retries: int,
) -> tuple[bool, str, dict[str, Any]]:
    if not str(prediction or "").strip():
        return False, "", {}
    if _normalize_for_exact(prediction) == _normalize_for_exact(answer):
        return True, "exact", {}
    if not api_key:
        if os.environ.get("M3BENCH_EVAL_ALLOW_EXACT_FALLBACK", "0") == "1":
            return False, "missing_api_key_exact_fallback", {}
        raise RuntimeError("Missing OpenAI API key for M3Bench semantic judge")

    prompt = PROMPT_AGENT_VERIFY_ANSWER_REFERENCING.format(
        question=question,
        ground_truth_answer=answer,
        agent_answer=prediction,
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            judge_text, usage = _judge_with_openai(prompt, model, api_key, timeout)
            normalized = _normalize_for_exact(judge_text)
            return normalized.startswith("yes"), judge_text, usage
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"M3Bench judge failed: {last_error}") from last_error


def eval_m3bench(output_path, anno=None):
    output_dir = Path(output_path)
    logs = []
    for path in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        try:
            logs.append(_load_json(path))
        except Exception:
            continue

    model = os.environ.get("M3BENCH_EVAL_MODEL", "gpt-4o")
    timeout = float(os.environ.get("M3BENCH_EVAL_TIMEOUT", "60"))
    max_retries = int(os.environ.get("M3BENCH_EVAL_MAX_RETRIES", "5"))
    api_config = _load_api_config()
    api_key = _api_key_for_model(model, api_config)

    judged = []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_video: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    selected_counts: list[int] = []

    for item in tqdm(logs, desc="Judging M3Bench"):
        prediction = str(item.get("response") or "")
        answer = str(item.get("answer") or "")
        question = str(item.get("question") or "")
        correct, judge_text, judge_usage = _judge_answer(
            question,
            prediction,
            answer,
            model,
            api_key,
            timeout,
            max_retries,
        )
        trace = item.get("retrieval_trace") or {}
        if isinstance(trace, dict) and trace.get("selected_count") is not None:
            try:
                selected_counts.append(int(trace["selected_count"]))
            except Exception:
                pass
        enriched = {
            "global_idx": item.get("global_idx"),
            "id": item.get("id") or item.get("question_id"),
            "question_id": item.get("question_id") or item.get("id"),
            "video_id": item.get("video_id"),
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "correct": bool(correct),
            "judge": judge_text,
            "judge_usage": judge_usage,
            "task_type": item.get("task_type"),
            "selected_count": trace.get("selected_count") if isinstance(trace, dict) else None,
            "selected_clip_indices": trace.get("selected_clip_indices") if isinstance(trace, dict) else None,
        }
        judged.append(enriched)
        for task_type in _normalize_task_types(item.get("task_type")):
            by_type[task_type]["total"] += 1
            by_type[task_type]["correct"] += int(correct)
        video_id = str(item.get("video_id") or "(none)")
        by_video[video_id]["total"] += 1
        by_video[video_id]["correct"] += int(correct)

    total = len(judged)
    correct_total = sum(int(item["correct"]) for item in judged)
    results = {
        "dataset": "m3bench",
        "eval_model": model,
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "average_selected_clips": (sum(selected_counts) / len(selected_counts)) if selected_counts else None,
        "by_type": {
            key: {
                "total": value["total"],
                "correct": value["correct"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(by_type.items())
        },
        "by_video": {
            key: {
                "total": value["total"],
                "correct": value["correct"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(by_video.items())
        },
    }

    judged_path = output_dir.parent / "m3bench_judged_items.json"
    _save_json(judged_path, judged)
    return results
