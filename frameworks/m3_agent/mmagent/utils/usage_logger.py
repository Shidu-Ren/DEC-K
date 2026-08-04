import json
import os
import threading
import time
from pathlib import Path


_LOCK = threading.Lock()


def _normalize_model(model):
    text = str(model or "")
    return text[len("models/"):] if text.startswith("models/") else text


def _price_per_1m(model, kind):
    normalized = _normalize_model(model).lower()
    if kind == "embedding":
        default = 0.13 if normalized.startswith("text-embedding-3-large") else 0.02
        return float(os.getenv(f"M3AGENT_PRICE_{normalized.upper().replace('-', '_')}_PER_1M", default))

    input_default = 2.50
    output_default = 10.00
    if normalized.startswith("gpt-4o-mini"):
        input_default = 0.15
        output_default = 0.60
    input_price = float(os.getenv("M3AGENT_JUDGE_INPUT_PRICE_PER_1M", input_default))
    output_price = float(os.getenv("M3AGENT_JUDGE_OUTPUT_PRICE_PER_1M", output_default))
    return input_price, output_price


def estimate_cost_usd(model, usage, kind):
    usage = usage or {}
    if kind == "embedding":
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        return total_tokens * float(_price_per_1m(model, kind)) / 1_000_000.0

    input_price, output_price = _price_per_1m(model, kind)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


def log_api_usage(kind, model, usage=None, *, cost_usd=None, metadata=None):
    """Append one API usage event to a JSONL file when enabled.

    Set M3AGENT_USAGE_LOG_PATH to choose the target path. By default this writes
    to logs/api_usage.jsonl so old runs are unaffected but new runs are auditable.
    """
    path = os.getenv("M3AGENT_USAGE_LOG_PATH", "logs/api_usage.jsonl")
    if str(path).strip().lower() in {"", "0", "false", "none", "off"}:
        return

    usage = usage or {}
    if cost_usd is None:
        cost_usd = estimate_cost_usd(model, usage, kind)
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "run_name": os.getenv("M3AGENT_RUN_NAME", ""),
        "kind": str(kind),
        "model": str(model),
        "usage": usage,
        "cost_usd": float(cost_usd or 0.0),
        "metadata": metadata or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with target.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
