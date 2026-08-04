#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CHOICES = ("A", "B", "C", "D")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def extract_choice(response: str) -> str:
    if response == "API Error" or response == "":
        return ""
    response = response.replace("\n", "")
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "
    candidates: list[str] = []
    pattern = ""
    for suffix in [".", ":", ")", " "]:
        if suffix == ")":
            candidates = [choice for choice in CHOICES if f"({choice})" in response]
        else:
            candidates = [choice for choice in CHOICES if f"{choice}{suffix}" in response]
        if candidates:
            pattern = suffix
            break
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    if pattern == ")":
        positions = [response.rfind(f"({choice})") for choice in candidates]
    else:
        positions = [response.rfind(f"{choice}{pattern}") for choice in candidates]
    return candidates[max(range(len(candidates)), key=lambda idx: positions[idx])]


def selected_count(record: dict[str, Any]) -> int | None:
    trace = record.get("retrieval_trace") or {}
    for source in (trace, trace.get("selector_trace") or {}):
        value = source.get("selected_count")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def load_run(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for log_path in (path / "logs").glob("*.json"):
        record = load_json(log_path)
        if not record:
            continue
        try:
            idx = int(log_path.stem)
        except ValueError:
            continue
        pred = extract_choice(str(record.get("response", "")))
        answer = str(record.get("answer", ""))
        rows[idx] = {
            "pred": pred,
            "answer": answer,
            "correct": pred == answer,
            "selected_count": selected_count(record),
        }
    return rows


def exact_sign_test(win: int, loss: int) -> float:
    n = win + loss
    if n == 0:
        return 1.0
    k = min(win, loss)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, p)


def avg_k(rows: dict[int, dict[str, Any]], common: list[int]) -> str:
    values = [
        int(rows[idx]["selected_count"])
        for idx in common
        if rows[idx].get("selected_count") is not None
    ]
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired comparison for two SiLVR VideoMME result directories."
    )
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    target = load_run(args.target)
    baseline = load_run(args.baseline)
    common = sorted(set(target) & set(baseline))
    both = sum(target[idx]["correct"] and baseline[idx]["correct"] for idx in common)
    target_only = sum(target[idx]["correct"] and not baseline[idx]["correct"] for idx in common)
    baseline_only = sum(not target[idx]["correct"] and baseline[idx]["correct"] for idx in common)
    neither = len(common) - both - target_only - baseline_only
    target_correct = both + target_only
    baseline_correct = both + baseline_only
    row = {
        "target": args.target_name,
        "baseline": args.baseline_name,
        "common_n": len(common),
        "target_correct": target_correct,
        "baseline_correct": baseline_correct,
        "target_acc": "" if not common else f"{100.0 * target_correct / len(common):.2f}",
        "baseline_acc": "" if not common else f"{100.0 * baseline_correct / len(common):.2f}",
        "delta_acc": "" if not common else f"{100.0 * (target_correct - baseline_correct) / len(common):+.2f}",
        "both_correct": both,
        "target_only": target_only,
        "baseline_only": baseline_only,
        "neither_correct": neither,
        "net_wins": target_only - baseline_only,
        "sign_test_p": f"{exact_sign_test(target_only, baseline_only):.6g}",
        "target_avg_k": avg_k(target, common),
        "baseline_avg_k": avg_k(baseline, common),
    }
    columns = list(row)
    line = "\t".join(str(row[column]) for column in columns)
    output = "\t".join(columns) + "\n" + line + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
