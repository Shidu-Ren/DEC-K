#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def extract_videomme_choice(response: str, all_choices: tuple[str, ...] = ("A", "B", "C", "D")) -> str:
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
            candidates = [choice for choice in all_choices if f"({choice})" in response]
        else:
            candidates = [choice for choice in all_choices if f"{choice}{suffix}" in response]
        if candidates:
            pattern = suffix
            break
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    if pattern == ")":
        indexes = [response.rfind(f"({choice})") for choice in candidates]
    else:
        indexes = [response.rfind(f"{choice}{pattern}") for choice in candidates]
    return candidates[max(range(len(candidates)), key=lambda idx: indexes[idx])]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def selected_count(record: dict[str, Any]) -> int | None:
    trace = record.get("retrieval_trace") or {}
    value = trace.get("selected_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    selector_trace = trace.get("selector_trace") or {}
    value = selector_trace.get("selected_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def summarize_output(path: Path) -> dict[str, Any]:
    results = load_json(path / "results.json")
    logs_dir = path / "logs"
    k_counts: dict[int, int] = {}
    selectors: dict[str, int] = {}
    memories: dict[str, int] = {}
    log_count = 0
    partial_correct = 0
    partial_total = 0
    if logs_dir.exists():
        for log_path in logs_dir.glob("*.json"):
            record = load_json(log_path)
            if not record:
                continue
            log_count += 1
            trace = record.get("retrieval_trace") or {}
            selector = trace.get("selector") or record.get("evidence_selector") or ""
            memory = trace.get("memory") or record.get("selector_memory") or ""
            if selector:
                selectors[str(selector)] = selectors.get(str(selector), 0) + 1
            if memory:
                memories[str(memory)] = memories.get(str(memory), 0) + 1
            k = selected_count(record)
            if k is not None:
                k_counts[k] = k_counts.get(k, 0) + 1
            if "answer" in record and "response" in record:
                pred = extract_videomme_choice(str(record.get("response", "")))
                partial_correct += pred == str(record.get("answer", ""))
                partial_total += 1

    total_k = sum(k_counts.values())
    avg_k = None
    if total_k:
        avg_k = sum(k * count for k, count in k_counts.items()) / total_k
    partial_acc = None
    if partial_total:
        partial_acc = 100.0 * partial_correct / partial_total

    return {
        "name": path.name,
        "overall": str(results.get("overall", "")).strip(),
        "long": str((results.get("Video Type") or {}).get("long", "")).strip(),
        "partial": partial_acc,
        "partial_n": partial_total,
        "logs": log_count,
        "avg_k": avg_k,
        "k_dist": dict(sorted(k_counts.items())),
        "selectors": selectors,
        "memories": memories,
        "path": path.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SiLVR QA result directories.")
    parser.add_argument("--run", required=True, help="Run name under output/videomme.")
    parser.add_argument(
        "--root",
        default="output/videomme",
        help="Root output directory containing run folders.",
    )
    parser.add_argument(
        "--contains",
        default="",
        help="Only include output directories whose name contains this substring.",
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Print a compact TSV table instead of JSON lines.",
    )
    args = parser.parse_args()

    run_root = Path(args.root) / args.run
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    rows = []
    for child in sorted(path for path in run_root.iterdir() if path.is_dir()):
        if args.contains and args.contains not in child.name:
            continue
        if not (child / "results.json").exists() and not (child / "logs").exists():
            continue
        rows.append(summarize_output(child))

    if args.tsv:
        print("name\toverall\tlong\tpartial\tpartial_n\tlogs\tavg_k\tk_dist\tselectors\tpath")
        for row in rows:
            avg_k = "" if row["avg_k"] is None else f"{row['avg_k']:.3f}"
            partial = "" if row["partial"] is None else f"{row['partial']:.1f}%"
            print(
                "\t".join(
                    [
                        row["name"],
                        row["overall"],
                        row["long"],
                        partial,
                        str(row["partial_n"]),
                        str(row["logs"]),
                        avg_k,
                        json.dumps(row["k_dist"], sort_keys=True),
                        json.dumps(row["selectors"], sort_keys=True),
                        row["path"],
                    ]
                )
            )
    else:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
