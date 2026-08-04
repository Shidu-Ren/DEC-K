#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def extract_videomme_choice(response: str, choices: tuple[str, ...] = ("A", "B", "C", "D")) -> str:
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
            candidates = [choice for choice in choices if f"({choice})" in response]
        else:
            candidates = [choice for choice in choices if f"{choice}{suffix}" in response]
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
    for source in [trace, trace.get("selector_trace") or {}]:
        value = source.get("selected_count")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def load_run(path: Path) -> dict[str, Any]:
    logs_dir = path / "logs"
    examples: dict[int, tuple[str, str]] = {}
    k_counts: Counter[int] = Counter()
    if logs_dir.exists():
        for log_path in logs_dir.glob("*.json"):
            record = load_json(log_path)
            if not record:
                continue
            try:
                idx = int(log_path.stem)
            except ValueError:
                continue
            examples[idx] = (
                extract_videomme_choice(str(record.get("response", ""))),
                str(record.get("answer", "")),
            )
            k = selected_count(record)
            if k is not None:
                k_counts[k] += 1
    total = len(examples)
    correct = sum(pred == answer for pred, answer in examples.values())
    avg_k = None
    if k_counts:
        avg_k = sum(k * count for k, count in k_counts.items()) / sum(k_counts.values())
    return {
        "name": path.name,
        "examples": examples,
        "n": total,
        "correct": correct,
        "acc": None if total == 0 else 100.0 * correct / total,
        "avg_k": avg_k,
        "k_dist": dict(sorted(k_counts.items())),
        "path": path,
    }


def format_pct(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SiLVR result directories on their full and common completed examples."
    )
    parser.add_argument("--run", required=True, help="Run name under output/videomme.")
    parser.add_argument(
        "--root",
        default="output/videomme",
        help="Root output directory containing run folders.",
    )
    parser.add_argument("--contains", default="", help="Only include output dirs containing this text.")
    parser.add_argument(
        "--baseline",
        default="",
        help="Baseline output directory name. If set, report common-index deltas against it.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target output directory name. May be repeated. Defaults to all included dirs.",
    )
    args = parser.parse_args()

    run_root = Path(args.root) / args.run
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    rows = []
    for child in sorted(path for path in run_root.iterdir() if path.is_dir()):
        if args.contains and args.contains not in child.name:
            continue
        if not (child / "logs").exists():
            continue
        rows.append(load_run(child))

    by_name = {row["name"]: row for row in rows}
    if args.baseline:
        if args.baseline not in by_name:
            raise KeyError(f"Baseline not found: {args.baseline}")
        baseline = by_name[args.baseline]
        targets = [by_name[name] for name in args.target] if args.target else rows
        print(
            "name\tn\tacc\tavg_k\tcommon_n\tbaseline_common\tcommon_acc\tdelta_vs_baseline\tk_dist\tpath"
        )
        for row in targets:
            common = sorted(set(row["examples"]) & set(baseline["examples"]))
            row_correct = sum(row["examples"][idx][0] == row["examples"][idx][1] for idx in common)
            base_correct = sum(
                baseline["examples"][idx][0] == baseline["examples"][idx][1] for idx in common
            )
            row_common_acc = None if not common else 100.0 * row_correct / len(common)
            base_common_acc = None if not common else 100.0 * base_correct / len(common)
            delta = None
            if row_common_acc is not None and base_common_acc is not None:
                delta = row_common_acc - base_common_acc
            print(
                "\t".join(
                    [
                        row["name"],
                        str(row["n"]),
                        format_pct(row["acc"]),
                        "" if row["avg_k"] is None else f"{row['avg_k']:.3f}",
                        str(len(common)),
                        format_pct(base_common_acc),
                        format_pct(row_common_acc),
                        "" if delta is None else f"{delta:+.2f}",
                        json.dumps(row["k_dist"], sort_keys=True),
                        row["path"].as_posix(),
                    ]
                )
            )
    else:
        print("name\tn\tacc\tavg_k\tk_dist\tpath")
        for row in rows:
            print(
                "\t".join(
                    [
                        row["name"],
                        str(row["n"]),
                        format_pct(row["acc"]),
                        "" if row["avg_k"] is None else f"{row['avg_k']:.3f}",
                        json.dumps(row["k_dist"], sort_keys=True),
                        row["path"].as_posix(),
                    ]
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
