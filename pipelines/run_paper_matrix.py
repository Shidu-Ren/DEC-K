#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from deck.api import ChatClient
from deck.config import ExperimentConfig
from deck.io import read_jsonl, write_jsonl
from deck.pipeline import answer_record, evaluate_records
from deck.runner import select_record


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_entry(entry: dict[str, Any], root: Path, output_root: Path) -> dict[str, Any]:
    name = str(entry["name"])
    config = ExperimentConfig.from_yaml(_resolve(root, entry["config"]))
    input_path = _resolve(root, entry["input"])
    run_dir = output_root / name
    selected_path = run_dir / "selected.jsonl"
    write_jsonl(
        selected_path,
        (select_record(row, config) for row in read_jsonl(input_path)),
    )
    prediction_path = selected_path

    answer = entry.get("answer")
    if answer:
        import os

        client = ChatClient(
            model=str(answer["model"]),
            base_url=str(answer.get("base_url", "http://127.0.0.1:8000")),
            api_key=os.getenv(str(answer.get("api_key_env", "OPENAI_API_KEY")), ""),
        )
        prediction_path = run_dir / "predictions.jsonl"
        write_jsonl(
            prediction_path,
            (answer_record(row, client) for row in read_jsonl(selected_path)),
        )

    evaluation = entry.get("evaluation")
    metrics: dict[str, Any] = {"name": name, "selected": str(selected_path)}
    if evaluation:
        mode = str(evaluation["mode"])
        judge_client = None
        if mode == "judge":
            import os

            judge_client = ChatClient(
                model=str(evaluation.get("model", "gpt-4o")),
                base_url=str(evaluation.get("base_url", "https://api.openai.com")),
                api_key=os.getenv(
                    str(evaluation.get("api_key_env", "OPENAI_API_KEY")), ""
                ),
            )
        evaluated, result = evaluate_records(
            read_jsonl(prediction_path), mode=mode, judge_client=judge_client
        )
        write_jsonl(run_dir / "evaluated.jsonl", evaluated)
        metrics.update(result)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DEC-K experiment manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    root = args.manifest.parent
    summaries = [
        run_entry(entry, root, args.output_dir) for entry in payload.get("runs", [])
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
