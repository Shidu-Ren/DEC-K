#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def option_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def render_question(question: str, options: list[str], query_field: str) -> str:
    question = str(question or "").strip()
    if query_field == "question":
        return question
    option_lines = [f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)]
    return "\n".join([question, *option_lines]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export VideoMME questions in the official starsuzi/Adaptive-RAG "
            "classifier predict.json format."
        )
    )
    parser.add_argument("--anno-path", required=True, help="VideoMME annotation parquet used by SiLVR.")
    parser.add_argument("--out", required=True, help="Output JSON array path for Adaptive-RAG classifier inference.")
    parser.add_argument(
        "--query-field",
        default="question",
        choices=["question", "question_with_options"],
        help="Official Adaptive-RAG trains on question text; question is the default.",
    )
    parser.add_argument("--dataset-name", default="videomme")
    args = parser.parse_args()

    import pandas as pd

    anno = pd.read_parquet(args.anno_path)
    records: list[dict[str, Any]] = []
    for global_idx, row in anno.iterrows():
        options = option_list(row.get("options"))
        records.append(
            {
                "id": str(global_idx),
                "question": render_question(row.get("question", ""), options, args.query_field),
                "answer": "",
                "dataset_name": args.dataset_name,
                "video_id": str(row.get("videoID", "")),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} Adaptive-RAG predict examples to {out_path}")


if __name__ == "__main__":
    main()
