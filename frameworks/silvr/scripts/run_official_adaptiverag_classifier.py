#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official starsuzi/Adaptive-RAG t5-large query-complexity "
            "classifier on a predict.json file."
        )
    )
    parser.add_argument("--adaptive-rag-root", default="third_party/Adaptive-RAG")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Official Adaptive-RAG trained t5-large classifier checkpoint.",
    )
    parser.add_argument("--predict-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--use-slow-tokenizer", action="store_true")
    args = parser.parse_args()

    root = Path(args.adaptive_rag_root).expanduser().resolve()
    classifier_dir = root / "classifier"
    run_classifier = classifier_dir / "run_classifier.py"
    if not run_classifier.exists():
        raise FileNotFoundError(f"Official Adaptive-RAG run_classifier.py not found: {run_classifier}")

    predict_json = Path(args.predict_json).expanduser().resolve()
    if not predict_json.exists():
        raise FileNotFoundError(f"Predict JSON not found: {predict_json}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(run_classifier),
        "--model_name_or_path",
        args.model_name_or_path,
        "--validation_file",
        str(predict_json),
        "--question_column",
        "question",
        "--answer_column",
        "answer",
        "--max_seq_length",
        str(args.max_seq_length),
        "--doc_stride",
        str(args.doc_stride),
        "--per_device_eval_batch_size",
        str(args.batch_size),
        "--output_dir",
        str(output_dir),
        "--overwrite_cache",
        "--val_column",
        "validation",
        "--do_eval",
    ]
    if args.use_slow_tokenizer:
        cmd.append("--use_slow_tokenizer")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{classifier_dir}:{root}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=str(classifier_dir), env=env, check=True)

    predictions = output_dir / "dict_id_pred_results.json"
    if not predictions.exists():
        raise FileNotFoundError(f"Adaptive-RAG classifier did not write {predictions}")
    print(predictions)


if __name__ == "__main__":
    main()
