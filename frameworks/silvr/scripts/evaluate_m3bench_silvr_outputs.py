#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.m3bench import eval_m3bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base-path", required=True)
    parser.add_argument("--eval-model", default=os.environ.get("M3BENCH_EVAL_MODEL", "gpt-4o"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["M3BENCH_EVAL_MODEL"] = args.eval_model
    output_base = Path(args.output_base_path)
    logs_dir = output_base / "logs"
    results = eval_m3bench(logs_dir)
    result_path = output_base / "results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
