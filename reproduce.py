#!/usr/bin/env python3
"""Run the paper configurations in the bundled M3-Agent and SiLVR frameworks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M3_ROOT = ROOT / "frameworks" / "m3_agent"
SILVR_ROOT = ROOT / "frameworks" / "silvr"


def print_command(command: list[str], cwd: Path) -> None:
    printable = " ".join(subprocess.list2cmdline([part]) for part in command)
    print(f"cd {cwd}\n{printable}", flush=True)


def m3_agent_command(args: argparse.Namespace) -> list[str]:
    data_file = (
        args.data_file or M3_ROOT / "data" / "annotations" / f"{args.benchmark}.json"
    )
    command = [
        sys.executable,
        "m3_agent/control.py",
        "--data_file",
        str(Path(data_file).resolve()),
        "--output_name",
        args.output_name or f"{args.benchmark}_{args.method}",
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--topk",
        str(
            2
            if args.method == "original"
            else args.fixed_k
            if args.method == "fixed_topk"
            else args.top_k
        ),
        "--retrieval_threshold=0.5"
        if args.method == "original"
        else "--retrieval_threshold=-inf",
        "--batch_size",
        str(args.batch_size),
        "--consumer_workers",
        str(args.consumer_workers),
    ]
    if args.max_model_len:
        command += ["--max_model_len", str(args.max_model_len)]
    if args.list_file:
        command += ["--list_file", str(Path(args.list_file).resolve())]
    if args.question_ids_file:
        command += ["--question_ids_file", str(Path(args.question_ids_file).resolve())]

    if args.method == "original":
        return command
    if args.method == "fixed_topk":
        command += ["--fixed_clip_backfill_current"]
        return command

    command += [
        "--diverse_clip_retrieval",
        "--diverse_clip_pool_size",
        str(args.fixed_k if args.method == "mmr_fixed_topk" else args.max_k + 1),
        "--diverse_clip_mmr_candidate_pool_size",
        str(args.candidate_pool),
        "--clip_intra_similarity_threshold",
        str(args.intra_clip_threshold),
        "--clip_mmr_lambda",
        str(args.lambda_),
        "--clip_max_nodes_for_diversity",
        str(args.max_nodes_per_clip),
        "--dynamic_mmr_min_clips",
        str(args.min_k),
        "--dynamic_mmr_max_clips",
        str(args.max_k),
    ]
    if args.method == "mmr_fixed_topk":
        return command
    if args.method == "deck":
        command += [
            "--dynamic_mmr_clip_retrieval",
            "--dynamic_mmr_policy",
            "raw_mmr_delta_root",
            "--dynamic_mmr_score_source",
            "clip_score",
        ]
    elif args.method == "adaptivek":
        command += [
            "--clip_adaptive_k_retrieval",
            "--clip_adaptive_k_min_clips",
            str(args.min_k),
            "--clip_adaptive_k_max_clips",
            str(args.max_k),
            "--clip_adaptive_k_retrieve_more",
            str(args.adaptivek_more),
            "--clip_adaptive_k_score_source",
            "max_node",
        ]
    elif args.method == "adaptiverag":
        if not args.adaptiverag_labels:
            raise SystemExit("--adaptiverag-labels is required for Adaptive-RAG")
        command += [
            "--adaptive_rag_retrieval",
            "--adaptive_rag_route_source",
            "file",
            "--adaptive_rag_classifier_path",
            str(Path(args.adaptiverag_labels).resolve()),
            "--adaptive_rag_zero_clips",
            str(args.adaptiverag_a),
            "--adaptive_rag_single_clips",
            str(args.adaptiverag_b),
            "--adaptive_rag_multi_clips",
            str(args.adaptiverag_c),
            "--adaptive_rag_selector",
            "top",
            "--adaptive_rag_score_source",
            "max_node",
        ]
    return command


def silvr_command(args: argparse.Namespace) -> list[str]:
    dataset = (
        "videomme" if args.benchmark == "videomme_long" else f"m3bench_{args.benchmark}"
    )
    if args.method == "original":
        raise SystemExit("The original control baseline is specific to M3-Agent")
    selector = "mmr_relret" if args.method == "deck" else args.method
    command = [
        sys.executable,
        "main.py",
        "--dataset",
        dataset,
        "--anno_path",
        str(Path(args.annotation).resolve()),
        "--output_base_path",
        str(Path(args.output_dir).resolve()),
        "--model",
        "qwen3vl",
        "--model_path",
        args.answer_model,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--prompt_type",
        "videomme" if args.benchmark == "videomme_long" else "m3bench",
        "--evidence_selector",
        selector,
        "--selector_memory",
        "caption_subtitle",
        "--retrieval_backend",
        "dense",
        "--embedding_model",
        args.embedding_model,
        "--embedding_cache_dir",
        str(Path(args.embedding_cache).resolve()),
        "--retrieval_top_k",
        str(args.candidate_pool),
        "--selector_pool_size",
        str(args.max_k + 1),
        "--qa_top_k",
        str(args.fixed_k),
        "--relret_min_chunks",
        str(args.min_k),
        "--relret_max_chunks",
        str(args.max_k),
        "--mmr_lambda",
        str(args.lambda_),
        "--clip_length",
        "30",
        "--single_process",
    ]
    if args.caption_path:
        command += ["--caption_path", str(Path(args.caption_path).resolve())]
    if args.subtitle_path:
        command += ["--subtitle_path", str(Path(args.subtitle_path).resolve())]
    if args.method == "adaptivek":
        command += [
            "--adaptivek_min_chunks",
            str(args.min_k),
            "--adaptivek_max_chunks",
            str(args.max_k),
            "--adaptivek_retrieve_more",
            str(args.adaptivek_more),
        ]
    elif args.method == "adaptiverag":
        if not args.adaptiverag_labels:
            raise SystemExit("--adaptiverag-labels is required for Adaptive-RAG")
        command += [
            "--adaptiverag_route_source",
            "file",
            "--adaptiverag_classifier_path",
            str(Path(args.adaptiverag_labels).resolve()),
            "--adaptiverag_a_chunks",
            str(args.adaptiverag_a),
            "--adaptiverag_b_chunks",
            str(args.adaptiverag_b),
            "--adaptiverag_c_chunks",
            str(args.adaptiverag_c),
        ]
    return command


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark", choices=["robot", "web", "videomme_long"], required=True
    )
    parser.add_argument(
        "--method",
        choices=[
            "deck",
            "original",
            "fixed_topk",
            "mmr_fixed_topk",
            "adaptivek",
            "adaptiverag",
        ],
        default="deck",
    )
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.85)
    parser.add_argument("--fixed-k", type=int, default=7)
    parser.add_argument("--adaptivek-more", type=int)
    parser.add_argument("--adaptiverag-labels", default="")
    parser.add_argument("--adaptiverag-a", type=int, default=2)
    parser.add_argument("--adaptiverag-b", type=int)
    parser.add_argument("--adaptiverag-c", type=int)
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="framework", required=True)

    m3 = subparsers.add_parser(
        "m3-agent", help="Run the bundled M3-Agent control pipeline"
    )
    add_common(m3)
    m3.add_argument("--data-file", default="")
    m3.add_argument("--output-name", default="")
    m3.add_argument("--list-file", default="")
    m3.add_argument("--question-ids-file", default="")
    m3.add_argument("--tensor-parallel-size", type=int, default=1)
    m3.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    m3.add_argument("--max-model-len", type=int, default=0)
    m3.add_argument("--top-k", type=int, default=5)
    m3.add_argument("--batch-size", type=int, default=64)
    m3.add_argument("--consumer-workers", type=int, default=0)
    m3.add_argument("--intra-clip-threshold", type=float, default=0.85)
    m3.add_argument("--max-nodes-per-clip", type=int, default=8)

    silvr = subparsers.add_parser("silvr", help="Run the bundled SiLVR pipeline")
    add_common(silvr)
    silvr.add_argument("--annotation", required=True)
    silvr.add_argument("--caption-path", default="")
    silvr.add_argument("--subtitle-path", default="")
    silvr.add_argument("--output-dir", required=True)
    silvr.add_argument("--answer-model", default="Qwen/Qwen3-VL-8B-Instruct")
    silvr.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    silvr.add_argument("--embedding-cache", default="data/embedding_cache")
    silvr.add_argument("--max-new-tokens", type=int, default=128)

    args = parser.parse_args()
    if args.max_k is None:
        args.max_k = 5 if args.framework == "m3-agent" else 8
    if args.adaptivek_more is None:
        args.adaptivek_more = 2 if args.framework == "m3-agent" else 5
    if args.adaptiverag_b is None:
        args.adaptiverag_b = 4 if args.framework == "m3-agent" else 7
    if args.adaptiverag_c is None:
        args.adaptiverag_c = 5 if args.framework == "m3-agent" else 8
    return args


def main() -> None:
    args = parse_args()
    if args.framework == "m3-agent":
        command = m3_agent_command(args)
        cwd = M3_ROOT
        env = None
    else:
        command = silvr_command(args)
        cwd = SILVR_ROOT
        env = os.environ.copy()
        env.setdefault("SILVR_MMR_RELEVANCE_MODE", "minmax")
        env.setdefault("SILVR_RELRET_POLICY", "raw_mmr_delta_root")
        env.setdefault("SILVR_RELRET_MIN_DECISION_K", "2")
    print_command(command, cwd)
    if not args.dry_run:
        subprocess.run(command, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
