from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .api import ChatClient
from .config import ExperimentConfig
from .embeddings import (
    OpenAICompatibleEmbeddingClient,
    SentenceTransformerEmbeddingClient,
)
from .io import read_jsonl, write_jsonl
from .pipeline import answer_record, embed_record, evaluate_records
from .runner import select_record


def _api_key(name: str) -> str:
    return os.getenv(name, "")


def _select(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    rows = (select_record(row, config) for row in read_jsonl(args.input))
    write_jsonl(args.output, rows)


def _embed(args: argparse.Namespace) -> None:
    if args.backend == "local":
        client = SentenceTransformerEmbeddingClient(
            args.model,
            device=args.device,
            query_prompt_name=args.query_prompt_name,
        )
    else:
        client = OpenAICompatibleEmbeddingClient(
            args.model,
            args.base_url,
            api_key=_api_key(args.api_key_env),
        )
    rows = (
        embed_record(
            row,
            client,
            query_field=args.query_field,
            documents_field=args.documents_field,
            batch_size=args.batch_size,
        )
        for row in read_jsonl(args.input)
    )
    write_jsonl(args.output, rows)


def _answer(args: argparse.Namespace) -> None:
    client = ChatClient(
        model=args.model,
        base_url=args.base_url,
        api_key=_api_key(args.api_key_env),
    )
    rows = (
        answer_record(
            row,
            client,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        for row in read_jsonl(args.input)
    )
    write_jsonl(args.output, rows)


def _evaluate(args: argparse.Namespace) -> None:
    judge_client = None
    if args.mode == "judge":
        judge_client = ChatClient(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=_api_key(args.judge_api_key_env),
        )
    evaluated, metrics = evaluate_records(
        read_jsonl(args.input), mode=args.mode, judge_client=judge_client
    )
    if args.output:
        write_jsonl(args.output, evaluated)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deck", description="DEC-K retrieval toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="Select evidence from prepared candidates")
    select.add_argument("--config", required=True)
    select.add_argument("--input", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(func=_select)

    embed = subparsers.add_parser("embed", help="Embed query/document JSONL")
    embed.add_argument("--input", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--backend", choices=["local", "openai"], default="local")
    embed.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    embed.add_argument("--device")
    embed.add_argument("--query-prompt-name", default="query")
    embed.add_argument("--base-url", default="http://127.0.0.1:8000")
    embed.add_argument("--api-key-env", default="OPENAI_API_KEY")
    embed.add_argument("--query-field", default="question")
    embed.add_argument("--documents-field", default="documents")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.set_defaults(func=_embed)

    answer = subparsers.add_parser("answer", help="Answer with selected textual evidence")
    answer.add_argument("--input", required=True)
    answer.add_argument("--output", required=True)
    answer.add_argument("--model", required=True)
    answer.add_argument("--base-url", default="http://127.0.0.1:8000")
    answer.add_argument("--api-key-env", default="OPENAI_API_KEY")
    answer.add_argument(
        "--system-prompt",
        default=(
            "Answer the question using only the retrieved video-memory evidence. "
            "If options are provided, return the option letter and a concise answer."
        ),
    )
    answer.add_argument("--temperature", type=float, default=0.0)
    answer.add_argument("--max-tokens", type=int, default=512)
    answer.set_defaults(func=_answer)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate prediction JSONL")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output")
    evaluate.add_argument("--metrics")
    evaluate.add_argument(
        "--mode", choices=["multiple_choice", "exact", "judge"], required=True
    )
    evaluate.add_argument("--judge-model", default="gpt-4o")
    evaluate.add_argument("--judge-base-url", default="https://api.openai.com")
    evaluate.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    evaluate.set_defaults(func=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
