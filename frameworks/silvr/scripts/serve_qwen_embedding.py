#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class EmbeddingService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model
        self.batch_size = max(1, int(args.batch_size))
        self.max_length = max(8, int(args.max_length))
        self.device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            padding_side="left",
        )
        model_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda" and args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        try:
            self.model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, **model_kwargs)
        except TypeError:
            self.model = AutoModel.from_pretrained(args.model, dtype=dtype, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        texts = [str(text or "") for text in texts]
        with self.lock:
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    model_output = self.model(**encoded)
                    if getattr(model_output, "last_hidden_state", None) is not None:
                        pooled = last_token_pool(model_output.last_hidden_state, encoded["attention_mask"])
                    elif getattr(model_output, "pooler_output", None) is not None:
                        pooled = model_output.pooler_output
                    else:
                        raise RuntimeError("Model output does not include last_hidden_state or pooler_output")
                    embeddings = F.normalize(pooled.float(), p=2, dim=1)
                outputs.extend(embeddings.cpu().tolist())
        return outputs


SERVICE: EmbeddingService | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "QwenEmbeddingHTTP/0.1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/health", "/v1/health"}:
            self._send_json(200, {"status": "ok", "model": SERVICE.model_name if SERVICE else ""})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": SERVICE.model_name if SERVICE else "", "object": "model"}],
                },
            )
            return
        self._send_json(404, {"error": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in {"/v1/embeddings", "/embeddings"}:
            self._send_json(404, {"error": f"Unknown path: {self.path}"})
            return
        if SERVICE is None:
            self._send_json(503, {"error": "Embedding service is not ready"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            input_value = request.get("input", [])
            if isinstance(input_value, str):
                texts = [input_value]
            elif isinstance(input_value, list):
                texts = [str(item or "") for item in input_value]
            else:
                raise ValueError("input must be a string or list of strings")
            embeddings = SERVICE.encode(texts)
            self._send_json(
                200,
                {
                    "object": "list",
                    "model": SERVICE.model_name,
                    "data": [
                        {"object": "embedding", "embedding": embedding, "index": idx}
                        for idx, embedding in enumerate(embeddings)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(traceback.format_exc(), flush=True)
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=30012, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--max-length", default=4096, type=int)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def main() -> None:
    global SERVICE
    args = parse_args()
    SERVICE = EmbeddingService(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving embeddings for {args.model} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
