from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def _endpoint(base_url: str, resource: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if base.endswith(f"/{resource}"):
        return base
    if base.endswith("/v1"):
        return f"{base}/{resource}"
    return f"{base}/v1/{resource}"


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "",
    timeout: float = 120.0,
    retries: int = 3,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(f"Unexpected JSON response from {url}")
            return value
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Request to {url} failed after {retries} attempts") from last_error


@dataclass
class ChatClient:
    model: str
    base_url: str
    api_key: str = ""
    timeout: float = 120.0
    retries: int = 3

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> tuple[str, dict[str, Any]]:
        result = post_json(
            _endpoint(self.base_url, "chat/completions"),
            {
                "model": self.model,
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {result}") from exc
        return str(content), dict(result.get("usage") or {})
