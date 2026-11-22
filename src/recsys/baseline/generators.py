"""Deterministic testing stub and optional OpenAI-compatible one-pass LLM client."""

import json
import os
from urllib import error, request
from urllib.parse import urlparse


class FixedJSONGenerator:
    """Offline test double; NOT real LLM generation or semantic quality evidence."""

    def __init__(self, response: str):
        self.response = response
        self.call_count = 0

    def generate(self, instructions: str, context: str) -> str:
        self.call_count += 1
        return self.response


class OpenAICompatibleGenerator:
    """Opt-in HTTP client for a trusted, self-hosted OpenAI-compatible chat endpoint.

    No external model credentials or network are used by automated tests.
    Avoid sending private requests/catalog records to untrusted endpoints.
    """

    def __init__(self, *, base_url: str, model: str, api_key: str | None = None,
                 timeout_seconds: float = 90.0):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("base_url must be a valid HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("non-local model endpoints must use HTTPS")
        if not model.strip() or timeout_seconds <= 0:
            raise ValueError("model must be named and timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("RAG_LLM_API_KEY")
        self.timeout_seconds = timeout_seconds

    def generate(self, instructions: str, context: str) -> str:
        data = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": context},
            ],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(f"{self.base_url}/chat/completions", data=data,
                              headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("chat endpoint returned non-text content")
            return content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("chat endpoint returned malformed response") from exc
