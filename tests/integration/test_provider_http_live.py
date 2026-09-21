"""End-to-end transport check: the real OpenAI SDK over real (local) HTTP.

Every other provider test injects a fake client or a fake provider object, so
none of them proves the transport layer itself — that the SDK, the configured
base URL, the request payload, and the response/usage mapping work together.

This file starts a throwaway OpenAI-compatible server on ``127.0.0.1`` (an
ephemeral port, bound by the OS) and drives both the provider adapter and the
``--generate`` CLI through it. There is no credential of any value, no network
access, and no model: the stub answers with a fixed string.

What this file deliberately does **not** claim is anything about model quality.
The stub is not a model; it exists to catch transport regressions and to prove
that a credential reaches the provider as an HTTP header rather than as part of
the recorded request or the run artifact.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from evaluation import run as run_cli
from evaluation.datasets import load_jsonl
from providers import ProviderConfig, create_provider

pytest.importorskip("openai")

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
STUB_KEY = "sk-stub-not-a-real-credential"
STUB_MODEL = "stub-model"


class _Handler(BaseHTTPRequestHandler):
    """Minimal ``/v1`` surface: ``GET /models`` and ``POST /chat/completions``."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": STUB_MODEL, "object": "model"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(  # type: ignore[attr-defined]
            {"path": self.path, "body": payload, "authorization": self.headers.get("Authorization")}
        )
        self._json(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": payload.get("model", STUB_MODEL),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": self.server.answer,  # type: ignore[attr-defined]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
            },
        )


class _Stub:
    """A live stub server plus the requests it received."""

    def __init__(self, answer: str) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.answer = answer  # type: ignore[attr-defined]
        self.server.requests = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return list(self.server.requests)  # type: ignore[attr-defined]

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture()
def stub():  # type: ignore[no-untyped-def]
    task = load_jsonl(DATASET)[0]
    server = _Stub(answer=task.expected_answer)
    try:
        yield server
    finally:
        server.stop()


def test_the_sdk_reaches_a_base_url_and_normalizes_the_response(stub) -> None:  # type: ignore[no-untyped-def]
    provider = create_provider(
        ProviderConfig(
            provider="openai-compatible",
            model=STUB_MODEL,
            api_key=STUB_KEY,
            base_url=stub.base_url,
            temperature=0.0,
            max_tokens=16,
        )
    )

    response = provider.generate([{"role": "user", "content": "Which database?"}])

    assert response.text == load_jsonl(DATASET)[0].expected_answer
    assert response.model == STUB_MODEL
    assert response.usage["total_tokens"] == 45

    request = stub.requests[-1]["body"]
    assert request["model"] == STUB_MODEL
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 16
    # The credential belongs in the header; the request body is what gets logged.
    assert stub.requests[-1]["authorization"] == f"Bearer {STUB_KEY}"
    assert STUB_KEY not in json.dumps(request)
    assert provider.list_models() == [STUB_MODEL]


@pytest.mark.requires_runtime
def test_a_generated_cli_run_records_usage_without_the_credential(
    stub, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", STUB_KEY)
    output = tmp_path / "generated.json"

    code = run_cli.main(
        [
            "--mode",
            "brainos",
            "--generate",
            "--provider",
            "openai-compatible",
            "--base-url",
            stub.base_url,
            "--model",
            STUB_MODEL,
            "--dataset",
            str(DATASET),
            "--limit",
            "1",
            "--max-tokens",
            "32",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert len(stub.requests) == 1  # one task, one request, one mode
    written = output.read_text(encoding="utf-8")
    payload = json.loads(written)

    assert payload["generation_settings"]["base_url"] == stub.base_url
    assert payload["generation_usage"]["requests"] == 1
    assert payload["generation_usage"]["total_tokens"] == 45
    assert payload["aggregate_metrics"]["answer_accuracy"] == 1.0
    assert payload["task_results"][0]["answer"] == load_jsonl(DATASET)[0].expected_answer
    assert STUB_KEY not in written
