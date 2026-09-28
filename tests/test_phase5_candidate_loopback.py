import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from benchmarks.phase5_candidate.client import (
    LoopbackError,
    fetch_models,
    oversize_probe,
    post_systemone,
    post_systemone_permute,
    post_systemone_separate,
    post_systemone_without_bearer,
)
from benchmarks.phase5_candidate.report import ModelsResponse

_BEARER = "test-token"


class _FakeServer(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _auth(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {_BEARER}"

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            if not self._auth():
                self._send(401, {})
                return
            self._send(
                200,
                {
                    "models": [
                        {
                            "name": name,
                            "backend": "mlx",
                            "dtype": "bfloat16",
                            "base": "Qwen/Qwen3.5-4B-Base",
                            "run": "/private/cache/checkpoint",
                        }
                        for name in ("kev-latest", "jev-latest")
                    ]
                },
            )
            return
        self._send(404, {})

    def do_POST(self) -> None:
        if not self._auth():
            self._send(401, {})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send(400, {})
            return
        if self.path in ("/v1/systemone", "/v1/systemone/separate", "/v1/systemone/permute"):
            questions = payload.get("questions", [])
            if isinstance(questions, list) and len(questions) > 50:
                self._send(422, {"error": "oversize branch"})
                return
            state = payload.get("state")
            if isinstance(state, dict) and state.get("blob", "") and len(state["blob"]) > 100:
                # pinned upstream truncates oversized state and still returns 200
                self._send(
                    200,
                    {
                        "answers": {},
                        "usage": {"input_tokens": 0},
                        "run": "/private/cache/checkpoint",
                    },
                )
                return
            self._send(
                200,
                {"answers": {}, "usage": {"input_tokens": 0}, "run": "/private/cache/checkpoint"},
            )
            return
        self._send(404, {})

    def _send(self, status: int, payload: dict[str, object]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def fake_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


def test_fetch_models_identity_backend_dtype(fake_server: str) -> None:
    models = fetch_models(fake_server, bearer=_BEARER, timeout=5.0)
    assert models.backend == "mlx"
    assert models.dtype == "bfloat16"
    assert models.base_model_id == "Qwen/Qwen3.5-4B-Base"


def test_fetch_models_without_bearer_rejected(fake_server: str) -> None:
    with pytest.raises(LoopbackError, match="401"):
        fetch_models(fake_server, bearer="wrong", timeout=5.0)


def test_systemone_without_bearer_rejected(fake_server: str) -> None:
    with pytest.raises(LoopbackError):
        post_systemone(fake_server, bearer="wrong", payload={"questions": []}, timeout=5.0)


def test_explicit_no_bearer_probe_requires_rejection(fake_server: str) -> None:
    assert post_systemone_without_bearer(fake_server, payload={"questions": []}, timeout=5.0) == 401


def test_oversize_branch_probe_returns_422(fake_server: str) -> None:
    assert oversize_probe(fake_server, bearer=_BEARER, timeout=5.0, kind="branch") == 422


def test_oversize_state_probe_returns_200_truncation(fake_server: str) -> None:
    # The pinned upstream truncates oversized state and returns HTTP 200; the
    # client surfaces that status verbatim so the evaluator records the
    # state_truncation_contract_failure deterministically.
    assert oversize_probe(fake_server, bearer=_BEARER, timeout=5.0, kind="state") == 200


def test_separate_endpoint_posts_and_parses(fake_server: str) -> None:
    status, parsed = post_systemone_separate(
        fake_server, bearer=_BEARER, payload={"questions": []}, timeout=5.0
    )
    assert status == 200
    assert isinstance(parsed, dict)


def test_permute_endpoint_posts_and_parses(fake_server: str) -> None:
    status, parsed = post_systemone_permute(
        fake_server, bearer=_BEARER, payload={"questions": []}, timeout=5.0
    )
    assert status == 200
    assert isinstance(parsed, dict)


def test_separate_without_bearer_rejected(fake_server: str) -> None:
    with pytest.raises(LoopbackError):
        post_systemone_separate(fake_server, bearer="wrong", payload={}, timeout=5.0)


def test_permute_without_bearer_rejected(fake_server: str) -> None:
    with pytest.raises(LoopbackError):
        post_systemone_permute(fake_server, bearer="wrong", payload={}, timeout=5.0)


def test_transport_rejects_non_loopback_url() -> None:
    with pytest.raises(LoopbackError, match="non-loopback"):
        fetch_models("http://192.168.1.10:8000", bearer=_BEARER, timeout=1.0)


def test_unknown_probe_kind_rejected(fake_server: str) -> None:
    with pytest.raises(LoopbackError, match="unknown probe kind"):
        oversize_probe(fake_server, bearer=_BEARER, timeout=5.0, kind="nope")


def test_models_response_rejects_non_absolute_run() -> None:
    with pytest.raises(ValueError, match="absolute local checkpoint"):
        ModelsResponse.model_validate(
            {
                "backend": "mlx",
                "dtype": "bfloat16",
                "base_model_id": "Qwen/Qwen3.5-4B-Base",
                "run": "relative/checkpoint",
            }
        )


def test_models_response_rejects_wrong_backend() -> None:
    with pytest.raises(ValueError):
        ModelsResponse.model_validate(
            {
                "backend": "torch",
                "dtype": "bfloat16",
                "base_model_id": "Qwen/Qwen3.5-4B-Base",
                "run": "/checkpoint",
            }
        )
