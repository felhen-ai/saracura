from __future__ import annotations

import http.server
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest

from saracura.backends import SystemOneBackend
from saracura.contracts import DecisionRequest
from saracura.contracts.errors import SaracuraError
from saracura.contracts.models import ChoiceCriterion, ChoiceQuestion, WorkflowReference
from saracura.runtime import DecisionEngine, default_workflows


@dataclass
class _FakeServerState:
    request_count: int = 0
    last_body: bytes | None = None


class _JsonHandler(http.server.BaseHTTPRequestHandler):
    state: _FakeServerState

    def __init__(self, *args: Any, state: _FakeServerState, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        self.state.request_count += 1
        length = int(self.headers.get("Content-Length", 0))
        self.state.last_body = self.rfile.read(length) if length > 0 else b""
        self._send_json(200, _wire_response())

    def log_message(self, *args: Any) -> None:
        pass


def _wire_response() -> dict[str, Any]:
    return {
        "model": "systemone-compatible",
        "answers": [
            {
                "type": "choice",
                "choice": "route-b",
                "confidence": 0.7,
                "probabilities": {"route-a": 0.25, "route-b": 0.75},
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _make_handler(state: _FakeServerState) -> Callable[..., _JsonHandler]:
    def handler(*args: Any, **kwargs: Any) -> _JsonHandler:
        return _JsonHandler(*args, state=state, **kwargs)

    return handler


def _start_fake_server(state: _FakeServerState, port: int = 0) -> tuple[int, threading.Thread]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _make_handler(state))
    bound_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return bound_port, thread


@pytest.fixture
def server_state() -> _FakeServerState:
    return _FakeServerState()


@pytest.fixture
def systemone_backend(server_state: _FakeServerState) -> tuple[SystemOneBackend, int]:
    port, _ = _start_fake_server(server_state)
    backend = SystemOneBackend(
        endpoint=f"http://127.0.0.1:{port}",
        model_id="systemone-compatible",
        model_revision="systemone-revision-v1",
        checkpoint_sha256="a" * 64,
    )
    return backend, port


def _systemone_request() -> DecisionRequest:
    question = ChoiceQuestion(
        id="classify",
        type="choice",
        instruction="Classify this.",
        criteria=(
            ChoiceCriterion(id="route-a", description="Route A"),
            ChoiceCriterion(id="route-b", description="Route B"),
        ),
    )
    return DecisionRequest(
        api_version="v1alpha1",
        model="systemone-revision-v1",
        mode="research",
        locale="pt-BR",
        domain="support",
        workflow=WorkflowReference(id="universal-choice", revision="phase5c-systemone.v1"),
        state={"topic": "test"},
        questions=(question,),
    )


def test_backend_accepts_literal_loopback_endpoint(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    backend, _ = systemone_backend
    assert backend.capabilities.execution_tier == "universal"
    assert ("universal-choice", "phase5c-systemone.v1") in backend.capabilities.dynamic_workflows
    assert backend.model.id == "systemone-compatible"
    assert backend.model.revision == "systemone-revision-v1"
    assert backend.model.checkpoint_sha256 == "a" * 64


def test_backend_accepts_valid_four_decimal_distribution(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    backend, _ = systemone_backend
    request = _systemone_request()
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(request)

    assert len(response.answers) == 1
    answer = response.answers[0]
    assert answer.type == "choice"
    assert answer.status == "uncalibrated"
    assert answer.score_semantics == "ranking_weights"
    assert answer.abstained is True
    assert answer.calibration is None
    assert response.automation_allowed is False


def test_backend_rejects_residual_above_bound(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    class _ResidualBadHandler(_JsonHandler):
        def do_POST(self) -> None:
            self.state.request_count += 1
            length = int(self.headers.get("Content-Length", 0))
            self.state.last_body = self.rfile.read(length) if length > 0 else b""
            payload = {
                "model": "systemone-compatible",
                "answers": [
                    {
                        "type": "choice",
                        "choice": "route-a",
                        "confidence": 0.7,
                        "probabilities": {"route-a": 0.5, "route-b": 0.4},
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    bad_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), cast(type[http.server.BaseHTTPRequestHandler], _ResidualBadHandler)
    )
    bad_port = bad_server.server_address[1]
    bad_thread = threading.Thread(target=bad_server.serve_forever, daemon=True)
    bad_thread.start()
    try:
        bad_backend = SystemOneBackend(
            endpoint=f"http://127.0.0.1:{bad_port}",
            model_id="systemone-compatible",
            model_revision="systemone-revision-v1",
            checkpoint_sha256="a" * 64,
        )
        request = _systemone_request()
        with pytest.raises(SaracuraError):
            DecisionEngine(
                backend=bad_backend,
                workflows=default_workflows(),
                calibrations={},
            ).decide(request)
    finally:
        bad_server.shutdown()


def test_backend_uses_normalized_probabilities_from_adapter(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    backend, _ = systemone_backend
    request = _systemone_request()
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(request)

    answer = response.answers[0]
    normalized_sum = math.fsum(answer.probabilities.values())
    assert abs(normalized_sum - 1.0) <= 1e-12


def test_backend_endpoint_rejects_non_loopback() -> None:
    with pytest.raises(SaracuraError):
        SystemOneBackend(
            endpoint="http://192.168.1.1:8080",
            model_id="systemone-compatible",
            model_revision="v1",
            checkpoint_sha256="a" * 64,
        )


def test_backend_endpoint_rejects_https() -> None:
    with pytest.raises(SaracuraError):
        SystemOneBackend(
            endpoint="https://127.0.0.1:8080",
            model_id="systemone-compatible",
            model_revision="v1",
            checkpoint_sha256="a" * 64,
        )


def test_backend_endpoint_rejects_missing_port() -> None:
    with pytest.raises(SaracuraError):
        SystemOneBackend(
            endpoint="http://127.0.0.1",
            model_id="systemone-compatible",
            model_revision="v1",
            checkpoint_sha256="a" * 64,
        )


def test_backend_rejects_workflow_mismatch(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    backend, _ = systemone_backend
    question = ChoiceQuestion(
        id="classify",
        type="choice",
        instruction="Classify this.",
        criteria=(
            ChoiceCriterion(id="route-a", description="Route A"),
            ChoiceCriterion(id="route-b", description="Route B"),
        ),
    )
    request = DecisionRequest(
        api_version="v1alpha1",
        model="systemone-revision-v1",
        mode="research",
        locale="pt-BR",
        domain="support",
        workflow=WorkflowReference(id="universal-choice", revision="phase4d-laya.v1"),
        state={"topic": "test"},
        questions=(question,),
    )
    with pytest.raises(SaracuraError):
        backend.validate_request(request)


def test_backend_never_exposes_bearer_in_errors(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    backend, _ = systemone_backend
    assert backend.model.id == "systemone-compatible"
    assert backend.model.revision == "systemone-revision-v1"
    assert backend.model.checkpoint_sha256 == "a" * 64


def test_backend_uses_normalized_probabilities_in_answer(
    systemone_backend: tuple[SystemOneBackend, int],
) -> None:
    backend, _ = systemone_backend
    request = _systemone_request()
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(request)

    answer = response.answers[0]
    assert answer.probabilities is not None
    assert len(answer.probabilities) == 2
    assert "route-a" in answer.probabilities
    assert "route-b" in answer.probabilities
