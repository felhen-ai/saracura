"""Offline synthetic coverage for the opt-in hosted Decisions adapter."""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def mod() -> Any:
    path = Path(__file__).parents[1] / "research" / "gemini_bridge.py"
    spec = importlib.util.spec_from_file_location("gemini_bridge", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["gemini_bridge"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("gemini_bridge", None)


def request(question: dict[str, Any], state: str = "texto") -> bytes:
    return json.dumps(
        {"model": "gemini-decision-1", "state": state, "questions": {"q": question}}
    ).encode()


def choice(count: int = 2) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": "instrução exata",
        "criteria": {f"chave-{index}": f"descrição {index}" for index in range(count)},
    }


def upstream_choice(
    count: int, winner: int = 0, identity: str | None = "gemini-decision-1@v1"
) -> bytes:
    ids = [f"{index:0{max(2, len(str(count - 1)))}d}" for index in range(count)]
    values = {key: 0.0 for key in ids}
    values[ids[winner]] = 1.0
    result: dict[str, Any] = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": ids[winner],
                "probabilities": values,
                "confidence": 1.0,
            }
        },
        "usage": {"total_input_tokens": 7, "total_output_tokens": 0, "total_tokens": 7},
    }
    if identity is not None:
        result["model"] = identity
    return json.dumps(result).encode()


def make_bridge(
    mod: Any, tmp_path: Path, replies: list[Any], **kwargs: Any
) -> tuple[Any, list[bytes]]:
    sent: list[bytes] = []

    def transport(payload: bytes, timeout: float) -> Any:
        sent.append(payload)
        assert timeout <= 30
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    journal = mod.Journal(tmp_path / "journal.jsonl")
    return mod.Bridge(transport, journal, **kwargs), sent


def journal(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def start_server(mod: Any, bridge: Any) -> tuple[Any, threading.Thread, int]:
    server = mod.make_server(0, bridge)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker, server.server_address[1]


def post(port: int, body: bytes) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("POST", "/v1/systemone", body, {"content-type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def raw_status(port: int, headers: bytes) -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n"
        )
        return int(connection.recv(128).split(None, 2)[1])


def incomplete_body_status(port: int) -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 2\r\nConnection: close\r\n\r\n{"
        )
        connection.shutdown(socket.SHUT_WR)
        return int(connection.recv(128).split(None, 2)[1])


def test_transport_preserves_text_and_remaps_32_choices(mod: Any, tmp_path: Path) -> None:
    bridge, sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, upstream_choice(32, 31))])
    state = "  texto legal completo\n" + "x" * 100_000
    status, answer = bridge.handle(request(choice(32), state))
    assert status == 200
    assert answer["answers"]["q"]["choice"] == "chave-31"
    assert len(answer["answers"]["q"]["probabilities"]) == 32
    translated = json.loads(sent[0])
    assert translated["input"] == state
    assert translated["questions"]["q"]["instructions"] == "instrução exata"
    assert list(translated["questions"]["q"]["criteria"]) == [f"{index:02d}" for index in range(32)]
    assert "chave-0" in translated["questions"]["q"]["criteria"]["00"]
    assert "descrição 0" in translated["questions"]["q"]["criteria"]["00"]


def test_noul_is_direct_and_validated(mod: Any, tmp_path: Path) -> None:
    response = json.dumps(
        {"answers": {"q": {"type": "noul", "noul": 0.25, "confidence": 0.75}}}
    ).encode()
    bridge, sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, response)])
    assert bridge.handle(request({"type": "noul", "instructions": "É verdade?"})) == (
        200,
        {"answers": {"q": {"noul": 0.25, "confidence": 0.75}}},
    )
    assert json.loads(sent[0])["questions"]["q"] == {"type": "noul", "instructions": "É verdade?"}


@pytest.mark.parametrize(
    "raw", [b"not json", request(choice(1)), request(choice(), state="x")[:-1]]
)
def test_invalid_local_request_has_no_network(mod: Any, tmp_path: Path, raw: bytes) -> None:
    bridge, sent = make_bridge(mod, tmp_path, [])
    assert bridge.handle(raw)[0] in {400, 413}
    assert sent == []


def test_wrong_model_and_oversize_request_have_no_network(mod: Any, tmp_path: Path) -> None:
    bridge, sent = make_bridge(mod, tmp_path, [])
    wrong_model = json.dumps(
        {"model": "other", "state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    ).encode()
    assert bridge.handle(wrong_model) == (400, {"error": "unknown_model"})
    assert bridge.handle(b" " * (mod.MAX_BODY_BYTES + 1)) == (413, {"error": "too_large"})
    assert sent == []


@pytest.mark.parametrize(
    "response",
    [
        b"not-json",
        json.dumps(
            {
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": "00",
                        "probabilities": {"00": 0.4, "01": 0.6},
                        "confidence": 0.6,
                    }
                }
            }
        ).encode(),
        json.dumps(
            {
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": "xx",
                        "probabilities": {"00": 0.5, "01": 0.5},
                        "confidence": 0.5,
                    }
                }
            }
        ).encode(),
        json.dumps(
            {
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": "00",
                        "probabilities": {"00": 1.0, "01": 0.0},
                        "confidence": 2.0,
                    }
                }
            }
        ).encode(),
    ],
)
def test_invalid_upstream_is_terminal_and_latches(
    mod: Any, tmp_path: Path, response: bytes
) -> None:
    bridge, sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, response)])
    assert bridge.handle(request(choice()))[0] == 400
    assert bridge.handle(request(choice())) == (409, {"error": "outbound_latched"})
    assert len(sent) == 1


def test_winner_must_be_an_argmax(mod: Any, tmp_path: Path) -> None:
    response = json.dumps(
        {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "00",
                    "probabilities": {"00": 0.4, "01": 0.6},
                    "confidence": 0.6,
                }
            }
        }
    ).encode()
    bridge, _sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, response)])
    assert bridge.handle(request(choice()))[0] == 400


def test_identity_drift_latches_after_first_identity(mod: Any, tmp_path: Path) -> None:
    bridge, sent = make_bridge(
        mod,
        tmp_path,
        [
            mod.TransportResponse(200, upstream_choice(2, identity="gemini-decision-1@a")),
            mod.TransportResponse(200, upstream_choice(2, identity="gemini-decision-1@b")),
        ],
    )
    assert bridge.handle(request(choice()))[0] == 200
    assert bridge.handle(request(choice()))[0] == 400
    assert bridge.health()["outbound_latched"] is True
    assert len(sent) == 2


def test_model_mismatch_and_denial_are_terminal(mod: Any, tmp_path: Path) -> None:
    mismatched = upstream_choice(2, identity="another-model@v1")
    bridge, sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, mismatched)])
    assert bridge.handle(request(choice())) == (400, {"error": "upstream_failure"})
    assert len(sent) == 1
    denied_path = tmp_path / "denied"
    denied_path.mkdir()
    denied, sent = make_bridge(
        mod, denied_path, [mod.TransportResponse(403, b"raw provider error")]
    )
    # A separate fresh journal is required for every attempt.
    assert denied.handle(request(choice())) == (400, {"error": "upstream_failure"})
    assert len(sent) == 1


def test_busy_rejection_does_not_call_transport(mod: Any, tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def transport(_payload: bytes, _timeout: float) -> Any:
        nonlocal calls
        calls += 1
        entered.set()
        release.wait(2)
        return mod.TransportResponse(200, upstream_choice(2))

    bridge = mod.Bridge(transport, mod.Journal(tmp_path / "journal.jsonl"))
    worker = threading.Thread(target=lambda: bridge.handle(request(choice())))
    worker.start()
    assert entered.wait(2)
    assert bridge.handle(request(choice())) == (429, {"error": "busy"})
    release.set()
    worker.join(2)
    assert calls == 1


def test_http_server_health_and_busy_request_do_not_queue(mod: Any, tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def transport(_payload: bytes, _timeout: float) -> Any:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(2)
        return mod.TransportResponse(200, upstream_choice(2))

    bridge = mod.Bridge(transport, mod.Journal(tmp_path / "journal.jsonl"))
    server, server_thread, port = start_server(mod, bridge)
    first: list[tuple[int, dict[str, Any]]] = []
    try:
        worker = threading.Thread(target=lambda: first.append(post(port, request(choice()))))
        worker.start()
        assert entered.wait(2)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/health")
            assert connection.getresponse().status == 200
        finally:
            connection.close()
        assert post(port, request(choice())) == (429, {"error": "busy"})
        assert calls == 1
        release.set()
        worker.join(2)
        assert first == [
            (
                200,
                {
                    "answers": {
                        "q": {
                            "choice": "chave-0",
                            "probabilities": {"chave-0": 1.0, "chave-1": 0.0},
                            "confidence": 1.0,
                        }
                    }
                },
            )
        ]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(2)


def test_http_rejects_bad_declared_lengths_before_body_read(mod: Any, tmp_path: Path) -> None:
    bridge, sent = make_bridge(mod, tmp_path, [])
    server, server_thread, port = start_server(mod, bridge)
    try:
        assert raw_status(port, b"Content-Length: -1\r\nConnection: close\r\n") == 400
        assert raw_status(port, b"Content-Length: nope\r\nConnection: close\r\n") == 400
        assert raw_status(port, b"Content-Length: 4194305\r\nConnection: close\r\n") == 413
        assert raw_status(port, b"Connection: close\r\n") == 400
        assert incomplete_body_status(port) == 400
        assert sent == []
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(2)


def test_one_attempt_no_redirect_or_retry_and_sanitized_journal(mod: Any, tmp_path: Path) -> None:
    bridge, sent = make_bridge(mod, tmp_path, [mod.TransportResponse(302, b"secret redirect body")])
    raw = request(choice(), "SECRET STATE")
    assert bridge.handle(raw) == (400, {"error": "upstream_failure"})
    assert len(sent) == 1
    record = journal(tmp_path / "journal.jsonl")[0]
    serialized = json.dumps(record)
    assert "SECRET STATE" not in serialized and "secret redirect body" not in serialized
    assert record["usage"] == "unknown"
    assert record["original_request_sha256"] != "unknown"


def test_pacing_is_bounded_and_unexpected_exception_is_terminal(mod: Any, tmp_path: Path) -> None:
    now = [0.0]
    slept: list[float] = []
    bridge, _sent = make_bridge(
        mod,
        tmp_path,
        [mod.TransportResponse(200, upstream_choice(2)), RuntimeError("raw secret")],
        clock=lambda: now[0],
        sleep=slept.append,
    )
    assert bridge.handle(request(choice()))[0] == 200
    now[0] = 0.25
    assert bridge.handle(request(choice())) == (400, {"error": "internal_failure"})
    assert slept == [0.75]
    assert bridge.handle(request(choice())) == (409, {"error": "outbound_latched"})
    records = journal(tmp_path / "journal.jsonl")
    assert records[-1]["error"] == "unexpected_failure"
    assert "raw secret" not in json.dumps(records)
    with pytest.raises(ValueError):
        mod.Bridge(lambda _payload, _timeout: None, mod.Journal(tmp_path / "limit.jsonl"), 61)


def test_journal_is_fresh_and_response_digests_are_recorded(mod: Any, tmp_path: Path) -> None:
    bridge, _sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, upstream_choice(2))])
    assert bridge.handle(request(choice()))[0] == 200
    record = journal(tmp_path / "journal.jsonl")[0]
    assert record["status"] == "ok"
    assert len(record["translated_request_sha256"]) == 64
    assert record["model_identity"] == "gemini-decision-1@v1"
    with pytest.raises(ValueError):
        mod.Journal(tmp_path / "journal.jsonl")


def test_journal_completes_checked_short_writes(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = os.write

    def short_write(fd: int, data: Any) -> int:
        count = max(1, len(data) // 2)
        return original_write(fd, data[:count])

    monkeypatch.setattr(mod.os, "write", short_write)
    bridge, _sent = make_bridge(mod, tmp_path, [mod.TransportResponse(200, upstream_choice(2))])
    assert bridge.handle(request(choice()))[0] == 200
    assert journal(tmp_path / "journal.jsonl")[0]["status"] == "ok"
