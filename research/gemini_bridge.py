"""Loopback-only, opt-in bridge for the hosted Gemini Decisions API.

This is deliberately a research adapter, not a model runtime.  It accepts the
existing System One request shape on localhost and makes one (and only one)
HTTPS request to the fixed Decisions endpoint.  The journal is deliberately
sanitized: it is evidence of an attempt, never a copy of benchmark text.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import http.server
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL = "gemini-decision-1"
HOST = "127.0.0.1"
ENDPOINT_HOST = "generativelanguage.googleapis.com"
ENDPOINT_PATH = "/v1beta/decisions"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_QUESTIONS = 16
MAX_CHOICES = 255
MAX_UPSTREAM_BYTES = 4 * 1024 * 1024
UPSTREAM_TIMEOUT = 30.0
MAX_PACING_WAIT = 60.0
PROBABILITY_SUM_TOLERANCE = 1e-6
SAFE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}")
USAGE_FIELDS = frozenset({"total_input_tokens", "total_output_tokens", "total_tokens"})


class RequestError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code


class UpstreamError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


@dataclass(frozen=True)
class Question:
    name: str
    kind: str
    instructions: str
    criteria: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ParsedRequest:
    state: str
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate")
        out[key] = value
    return out


def _text(value: Any) -> str:
    if not isinstance(value, str) or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise RequestError(400, "invalid_request")
    return value


def _closed(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RequestError(400, "invalid_request")
    return value


def _question(name: str, raw: Any) -> Question:
    raw = (
        _closed(raw, {"type", "instructions"})
        if isinstance(raw, dict) and raw.get("type") == "noul"
        else raw
    )
    if isinstance(raw, dict) and raw.get("type") == "noul":
        return Question(_text(name), "noul", _text(raw["instructions"]))
    raw = _closed(raw, {"type", "instructions", "criteria"})
    if raw["type"] != "choice" or not isinstance(raw["criteria"], dict):
        raise RequestError(400, "invalid_request")
    if not 2 <= len(raw["criteria"]) <= MAX_CHOICES:
        raise RequestError(400, "invalid_request")
    criteria = tuple(
        (_text(key), _text(description)) for key, description in raw["criteria"].items()
    )
    return Question(_text(name), "choice", _text(raw["instructions"]), criteria)


def parse_request(raw: bytes) -> ParsedRequest:
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError(413, "too_large")
    try:
        body = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise RequestError(400, "invalid_json") from None
    body = _closed(body, {"model", "state", "questions"})
    if _text(body["model"]) != MODEL:
        raise RequestError(400, "unknown_model")
    questions = body["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(400, "invalid_request")
    return ParsedRequest(
        _text(body["state"]),
        tuple(_question(name, question) for name, question in questions.items()),
    )


def _option_id(index: int, count: int) -> str:
    return f"{index:0{max(2, len(str(count - 1)))}d}"


def translate(request: ParsedRequest) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    """Return the provider JSON and its positional choice reverse maps."""
    questions: dict[str, Any] = {}
    reverse: dict[str, tuple[str, ...]] = {}
    for question in request.questions:
        if question.kind == "noul":
            questions[question.name] = {"type": "noul", "instructions": question.instructions}
            continue
        keys = tuple(key for key, _description in question.criteria)
        reverse[question.name] = keys
        # Google sorts criterion keys bytewise. Fixed-width ASCII ordinals retain incoming order.
        criteria = {
            _option_id(
                index, len(keys)
            ): f"original_key: {key}\noriginal_description: {description}"
            for index, (key, description) in enumerate(question.criteria)
        }
        questions[question.name] = {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": criteria,
        }
    return {"model": MODEL, "input": request.state, "questions": questions}, reverse


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise UpstreamError("invalid_response")
    value = float(value)
    if not 0 <= value <= 1:
        raise UpstreamError("invalid_response")
    return value


def _identity(response: dict[str, Any]) -> str | None:
    value = response.get("model")
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not SAFE_MODEL.fullmatch(value)
        or not (value == MODEL or value.startswith(f"{MODEL}@"))
    ):
        raise UpstreamError("invalid_response")
    return value


def remap_response(
    response_bytes: bytes, request: ParsedRequest, reverse: dict[str, tuple[str, ...]]
) -> tuple[dict[str, Any], str | None, dict[str, int] | str]:
    if len(response_bytes) > MAX_UPSTREAM_BYTES:
        raise UpstreamError("invalid_response")
    try:
        response = json.loads(response_bytes.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise UpstreamError("invalid_response") from None
    if not isinstance(response, dict) or set(response) - {"answers", "usage", "model"}:
        raise UpstreamError("invalid_response")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != {
        question.name for question in request.questions
    }:
        raise UpstreamError("invalid_response")
    output: dict[str, Any] = {}
    for question in request.questions:
        answer = answers[question.name]
        if not isinstance(answer, dict) or answer.get("type") != question.kind:
            raise UpstreamError("invalid_response")
        confidence = _number(answer.get("confidence"))
        if question.kind == "noul":
            if set(answer) != {"type", "noul", "confidence"}:
                raise UpstreamError("invalid_response")
            output[question.name] = {"noul": _number(answer["noul"]), "confidence": confidence}
            continue
        keys = reverse[question.name]
        ids = tuple(_option_id(index, len(keys)) for index in range(len(keys)))
        probabilities = answer.get("probabilities")
        if set(answer) != {"type", "choice", "probabilities", "confidence"} or not isinstance(
            probabilities, dict
        ):
            raise UpstreamError("invalid_response")
        if set(probabilities) != set(ids) or answer.get("choice") not in probabilities:
            raise UpstreamError("invalid_response")
        values = {option_id: _number(probabilities[option_id]) for option_id in ids}
        if abs(math.fsum(values.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise UpstreamError("invalid_response")
        winner = answer["choice"]
        if values[winner] != max(values.values()):
            raise UpstreamError("invalid_response")
        output[question.name] = {
            "choice": keys[ids.index(winner)],
            "probabilities": {
                keys[index]: values[option_id] for index, option_id in enumerate(ids)
            },
            "confidence": confidence,
        }
    usage = response.get("usage", "unknown")
    if usage != "unknown":
        if not isinstance(usage, dict):
            raise UpstreamError("invalid_response")
        usage = {
            key: value
            for key, value in usage.items()
            if key in USAGE_FIELDS
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
        if not usage:
            raise UpstreamError("invalid_response")
    return {"answers": output}, _identity(response), usage


def sanitized_answers(request: ParsedRequest, answers: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence of returned values without copying request/upstream strings to the journal."""
    out: list[dict[str, Any]] = []
    for question in request.questions:
        answer = answers[question.name]
        if question.kind == "noul":
            out.append({"type": "noul", "noul": answer["noul"], "confidence": answer["confidence"]})
            continue
        keys = tuple(key for key, _description in question.criteria)
        out.append(
            {
                "type": "choice",
                "winner_position": keys.index(answer["choice"]),
                "probabilities": [answer["probabilities"][key] for key in keys],
                "confidence": answer["confidence"],
            }
        )
    return out


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise ValueError("journal_unavailable") from exc

    def append(self, record: dict[str, Any]) -> None:
        line = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        with self._lock:
            try:
                remaining = memoryview(line)
                while remaining:
                    written = os.write(self._fd, remaining)
                    if not isinstance(written, int) or written <= 0:
                        raise OSError("short journal write")
                    remaining = remaining[written:]
                os.fsync(self._fd)
            except OSError as exc:
                raise UpstreamError("journal_failure") from exc

    def close(self) -> None:
        os.close(self._fd)


def https_transport(api_key: str) -> Callable[[bytes, float], TransportResponse]:
    def send(payload: bytes, timeout: float) -> TransportResponse:
        connection = http.client.HTTPSConnection(ENDPOINT_HOST, 443, timeout=timeout)
        try:
            connection.request(
                "POST",
                ENDPOINT_PATH,
                payload,
                {"content-type": "application/json", "x-goog-api-key": api_key},
            )
            response = connection.getresponse()
            return TransportResponse(response.status, response.read(MAX_UPSTREAM_BYTES + 1))
        finally:
            connection.close()

    return send


class Bridge:
    def __init__(
        self,
        transport: Callable[[bytes, float], TransportResponse],
        journal: Journal,
        minimum_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 <= minimum_interval <= MAX_PACING_WAIT:
            raise ValueError("minimum_interval")
        self.transport, self.journal, self.minimum_interval = transport, journal, minimum_interval
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()
        self._last_attempt: float | None = None
        self._latched = False
        self._pinned_identity: str | None = None

    def health(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "model_revision": "unknown",
            "remote_serving": True,
            "strict_local_weights_evidence": False,
            "outbound_latched": self._latched,
            "provider_model_identity": self._pinned_identity or "unknown",
            "minimum_interval_seconds": self.minimum_interval,
            "upstream_timeout_seconds": UPSTREAM_TIMEOUT,
        }

    def handle(self, raw: bytes) -> tuple[int, dict[str, Any]]:
        try:
            request = parse_request(raw)
        except RequestError as exc:
            return exc.status, {"error": exc.code}
        if not self._lock.acquire(blocking=False):
            return 429, {"error": "busy"}
        try:
            if self._latched:
                return 409, {"error": "outbound_latched"}
            return self._admitted(raw, request)
        except Exception:
            self._latched = True
            return 400, {"error": "internal_failure"}
        finally:
            self._lock.release()

    def _admitted(self, raw: bytes, request: ParsedRequest) -> tuple[int, dict[str, Any]]:
        now = self.clock()
        if self._last_attempt is not None:
            wait = self.minimum_interval - (now - self._last_attempt)
            if wait > 0:
                if wait > MAX_PACING_WAIT:
                    return 429, {"error": "pacing_limit"}
                self.sleep(wait)
        translated, reverse = translate(request)
        payload = json.dumps(
            translated, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        self._last_attempt = self.clock()
        started = time.monotonic()
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "original_request_sha256": digest(raw),
            "translated_request_sha256": digest(payload),
            "upstream_response_sha256": "unknown",
            "duration_ms": 0,
            "status": "failed",
            "error": "unknown",
            "usage": "unknown",
            "model_identity": "unknown",
        }
        try:
            remote = self.transport(payload, UPSTREAM_TIMEOUT)
            record["upstream_response_sha256"] = digest(remote.body)
            if remote.status != 200:
                raise UpstreamError(
                    "upstream_denied" if remote.status in {403, 429} else "upstream_failure"
                )
            result, identity, usage = remap_response(remote.body, request, reverse)
            if (
                identity is not None
                and self._pinned_identity is not None
                and identity != self._pinned_identity
            ):
                raise UpstreamError("model_drift")
            if identity is not None:
                self._pinned_identity = identity
            record.update(
                {
                    "status": "ok",
                    "error": "none",
                    "usage": usage,
                    "model_identity": identity or "unknown",
                    "answer": sanitized_answers(request, result["answers"]),
                }
            )
            self.journal.append(record | {"duration_ms": int((time.monotonic() - started) * 1000)})
            return 200, result
        except UpstreamError as error:
            self._latched = True
            record["error"] = error.code
            with contextlib.suppress(UpstreamError):
                self.journal.append(
                    record | {"duration_ms": int((time.monotonic() - started) * 1000)}
                )
            return 400, {"error": "upstream_failure"}
        except (OSError, TimeoutError, http.client.HTTPException):
            self._latched = True
            record["error"] = "transport_failure"
            with contextlib.suppress(UpstreamError):
                self.journal.append(
                    record | {"duration_ms": int((time.monotonic() - started) * 1000)}
                )
            return 400, {"error": "upstream_failure"}
        except Exception:
            self._latched = True
            record["error"] = "unexpected_failure"
            with contextlib.suppress(UpstreamError):
                self.journal.append(
                    record | {"duration_ms": int((time.monotonic() - started) * 1000)}
                )
            return 400, {"error": "internal_failure"}


class Handler(http.server.BaseHTTPRequestHandler):
    bridge: Bridge

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._reply(200, self.bridge.health()) if self.path == "/health" else self._reply(
            404, {"error": "not_found"}
        )

    def do_POST(self) -> None:
        if self.path != "/v1/systemone":
            self._reply(404, {"error": "not_found"})
            return
        values = self.headers.get_all("content-length")
        if values is None or len(values) != 1 or not re.fullmatch(r"[0-9]+", values[0]):
            self._reply(400, {"error": "invalid_request"})
            return
        try:
            length = int(values[0])
        except ValueError:
            self._reply(400, {"error": "invalid_request"})
            return
        if length > MAX_BODY_BYTES:
            self._reply(413, {"error": "too_large"})
            return
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._reply(400, {"error": "invalid_request"})
            return
        self._reply(*self.bridge.handle(raw))


def make_server(port: int, bridge: Bridge) -> http.server.ThreadingHTTPServer:
    """Create the loopback server; request workers keep health outside admission."""

    class BoundHandler(Handler):
        pass

    BoundHandler.bridge = bridge
    server = http.server.ThreadingHTTPServer((HOST, port), BoundHandler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18041)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--minimum-interval", type=float, default=1.0)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required")
    journal = Journal(args.journal)
    bridge = Bridge(https_transport(api_key), journal, args.minimum_interval)
    with make_server(args.port, bridge) as server:
        try:
            server.serve_forever()
        finally:
            journal.close()


if __name__ == "__main__":
    main()
