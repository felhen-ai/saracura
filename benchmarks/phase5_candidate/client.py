"""Phase 5B loopback client for the pinned Kev service.

Phase 5B closure uses a Linux runner and report model for oversize probing on
127.0.0.1.  Implements the fail-closed client side of the public
TypeSafe-style contract: ``GET /v1/models`` must confirm backend, dtype,
base id and an exact local checkpoint run; every decision endpoint requires
the per-run bearer token and rejects oversize/invalid probes.  This module
uses only the standard library (``urllib``) and therefore runs in the
default environment; the fake-loopback pytest exercises every boundary
without weights or network.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from benchmarks.phase5_candidate.report import ModelsResponse, SystemOneResponse


class LoopbackError(RuntimeError):
    """A closed-loop boundary failed; never falls through silently."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def _transport(
    method: str,
    url: str,
    *,
    bearer: str | None,
    body: bytes | None,
    timeout: float,
) -> tuple[int, Mapping[str, str], bytes]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise LoopbackError("non-loopback endpoint rejected")
    headers = {"Content-Type": "application/json"}
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
    except TimeoutError as error:
        raise LoopbackError("request timed out") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise LoopbackError("request timed out") from error
        raise LoopbackError("transport failed") from error


def _parse_object(raw: bytes, endpoint: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LoopbackError(f"{endpoint} response is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise LoopbackError(f"{endpoint} response root is not an object")
    return parsed


def _validate_success(status: int, payload: dict[str, Any], endpoint: str) -> None:
    if status != 200:
        return
    try:
        SystemOneResponse.model_validate(payload)
    except Exception as error:
        raise LoopbackError(f"{endpoint} response contract failed: {error}") from error


def fetch_models(base: str, *, bearer: str, timeout: float) -> ModelsResponse:
    status, _headers, raw = _transport(
        "GET", f"{base}/v1/models", bearer=bearer, body=None, timeout=timeout
    )
    if status != 200:
        raise LoopbackError(f"/v1/models returned {status}")
    payload = _parse_object(raw, "/v1/models")
    try:
        cards = payload.get("models")
        if not isinstance(cards, list) or len(cards) != 2:
            raise ValueError("models must contain the two Kev aliases")
        if any(not isinstance(card, dict) for card in cards):
            raise ValueError("every model card must be an object")
        by_name = {str(card.get("name")): card for card in cards}
        if set(by_name) != {"kev-latest", "jev-latest"}:
            raise ValueError("unexpected model aliases")
        identities = {
            (
                card.get("backend"),
                card.get("dtype"),
                card.get("base"),
                card.get("run"),
            )
            for card in by_name.values()
        }
        if len(identities) != 1:
            raise ValueError("model aliases do not expose the same runtime identity")
        backend, dtype, base_model_id, run = identities.pop()
        normalized_dtype = "bfloat16" if str(dtype).endswith("bfloat16") else dtype
        return ModelsResponse.model_validate(
            {
                "backend": backend,
                "dtype": normalized_dtype,
                "base_model_id": base_model_id,
                "run": run,
            }
        )
    except Exception as error:
        raise LoopbackError(f"models identity check failed: {error}") from error


def post_systemone(
    base: str,
    *,
    bearer: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    status, _headers, raw = _transport(
        "POST",
        f"{base}/v1/systemone",
        bearer=bearer,
        body=body,
        timeout=timeout,
    )
    if status == 401:
        raise LoopbackError("request without a valid bearer token was rejected")
    if raw == b"":
        return status, {}
    parsed = _parse_object(raw, "/v1/systemone")
    _validate_success(status, parsed, "/v1/systemone")
    return status, parsed


def post_systemone_without_bearer(base: str, *, payload: dict[str, Any], timeout: float) -> int:
    """Probe the authorization boundary without sending an Authorization header."""
    body = json.dumps(payload).encode("utf-8")
    status, _headers, _raw = _transport(
        "POST", f"{base}/v1/systemone", bearer=None, body=body, timeout=timeout
    )
    return status


def post_systemone_separate(
    base: str,
    *,
    bearer: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """The required pinned ``/v1/systemone/separate`` question-separate endpoint."""
    body = json.dumps(payload).encode("utf-8")
    status, _headers, raw = _transport(
        "POST",
        f"{base}/v1/systemone/separate",
        bearer=bearer,
        body=body,
        timeout=timeout,
    )
    if status == 401:
        raise LoopbackError("request without a valid bearer token was rejected")
    if raw == b"":
        return status, {}
    parsed = _parse_object(raw, "/v1/systemone/separate")
    _validate_success(status, parsed, "/v1/systemone/separate")
    return status, parsed


def post_systemone_permute(
    base: str,
    *,
    bearer: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """The required pinned ``/v1/systemone/permute`` option-order permutation endpoint."""
    body = json.dumps(payload).encode("utf-8")
    status, _headers, raw = _transport(
        "POST",
        f"{base}/v1/systemone/permute",
        bearer=bearer,
        body=body,
        timeout=timeout,
    )
    if status == 401:
        raise LoopbackError("request without a valid bearer token was rejected")
    if raw == b"":
        return status, {}
    parsed = _parse_object(raw, "/v1/systemone/permute")
    _validate_success(status, parsed, "/v1/systemone/permute")
    return status, parsed


def oversize_probe(base: str, *, bearer: str, timeout: float, kind: str) -> int:
    """Issue an oversized probe.

    The two oversize paths are evaluated independently and have distinct
    contract outcomes: an oversized ``branch`` must return HTTP 422, while the
    pinned upstream currently truncates an oversized ``state`` and returns HTTP
    200.  Both statuses are returned verbatim; the caller records
    ``state_truncation_contract_failure`` when a ``state`` probe returns 200.
    """
    if kind == "branch":
        payload: dict[str, Any] = {
            "questions": [{"id": f"q{i}", "type": "choice"} for i in range(1000)]
        }
        status, _parsed = post_systemone(base, bearer=bearer, payload=payload, timeout=timeout)
        return status
    if kind == "state":
        payload = {"state": {"blob": "x" * 100_000}, "questions": [{"id": "q0", "type": "choice"}]}
        status, _parsed = post_systemone(base, bearer=bearer, payload=payload, timeout=timeout)
        return status
    raise LoopbackError(f"unknown probe kind: {kind}")
