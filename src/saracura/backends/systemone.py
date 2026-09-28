"""Optional loopback-only System One client backend for universal-choice."""

from __future__ import annotations

import http.client
import json
import math
import os
import re
from typing import Any, cast
from urllib.parse import urlsplit

from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceQuestion, DecisionRequest, ModelReference
from saracura.serialization import canonical_json_bytes

SYSTEMONE_WORKFLOW_ID = "universal-choice"
SYSTEMONE_WORKFLOW_REVISION = "phase5c-systemone.v1"
SYSTEMONE_MODEL_ID = "systemone-compatible"
SYSTEMONE_TIMEOUT_SECONDS = 30
SYSTEMONE_MAX_RESPONSE_BYTES = 1_048_576
RESIDUAL_TOLERANCE_FACTOR = 0.00005
_RESIDUAL_FLOOR = 1e-9
_NORMALIZED_SUM_TOLERANCE = 1e-12
_ENDPOINT_PATTERN = re.compile(r"^http://127\.0\.0\.1:\d+$")


def _rounding_bound(option_count: int) -> float:
    return max(1e-6, option_count * RESIDUAL_TOLERANCE_FACTOR + _RESIDUAL_FLOOR)


class SystemOneError(ValueError):
    """Adapter internal error; detail is never exposed publicly."""


def _validate_endpoint(endpoint: str) -> None:
    if not _ENDPOINT_PATTERN.fullmatch(endpoint):
        raise SaracuraError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            "endpoint must be a literal http://127.0.0.1:<port> URL",
            "/endpoint",
        )


def _resolve_bearer() -> str | None:
    return os.environ.get("SARACURA_SYSTEMONE_API_KEY")


def _build_wire_request(
    request: DecisionRequest,
    question: ChoiceQuestion,
) -> dict[str, Any]:
    criteria: dict[str, str] = {}
    for criterion in question.criteria:
        criteria[criterion.id] = criterion.description
    wire_question: dict[str, Any] = {
        "type": "choice",
        "instructions": question.instruction,
        "criteria": criteria,
    }
    questions: dict[str, Any] = {question.id: wire_question}
    return {
        "model": request.model,
        "questions": questions,
        "state": request.state,
    }


def _parse_wire_response(raw: bytes) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = child
        return value

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemOneError("System One response is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise SystemOneError("System One response root must be an object")
    allowed_fields = {"model", "answers", "usage", "latency_ms"}
    if not set(payload).issubset(allowed_fields):
        raise SystemOneError("System One response contains unknown fields")
    if "latency_ms" in payload:
        latency = payload.get("latency_ms")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            raise SystemOneError("System One latency_ms must be numeric")
    for required_field in ("model", "answers", "usage"):
        if required_field not in payload:
            raise SystemOneError(f"System One response missing {required_field}")
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise SystemOneError("System One answers must be a list")
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise SystemOneError(f"System One answer {index} is not an object")
        if set(answer) != {"type", "choice", "confidence", "probabilities"}:
            raise SystemOneError(f"System One answer {index} fields are invalid")
        for field in ("type", "choice", "confidence", "probabilities"):
            if field not in answer:
                raise SystemOneError(f"System One answer {index} missing {field}")
        if answer.get("type") != "choice":
            raise SystemOneError(f"System One answer {index} type is not choice")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            raise SystemOneError(f"System One answer {index} probabilities is not an object")
        for option_id, value in probabilities.items():
            if not isinstance(option_id, str):
                raise SystemOneError(f"System One answer {index} has non-string option id")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SystemOneError(f"System One answer {index} has non-numeric probability")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemOneError(f"System One answer {index} has out-of-range probability")
        confidence = answer.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise SystemOneError(f"System One answer {index} confidence is not numeric")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise SystemOneError(f"System One answer {index} confidence is out of range")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise SystemOneError("System One usage is not an object")
    if set(usage) != {"input_tokens", "output_tokens"}:
        raise SystemOneError("System One usage fields are invalid")
    for field in ("input_tokens", "output_tokens"):
        if field not in usage:
            raise SystemOneError(f"System One usage missing {field}")
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemOneError(f"System One usage {field} is not a non-negative integer")
    return payload


class SystemOneBackend:
    """Loopback-only System One adapter for the exact universal Choice workflow."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        model_revision: str,
        checkpoint_sha256: str,
    ) -> None:
        _validate_endpoint(endpoint)
        self._endpoint = endpoint
        self._model = ModelReference(
            id=model_id,
            revision=model_revision,
            checkpoint_sha256=checkpoint_sha256,
        )
        self._bearer = _resolve_bearer()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=10,
            max_criteria=20,
            execution_boundary="loopback-systemone-v1",
            cold_warm_semantics="not-loaded",
            quality_claims=False,
            dynamic_workflows=frozenset({(SYSTEMONE_WORKFLOW_ID, SYSTEMONE_WORKFLOW_REVISION)}),
        )

    @property
    def model(self) -> ModelReference:
        return self._model

    def validate_request(self, request: DecisionRequest) -> None:
        if (
            request.workflow.id != SYSTEMONE_WORKFLOW_ID
            or request.workflow.revision != SYSTEMONE_WORKFLOW_REVISION
        ):
            raise SaracuraError(
                ErrorCode.WORKFLOW_UNSUPPORTED,
                "System One backend accepts only universal-choice@phase5c-systemone.v1.",
                "/workflow",
            )
        if request.locale not in {"pt-BR", "en"}:
            raise SaracuraError(
                ErrorCode.LOCALE_UNVERIFIED,
                "System One backend accepts only pt-BR or en.",
                "/locale",
            )
        if len(request.questions) > 1:
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "System One sends exactly one question per call.",
                "/questions",
            )
        for index, question in enumerate(request.questions):
            if len(question.criteria) < 2:
                raise SaracuraError(
                    ErrorCode.CARDINALITY_EXCEEDED,
                    "System One requires at least two criteria per question.",
                    f"/questions/{index}/criteria",
                )

    def score_universal_choice(
        self,
        request: DecisionRequest,
        question: ChoiceQuestion,
        state_payload: bytes,
    ) -> ScoredChoice:
        del state_payload
        self.validate_request(request)
        wire_request = _build_wire_request(request, question)
        body = canonical_json_bytes(wire_request)

        parsed = urlsplit(self._endpoint)
        host: str = parsed.hostname or "127.0.0.1"
        connection = http.client.HTTPConnection(
            host,
            parsed.port or 80,
            timeout=SYSTEMONE_TIMEOUT_SECONDS,
        )
        try:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self._bearer is not None:
                headers["Authorization"] = f"Bearer {self._bearer}"
            connection.request(
                "POST",
                "/v1/systemone",
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            raw = response.read(SYSTEMONE_MAX_RESPONSE_BYTES + 1)
            status_code = response.status
            response_msg = response.reason
        except OSError as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One transport failed.",
                "/model",
            ) from error
        finally:
            connection.close()

        if status_code != 200:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"System One returned {status_code} {response_msg}.",
                "/model",
            )
        if len(raw) > SYSTEMONE_MAX_RESPONSE_BYTES:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One response exceeded the byte limit.",
                "/model",
            )

        try:
            payload = _parse_wire_response(raw)
        except SystemOneError as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One response validation failed.",
                "/model",
            ) from error

        if len(payload.get("answers", [])) != 1:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One response must contain exactly one answer.",
                "/model",
            )

        answer = payload["answers"][0]
        probabilities: dict[str, float] = {}
        for option_id, value in answer["probabilities"].items():
            probabilities[option_id] = float(value)

        criterion_ids = {c.id for c in question.criteria}
        if set(probabilities) != criterion_ids:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One probabilities do not match the question criteria.",
                "/questions/" + question.id,
            )

        for value in probabilities.values():
            if not math.isfinite(value):
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE,
                    "System One probabilities contain non-finite values.",
                    "/questions/" + question.id,
                )

        raw_sum = math.fsum(probabilities.values())
        option_count = len(probabilities)
        bound = _rounding_bound(option_count)
        if raw_sum - 1.0 > bound or 1.0 - raw_sum > bound:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One serialized probabilities exceed the rounding bound.",
                "/questions/" + question.id,
            )

        normalized = dict(probabilities)
        normalized_sum = math.fsum(normalized.values())
        if abs(normalized_sum - 1.0) > _NORMALIZED_SUM_TOLERANCE:
            total = normalized_sum
            normalized = {key: value / total for key, value in normalized.items()}

        selected = cast(str, answer["choice"])
        if selected not in probabilities:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One selected choice is not a known option.",
                "/questions/" + question.id,
            )
        max_probability = max(probabilities.values())
        if abs(probabilities[selected] - max_probability) > _NORMALIZED_SUM_TOLERANCE:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "System One selected choice does not match the maximum probability.",
                "/questions/" + question.id,
            )

        usage = payload.get("usage", {})
        input_tokens = 0
        if isinstance(usage, dict):
            tokens = usage.get("input_tokens")
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                input_tokens = tokens

        return ScoredChoice(
            question_id=question.id,
            raw_scores=dict(probabilities),
            normalized_probabilities=normalized,
            selected_choice=selected,
            input_tokens=input_tokens,
        )
