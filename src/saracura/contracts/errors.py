"""Stable error codes and envelope for v1alpha1."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, JsonValue

from saracura.contracts.models import ClosedModel


class ErrorCode(StrEnum):
    REQUEST_INVALID = "REQUEST_INVALID"
    API_VERSION_UNSUPPORTED = "API_VERSION_UNSUPPORTED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_ALIAS_FORBIDDEN = "MODEL_ALIAS_FORBIDDEN"
    WORKFLOW_UNSUPPORTED = "WORKFLOW_UNSUPPORTED"
    SCHEMA_UNSUPPORTED = "SCHEMA_UNSUPPORTED"
    CARDINALITY_EXCEEDED = "CARDINALITY_EXCEEDED"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    CALIBRATION_MISSING = "CALIBRATION_MISSING"
    CALIBRATION_INCOMPATIBLE = "CALIBRATION_INCOMPATIBLE"
    CACHE_INCOMPATIBLE = "CACHE_INCOMPATIBLE"
    DOMAIN_UNVERIFIED = "DOMAIN_UNVERIFIED"
    LOCALE_UNVERIFIED = "LOCALE_UNVERIFIED"
    RISK_POLICY_MISSING = "RISK_POLICY_MISSING"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorPayload(ClosedModel):
    code: ErrorCode
    message: str
    path: str
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ErrorEnvelope(ClosedModel):
    api_version: Literal["v1alpha1"] = "v1alpha1"
    error: ErrorPayload


class SaracuraError(Exception):
    """Expected contract/runtime error with a stable public representation."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        path: str,
        *,
        retryable: bool = False,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = ErrorPayload(
            code=code,
            message=message,
            path=path,
            retryable=retryable,
            details=details or {},
        )

    def envelope(self) -> ErrorEnvelope:
        return ErrorEnvelope(error=self.payload)

    def as_dict(self) -> dict[str, Any]:
        return self.envelope().model_dump(mode="json")
