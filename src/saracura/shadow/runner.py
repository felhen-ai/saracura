"""Resident, read-only runner for bounded shadow decision batches."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from saracura.backends.base import UniversalBackend
from saracura.backends.saracura_universal import validate_saracura_request_structure
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceQuestion, DecisionRequest
from saracura.runtime import DecisionEngine, default_workflows
from saracura.shadow.evaluation import project_shadow_decision, validate_shadow_item_state
from saracura.shadow.models import ShadowDecisionRecord, ShadowItem, ShadowPolicy


def _unavailable() -> SaracuraError:
    return SaracuraError(
        ErrorCode.BACKEND_UNAVAILABLE,
        "Shadow runner is not available in its current lifecycle state.",
        "/model",
    )


def _invalid_item() -> SaracuraError:
    return SaracuraError(ErrorCode.REQUEST_INVALID, "Shadow item is invalid.", "/state")


def build_shadow_request(policy: ShadowPolicy, item: ShadowItem) -> DecisionRequest:
    """Build and structurally validate one closed request without backend access."""

    failed = False
    request: DecisionRequest | None = None
    try:
        validate_shadow_item_state(policy, item)
        request = DecisionRequest(
            api_version="v1alpha1",
            model=policy.model,
            mode="research",
            locale=policy.locale,
            domain=policy.domain,
            workflow=policy.workflow,
            state=dict(item.state),
            questions=(
                ChoiceQuestion(
                    id=policy.question_id,
                    type="choice",
                    instruction=policy.instruction,
                    criteria=policy.criteria,
                ),
            ),
        )
        validate_saracura_request_structure(request)
    except BaseException:
        failed = True
    if failed or request is None:
        raise _invalid_item()
    return request


class ShadowRunner:
    """Own one backend for a context-managed sequence of content-free decisions."""

    def __init__(
        self,
        policy: ShadowPolicy,
        backend_factory: Callable[[], UniversalBackend],
    ) -> None:
        if not isinstance(policy, ShadowPolicy) or not callable(backend_factory):
            raise _invalid_item()
        failed = False
        validated_policy: ShadowPolicy | None = None
        try:
            validated_policy = ShadowPolicy.model_validate(policy.model_dump(mode="python"))
        except BaseException:
            failed = True
        if failed or validated_policy is None:
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Shadow policy is invalid.",
                "/policy",
            )
        self._policy = validated_policy
        self._backend_factory = backend_factory
        self._backend: UniversalBackend | None = None
        self._engine: DecisionEngine | None = None
        self._entered = False
        self._closed = False

    def __enter__(self) -> ShadowRunner:
        if self._entered or self._closed:
            raise _unavailable()
        self._entered = True
        backend: UniversalBackend | None = None
        failed = False
        try:
            backend = self._backend_factory()
            prepare = getattr(backend, "prepare", None)
            if callable(prepare):
                prepare()
            if backend.model.revision != self._policy.model:
                failed = True
            else:
                self._backend = backend
                self._engine = DecisionEngine(
                    backend=backend,
                    workflows=default_workflows(),
                    calibrations={},
                )
        except BaseException:
            failed = True
        if failed:
            if backend is not None:
                self._close_backend(backend)
            self._closed = True
            raise _unavailable()
        return self

    def decide(self, item: ShadowItem) -> ShadowDecisionRecord:
        if not self._entered or self._closed or self._engine is None:
            raise _unavailable()
        request = build_shadow_request(self._policy, item)
        failed = False
        response: Any = None
        try:
            response = self._engine.decide(request, include_timing=False)
        except BaseException:
            failed = True
        if failed:
            raise _unavailable()
        failed = False
        record: ShadowDecisionRecord | None = None
        try:
            record = project_shadow_decision(
                self._policy,
                item.item_ref,
                response,
            )
        except BaseException:
            failed = True
        if failed or record is None:
            raise _unavailable()
        return record

    @staticmethod
    def _close_backend(backend: UniversalBackend) -> None:
        with suppress(BaseException):
            backend.close()  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        backend, self._backend = self._backend, None
        self._engine = None
        if backend is not None:
            self._close_backend(backend)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        self.close()
        return False
