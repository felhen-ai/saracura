"""Direct, opt-in Julia-1 backend for the experimental public preview."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import (
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    ModelReference,
)

JULIA_MODEL_ID = "supersoniclabs/julia-1"
JULIA_MODEL_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
JULIA_CHECKPOINT_SHA256 = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
JULIA_WORKFLOW_ID = "universal-choice"
JULIA_WORKFLOW_REVISION = "phase5c-julia.v1"
JULIA_POSITION_ENSEMBLE_WORKFLOW_REVISION = "phase5d1-julia-cyclic-mean.v1"
JULIA_POSITION_ENSEMBLE_MAX_CRITERIA = 20


class _JuliaEngine(Protocol):
    def predict(
        self,
        rows: object | None = None,
        questions: Mapping[str, object] | None = None,
        *,
        state: object | None = None,
    ) -> object: ...


class JuliaBackend:
    """Load the pinned Julia-1 checkpoint directly from a local snapshot."""

    def __init__(
        self,
        *,
        model_snapshot: Path,
        device: str = "cpu",
        position_ensemble: bool = False,
    ) -> None:
        if device != "cpu":
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "The Julia public preview currently supports CPU only.",
                "/backend",
            )
        weight_path = model_snapshot / "model.safetensors"
        try:
            with weight_path.open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise SaracuraError(
                ErrorCode.MODEL_NOT_FOUND,
                "The pinned Julia-1 checkpoint was not found.",
                "/model",
            ) from error
        if observed != JULIA_CHECKPOINT_SHA256:
            raise SaracuraError(
                ErrorCode.MODEL_NOT_FOUND,
                "The Julia-1 checkpoint digest does not match the pinned release.",
                "/model",
            )
        self._model_snapshot = model_snapshot
        self._position_ensemble = position_ensemble
        self._engine: _JuliaEngine | None = None

    @property
    def workflow_revision(self) -> str:
        return (
            JULIA_POSITION_ENSEMBLE_WORKFLOW_REVISION
            if self._position_ensemble
            else JULIA_WORKFLOW_REVISION
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=10,
            # Let the workflow validator report the ensemble's aggregate
            # request budget at /questions before enforcing the 20-item total.
            max_criteria=21 if self._position_ensemble else 20,
            execution_boundary="direct-local-julia-1-cpu",
            cold_warm_semantics="resident-after-first-request",
            quality_claims=False,
            dynamic_workflows=frozenset({(JULIA_WORKFLOW_ID, self.workflow_revision)}),
        )

    @property
    def model(self) -> ModelReference:
        return ModelReference(
            id=JULIA_MODEL_ID,
            revision=JULIA_MODEL_REVISION,
            checkpoint_sha256=JULIA_CHECKPOINT_SHA256,
        )

    def validate_request(self, request: DecisionRequest) -> None:
        if (request.workflow.id, request.workflow.revision) != (
            JULIA_WORKFLOW_ID,
            self.workflow_revision,
        ):
            raise SaracuraError(
                ErrorCode.WORKFLOW_UNSUPPORTED,
                f"Julia accepts only universal-choice@{self.workflow_revision}.",
                "/workflow",
            )
        if (
            self._position_ensemble
            and sum(len(question.criteria) for question in request.questions)
            > JULIA_POSITION_ENSEMBLE_MAX_CRITERIA
        ):
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Julia position ensemble supports at most 20 criteria per request.",
                "/questions",
            )
        if request.locale not in {"pt-BR", "en"}:
            raise SaracuraError(
                ErrorCode.LOCALE_UNVERIFIED,
                "The Julia preview accepts only pt-BR or en.",
                "/locale",
            )

    def prepare(self) -> None:
        if self._engine is not None:
            return
        try:
            module = importlib.import_module("julia")
            load_model = cast(Callable[..., object], module.load_model)
            engine = load_model(
                self._model_snapshot,
                device="cpu",
                strict_encoding=True,
                max_length=8192,
                head_length=512,
            )
        except (ImportError, AttributeError) as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Install the pinned Julia-1 repository before using this backend.",
                "/model",
            ) from error
        except Exception as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Julia-1 could not be loaded from the pinned local snapshot.",
                "/model",
            ) from error
        self._engine = cast(_JuliaEngine, engine)

    def close(self) -> None:
        self._engine = None

    def score_universal_choice(
        self,
        request: DecisionRequest,
        question: ChoiceQuestion,
        state_payload: bytes,
    ) -> ScoredChoice:
        del state_payload
        self.validate_request(request)
        criteria = {criterion.id: criterion.description for criterion in question.criteria}
        try:
            if not self._position_ensemble:
                self.prepare()
                assert self._engine is not None
                result = self._engine.predict(
                    state=request.state,
                    questions={
                        question.id: {
                            "type": "choice",
                            "instructions": question.instruction,
                            "criteria": criteria,
                        }
                    },
                )
                answer = _choice_answer(result, question)
            else:
                answer = self._cyclic_mean(request, question)
        except SaracuraError:
            raise
        except Exception as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Julia-1 inference failed.",
                f"/questions/{question.id}",
            ) from error
        return ScoredChoice(
            question_id=question.id,
            raw_scores=dict(answer["probabilities"]),
            normalized_probabilities=dict(answer["probabilities"]),
            selected_choice=cast(str, answer["choice"]),
            input_tokens=0,
        )

    def _cyclic_mean(self, request: DecisionRequest, question: ChoiceQuestion) -> dict[str, Any]:
        self.prepare()
        assert self._engine is not None
        criteria = [(criterion.id, criterion.description) for criterion in question.criteria]
        collected: dict[str, list[float]] = {criterion_id: [] for criterion_id, _ in criteria}
        for offset in range(len(criteria)):
            rotated = criteria[offset:] + criteria[:offset]
            rotated_question = ChoiceQuestion(
                id=question.id,
                type="choice",
                instruction=question.instruction,
                criteria=tuple(
                    ChoiceCriterion(id=criterion_id, description=description)
                    for criterion_id, description in rotated
                ),
            )
            result = self._engine.predict(
                state=request.state,
                questions={
                    question.id: {
                        "type": "choice",
                        "instructions": question.instruction,
                        "criteria": dict(rotated),
                    }
                },
            )
            answer = _choice_answer(result, rotated_question)
            for criterion_id, probability in answer["probabilities"].items():
                collected[criterion_id].append(probability)
        means = {
            criterion_id: math.fsum(values) / len(criteria)
            for criterion_id, values in collected.items()
        }
        total = math.fsum(means.values())
        if not math.isfinite(total) or total <= 0:
            raise _invalid_response(question)
        normalized = {criterion_id: value / total for criterion_id, value in means.items()}
        selected = max(
            (criterion.id for criterion in question.criteria),
            key=normalized.__getitem__,
        )
        return {"choice": selected, "probabilities": normalized}


def _choice_answer(result: object, question: ChoiceQuestion) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != {"answers"}:
        raise _invalid_response(question)
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != {question.id}:
        raise _invalid_response(question)
    answer = answers.get(question.id)
    if not isinstance(answer, dict) or set(answer) != {
        "type",
        "probabilities",
        "choice",
        "max_probability",
    }:
        raise _invalid_response(question)
    if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str):
        raise _invalid_response(question)
    probabilities = answer.get("probabilities")
    expected = {criterion.id for criterion in question.criteria}
    if not isinstance(probabilities, dict) or set(probabilities) != expected:
        raise _invalid_response(question)
    normalized: dict[str, float] = {}
    for key, value in probabilities.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise _invalid_response(question)
        normalized[key] = float(value)
    if abs(math.fsum(normalized.values()) - 1.0) > 1e-12:
        raise _invalid_response(question)
    choice = answer["choice"]
    if choice not in normalized:
        raise _invalid_response(question)
    maximum = max(normalized.values())
    if abs(normalized[choice] - maximum) > 1e-12:
        raise _invalid_response(question)
    return {"choice": choice, "probabilities": normalized}


def _invalid_response(question: ChoiceQuestion) -> SaracuraError:
    return SaracuraError(
        ErrorCode.BACKEND_UNAVAILABLE,
        "Julia-1 returned an invalid typed-decision response.",
        f"/questions/{question.id}",
    )
