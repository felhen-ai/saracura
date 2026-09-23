"""Offline contracts for the planned, unsupported Phase 4E universal ranker."""

from __future__ import annotations

import json
import math
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

import saracura.cli as cli
from benchmarks.data_policy_registry import bundled_registry_path, load_registry
from benchmarks.saracura_universal_policy import Phase4EPolicyError, validate_phase4e_policy
from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts import (
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    ErrorCode,
    ModelReference,
    SaracuraError,
    WorkflowReference,
)
from saracura.runtime import DecisionEngine, default_workflows
from saracura.runtime.workflows import (
    SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION,
    UNIVERSAL_CHOICE_WORKFLOW_ID,
    UNIVERSAL_CHOICE_WORKFLOW_REVISION,
)
from saracura.universal.checkpoint import (
    BASE_ENCODER_ID,
    BASE_ENCODER_REVISION,
    CHECKPOINT_ARCHITECTURE_REVISION,
    CheckpointDescriptor,
    CheckpointTensor,
)
from saracura.universal.ranker import (
    RankerParameters,
    SaracuraUniversalRanker,
    Vector,
    VectorProjection,
)
from saracura.universal.rendering import (
    CapacityError,
    RenderedTask,
    render_task,
    rendered_bytes,
    validate_rendered_capacity,
)
from saracura.universal.tasks import UniversalTask

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-saracura-universal-policy.v1.json"


def _task(*, count: int = 2, locale: Literal["pt-BR", "en"] = "pt-BR") -> UniversalTask:
    return UniversalTask(
        task_id="task-1",
        family_id="family-1",
        locale=locale,
        domain="email_triage",
        instruction="Escolha a rota segura." if locale == "pt-BR" else "Choose the safe route.",
        state={"subject": "Cobrança" if locale == "pt-BR" else "Billing", "priority": 1},
        criteria=tuple(
            ChoiceCriterion(
                id=f"option-{index}",
                description=(
                    "Cobrança ou pagamento." if index < 2 else f"Critério repetido {index}."
                ),
            )
            for index in range(count)
        ),
        selected_criterion_id="option-0",
    )


def _parse_frames(payload: bytes) -> tuple[bytes, ...]:
    fields: list[bytes] = []
    position = 0
    while position < len(payload):
        colon = payload.index(b":", position)
        size_bytes = payload[position:colon]
        assert size_bytes.isdigit()
        size = int(size_bytes)
        start = colon + 1
        end = start + size
        fields.append(payload[start:end])
        position = end
    return tuple(fields)


class _Counter:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def count(self, text: str) -> int:
        return self._counts.get(text, 1)


class _FakeEncoder:
    def __init__(self, *, invalid: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._invalid = invalid

    def encode(self, inputs: Sequence[str]) -> tuple[Vector, ...]:
        self.calls.append(tuple(inputs))
        if self._invalid:
            return ((math.inf, 0.0), *((1.0, 0.0) for _ in inputs[1:]))
        return tuple((1.0, 0.0) if index % 2 == 0 else (0.0, 1.0) for index, _ in enumerate(inputs))


def _parameters(*, log_scale: float = 0.0) -> RankerParameters:
    projection = VectorProjection(
        weight=((1.0, 0.0), (0.0, 1.0)),
        bias=(0.0, 0.0),
        layernorm_weight=(1.0, 1.0),
        layernorm_bias=(0.0, 0.0),
    )
    return RankerParameters(
        context_projection=projection,
        criterion_projection=projection,
        log_scale=log_scale,
    )


def _checkpoint_payload() -> dict[str, Any]:
    tensors = {
        "context_projection.weight": (192, 384),
        "context_projection.bias": (192,),
        "context_projection.layernorm.weight": (192,),
        "context_projection.layernorm.bias": (192,),
        "criterion_projection.weight": (192, 384),
        "criterion_projection.bias": (192,),
        "criterion_projection.layernorm.weight": (192,),
        "criterion_projection.layernorm.bias": (192,),
        "log_scale": (1,),
    }
    return {
        "architecture_revision": CHECKPOINT_ARCHITECTURE_REVISION,
        "base_encoder_id": BASE_ENCODER_ID,
        "base_encoder_revision": BASE_ENCODER_REVISION,
        "base_encoder_snapshot_sha256": "a" * 64,
        "training_manifest_sha256": "b" * 64,
        "packet_sha256": "c" * 64,
        "code_sha256": "d" * 64,
        "tensors": [
            {"name": name, "shape": shape, "dtype": "F32"} for name, shape in tensors.items()
        ],
    }


class _FakeUniversalBackend:
    def __init__(self, workflows: frozenset[tuple[str, str]]) -> None:
        self.validate_calls = 0
        self.score_calls = 0
        self._capabilities = BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=10,
            max_criteria=20,
            execution_boundary="phase4e-fake-only",
            cold_warm_semantics="no-load",
            quality_claims=False,
            dynamic_workflows=workflows,
        )
        self._model = ModelReference(
            id="phase4e-fake",
            revision="phase4e-fake-v1",
            checkpoint_sha256="e" * 64,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    def validate_request(self, request: DecisionRequest) -> None:
        del request
        self.validate_calls += 1

    def score_universal_choice(
        self,
        request: DecisionRequest,
        question: ChoiceQuestion,
        state_payload: bytes,
    ) -> ScoredChoice:
        del request, state_payload
        self.score_calls += 1
        return ScoredChoice(
            question_id=question.id,
            raw_scores={
                criterion.id: float(index) for index, criterion in enumerate(question.criteria)
            },
        )


def _request(revision: str) -> DecisionRequest:
    return DecisionRequest(
        api_version="v1alpha1",
        model="phase4e-fake-v1",
        mode="research",
        locale="pt-BR",
        domain="email-triage",
        workflow=WorkflowReference(id=UNIVERSAL_CHOICE_WORKFLOW_ID, revision=revision),
        state={"subject": "Teste sintético"},
        questions=(
            ChoiceQuestion(
                id="triage",
                type="choice",
                instruction="Escolha uma rota.",
                criteria=(
                    ChoiceCriterion(id="route-a", description="Rota A."),
                    ChoiceCriterion(id="route-b", description="Rota B."),
                ),
            ),
        ),
    )


def test_phase4e_policy_and_phase3b_exception_are_closed_and_bound() -> None:
    policy = validate_phase4e_policy()
    registry = load_registry(bundled_registry_path().read_bytes())
    phase3b = registry.exceptions[0].model_dump(mode="json")
    synthetic_policy = json.loads(
        (ROOT / "benchmarks/manifests/synthetic-research-policy.v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert policy["source_policy_exception_id"] == registry.exceptions[1].id
    assert phase3b["workflow_revision"] == synthetic_policy["workflow_revision"]
    assert phase3b["author_model"] == synthetic_policy["author_model"]
    assert phase3b["reviewer_model"] == synthetic_policy["reviewer_model"]
    assert phase3b["canonical_training_authorized"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("provider", "host"), "example.test"),
        (("capacity", "criteria_max"), 9),
        (("budget", "total_usd"), 2.01),
        (("authorizations", "runtime_registration_authorized"), True),
        (("planning", "domains"), []),
    ],
)
def test_phase4e_policy_rejects_any_reviewed_axis_change(
    tmp_path: Path, path: tuple[str, str], replacement: object
) -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = replacement
    candidate = tmp_path / "policy.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase4EPolicyError):
        validate_phase4e_policy(candidate)


def test_phase4e_policy_and_registry_reject_duplicate_or_unknown_exceptions(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        POLICY_PATH.read_text(encoding="utf-8").replace(
            '"id": "phase4e-saracura-universal-synthetic",',
            '"id": "phase4e-saracura-universal-synthetic", "id": "other",',
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase4EPolicyError):
        validate_phase4e_policy(duplicate)

    registry_payload = json.loads(bundled_registry_path().read_text(encoding="utf-8"))
    registry_payload["exceptions"][1]["id"] = "unknown_exception"
    with pytest.raises(ValueError, match="invalid data policy registry"):
        load_registry(json.dumps(registry_payload).encode())

    registry_payload = json.loads(bundled_registry_path().read_text(encoding="utf-8"))
    registry_payload["exceptions"][0]["canonical_training_authorized"] = True
    with pytest.raises(ValueError, match="invalid data policy registry"):
        load_registry(json.dumps(registry_payload).encode())

    registry_payload = json.loads(bundled_registry_path().read_text(encoding="utf-8"))
    registry_payload["exceptions"][1]["spend_ceiling_usd"] = 2
    with pytest.raises(ValueError, match="invalid data policy registry"):
        load_registry(json.dumps(registry_payload).encode())


@pytest.mark.parametrize("locale", ["pt-BR", "en"])
def test_renderer_preserves_utf8_nfc_order_and_duplicate_descriptions(
    locale: Literal["pt-BR", "en"],
) -> None:
    task = _task(count=2, locale=locale).model_copy(
        update={
            "criteria": (
                ChoiceCriterion(id="first", description="mesma descrição"),
                ChoiceCriterion(id="second", description="mesma descrição"),
            )
        }
    )
    rendered = render_task(task)
    context, first, second = (
        _parse_frames(rendered_bytes(rendered)[0]),
        _parse_frames(rendered_bytes(rendered)[1]),
        _parse_frames(rendered_bytes(rendered)[2]),
    )

    assert "Cobrança" in rendered.context or "Billing" in rendered.context
    assert context[0] != b""
    assert first != second
    assert b"first" in first and b"second" in second
    assert rendered.criteria[0] != rendered.criteria[1]


def test_renderer_frames_delimiters_and_rejects_non_nfc_or_capacity_overflow() -> None:
    task = _task().model_copy(update={"instruction": "a:7:|b", "state": {"field:1": "v:2"}})
    rendered = render_task(task)
    fields = _parse_frames(rendered_bytes(rendered)[0])
    assert b"a:7:|b" in fields
    assert b'{"field:1":"v:2"}' in fields

    with pytest.raises(ValidationError, match="NFC"):
        UniversalTask.model_validate(
            {**_task().model_dump(mode="json"), "instruction": "Cafe\u0301"}
        )
    with pytest.raises(ValidationError, match="NFC"):
        UniversalTask.model_validate(
            {**_task().model_dump(mode="json"), "state": {"Cafe\u0301": "value"}}
        )

    counter = _Counter({rendered.context: 129})
    with pytest.raises(CapacityError, match="context"):
        validate_rendered_capacity(rendered, counter)
    counter = _Counter({rendered.criteria[0]: 97})
    with pytest.raises(CapacityError, match="criterion"):
        validate_rendered_capacity(rendered, counter)
    validate_rendered_capacity(
        RenderedTask(context="context", criteria=tuple("criterion" for _ in range(20))),
        _Counter({}),
    )
    with pytest.raises(CapacityError, match="batch"):
        validate_rendered_capacity(
            RenderedTask(context="context", criteria=tuple("criterion" for _ in range(21))),
            _Counter({}),
        )


def test_task_rejects_closed_instruction_and_state_capacity_overflow() -> None:
    payload = _task().model_dump(mode="json")
    payload["instruction"] = "a" * 121
    with pytest.raises(ValidationError):
        UniversalTask.model_validate(payload)

    payload = _task().model_dump(mode="json")
    payload["state"] = {"value": "a" * 201}
    with pytest.raises(ValidationError, match="state exceeds"):
        UniversalTask.model_validate(payload)


@pytest.mark.parametrize("count", [2, 8])
def test_ranker_scores_only_declared_variable_options_in_order(count: int) -> None:
    task = _task(count=count)
    encoder = _FakeEncoder()
    scores = SaracuraUniversalRanker(encoder, _parameters()).score(task)

    assert tuple(scores) == tuple(criterion.id for criterion in task.criteria)
    assert len(scores) == count
    assert encoder.calls == [(render_task(task).context, *render_task(task).criteria)]
    assert all(math.isfinite(value) for value in scores.values())


def test_ranker_rejects_nonfinite_and_mismatched_encoder_vectors() -> None:
    with pytest.raises(ValueError, match="encoder embedding"):
        SaracuraUniversalRanker(_FakeEncoder(invalid=True), _parameters()).score(_task())
    with pytest.raises(ValueError, match="log scale"):
        _parameters(log_scale=4.1)
    with pytest.raises(ValueError, match="finite"):
        VectorProjection(
            weight=((math.inf, 0.0), (0.0, 1.0)),
            bias=(0.0, 0.0),
            layernorm_weight=(1.0, 1.0),
            layernorm_bias=(0.0, 0.0),
        )


def test_checkpoint_descriptor_accepts_only_closed_saracura_tensor_set() -> None:
    descriptor = CheckpointDescriptor.model_validate(_checkpoint_payload())
    assert len(descriptor.tensors) == 9
    assert {tensor.name for tensor in descriptor.tensors} == {
        "context_projection.weight",
        "context_projection.bias",
        "context_projection.layernorm.weight",
        "context_projection.layernorm.bias",
        "criterion_projection.weight",
        "criterion_projection.bias",
        "criterion_projection.layernorm.weight",
        "criterion_projection.layernorm.bias",
        "log_scale",
    }

    payload = _checkpoint_payload()
    payload["tensors"][0]["name"] = "encoder.layer.weight"
    with pytest.raises(ValidationError, match="closed"):
        CheckpointDescriptor.model_validate(payload)
    payload = _checkpoint_payload()
    payload["tensors"][0]["name"] = "laya.option_marker.weight"
    with pytest.raises(ValidationError, match="closed"):
        CheckpointDescriptor.model_validate(payload)
    payload = _checkpoint_payload()
    payload["tensors"][0]["shape"] = [1, 1]
    with pytest.raises(ValidationError, match="shape"):
        CheckpointDescriptor.model_validate(payload)
    with pytest.raises(ValidationError):
        CheckpointTensor.model_validate({"name": "log_scale", "shape": [1], "dtype": "F16"})


@pytest.mark.parametrize(
    ("backend_revision", "request_revision"),
    [
        (UNIVERSAL_CHOICE_WORKFLOW_REVISION, SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION),
        (SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION, UNIVERSAL_CHOICE_WORKFLOW_REVISION),
    ],
)
def test_crossed_universal_backend_workflow_revisions_fail_before_backend_access(
    backend_revision: str, request_revision: str
) -> None:
    backend = _FakeUniversalBackend(frozenset({(UNIVERSAL_CHOICE_WORKFLOW_ID, backend_revision)}))
    engine = DecisionEngine(backend=backend, workflows=default_workflows(), calibrations={})

    with pytest.raises(SaracuraError) as captured:
        engine.decide(_request(request_revision))

    assert captured.value.payload.code == ErrorCode.WORKFLOW_UNSUPPORTED
    assert backend.validate_calls == backend.score_calls == 0


def test_laya_cli_prevalidation_rejects_phase4e_workflow_before_candidate_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "phase4e-request.json"
    request_path.write_text(
        _request(SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION).model_dump_json(),
        encoding="utf-8",
    )

    def candidate_access_forbidden() -> object:
        raise AssertionError("Laya candidate access attempted")

    monkeypatch.setattr(cli, "load_laya_candidate", candidate_access_forbidden)
    with pytest.raises(SaracuraError) as captured:
        cli._prevalidate_request(request_path, execution_tier="universal")

    assert captured.value.payload.code == ErrorCode.WORKFLOW_UNSUPPORTED


def test_phase4e_workflow_contract_has_the_lower_cardinality_before_backend_access() -> None:
    backend = _FakeUniversalBackend(
        frozenset({(UNIVERSAL_CHOICE_WORKFLOW_ID, SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION)})
    )
    template = _request(SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION).questions[0]
    expanded = template.model_copy(
        update={
            "criteria": tuple(
                ChoiceCriterion(id=f"route-{index}", description=f"Route {index}.")
                for index in range(9)
            )
        }
    )
    request = _request(SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION).model_copy(
        update={"questions": (expanded,)}
    )

    with pytest.raises(SaracuraError) as captured:
        DecisionEngine(backend=backend, workflows=default_workflows(), calibrations={}).decide(
            request
        )

    assert captured.value.payload.code == ErrorCode.CARDINALITY_EXCEEDED
    assert backend.validate_calls == backend.score_calls == 0


def test_phase4e_contract_paths_are_network_and_optional_ml_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_network(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(socket, "getaddrinfo", blocked_network)
    validate_phase4e_policy()
    SaracuraUniversalRanker(_FakeEncoder(), _parameters()).score(_task())
    default_workflows().validate(
        _request(SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION),
        "universal",
        _FakeUniversalBackend(
            frozenset({(UNIVERSAL_CHOICE_WORKFLOW_ID, SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION)})
        ).capabilities,
    )

    assert not {
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
        "sentence_transformers",
    }.intersection(sys.modules)
