from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

import saracura.shadow.runner as shadow_runner_module
from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import (
    Answer,
    ChoiceCriterion,
    DecisionResponse,
    ModelReference,
    Timing,
    Usage,
    WorkflowReference,
)
from saracura.serialization import canonical_json_bytes
from saracura.shadow import (
    ShadowDecisionRecord,
    ShadowFeedbackRecord,
    ShadowItem,
    ShadowPolicy,
    ShadowRunner,
    evaluate_shadow_feedback,
    parse_shadow_decision_json,
    parse_shadow_feedback_json,
    parse_shadow_item_json,
    parse_shadow_policy_json,
    project_shadow_decision,
    shadow_policy_digest,
    validate_shadow_item_state,
    validate_shadow_state_capacity,
)

MODEL_REVISION = "phase4e-saracura-ranker.v1.fixture"
LABELS = ("action_required", "finance", "manual_review")


class _FakeUniversalBackend:
    def __init__(self, *, fail_with: BaseException | None = None) -> None:
        self.prepared = 0
        self.closed = 0
        self.decisions = 0
        self.include_timing: list[bool] = []
        self.fail_with = fail_with

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=10,
            max_criteria=8,
            execution_boundary="synthetic-shadow-test",
            cold_warm_semantics="synthetic",
            quality_claims=False,
            dynamic_workflows=frozenset({("universal-choice", "phase4e-saracura-ranker.v1")}),
        )

    @property
    def model(self) -> ModelReference:
        return ModelReference(
            id="saracura/universal-ranker",
            revision=MODEL_REVISION,
            checkpoint_sha256="a" * 64,
        )

    def prepare(self) -> None:
        self.prepared += 1

    def validate_request(self, request: Any) -> None:
        del request

    def score_universal_choice(self, request: Any, question: Any, state: bytes) -> ScoredChoice:
        del request, state
        self.decisions += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ScoredChoice(
            question_id=question.id,
            raw_scores={
                criterion.id: float(len(question.criteria) - index)
                for index, criterion in enumerate(question.criteria)
            },
            input_tokens=5,
        )

    def close(self) -> None:
        self.closed += 1


def _policy(**updates: object) -> ShadowPolicy:
    value: dict[str, object] = {
        "schema_version": "saracura-shadow-policy.v1",
        "id": "email-triage",
        "revision": "pilot.v1",
        "model": MODEL_REVISION,
        "locale": "pt-BR",
        "domain": "email_triage",
        "workflow": WorkflowReference(id="universal-choice", revision="phase4e-saracura-ranker.v1"),
        "state_keys": ("subject", "preview"),
        "question_id": "classification",
        "instruction": "Classifique o item em uma categoria.",
        "criteria": (
            ChoiceCriterion(id=LABELS[0], description="Exige uma ação do operador."),
            ChoiceCriterion(id=LABELS[1], description="Trata de finanças ou contabilidade."),
            ChoiceCriterion(id=LABELS[2], description="Use quando houver ambiguidade."),
        ),
    }
    value.update(updates)
    return ShadowPolicy.model_validate(value)


def _item(item_ref: str = "msg-001", state: dict[str, str] | None = None) -> ShadowItem:
    return ShadowItem(
        schema_version="saracura-shadow-item.v1",
        item_ref=item_ref,
        state=state or {"subject": "Assunto fictício", "preview": "Prévia fictícia"},
    )


def _response(
    policy: ShadowPolicy | None = None,
    *,
    weights: dict[str, float] | None = None,
    model: ModelReference | None = None,
    value: str | None = None,
) -> DecisionResponse:
    selected_policy = policy or _policy()
    labels = tuple(criterion.id for criterion in selected_policy.criteria)
    scores = weights or {labels[0]: 0.6, labels[1]: 0.3, labels[2]: 0.1}
    selected = value or max(labels, key=lambda label: scores.get(label, 0.0))
    return DecisionResponse(
        api_version="v1alpha1",
        answers=(
            Answer(
                question_id=selected_policy.question_id,
                type="choice",
                value=selected,
                raw_scores={label: float(index) for index, label in enumerate(labels)},
                probabilities=scores,
                status="uncalibrated",
                score_semantics="ranking_weights",
                abstained=True,
                reason="uncalibrated_research",
                calibration=None,
            ),
        ),
        model=model
        or ModelReference(
            id="saracura/universal-ranker",
            revision=selected_policy.model,
            checkpoint_sha256="a" * 64,
        ),
        timing=Timing(total_ms=123.0),
        usage=Usage(input_tokens=19, questions=1, criteria=len(labels)),
        automation_allowed=False,
    )


def _decision(
    item_ref: str = "msg-001",
    policy: ShadowPolicy | None = None,
    *,
    weights: dict[str, float] | None = None,
    model: ModelReference | None = None,
) -> ShadowDecisionRecord:
    selected_policy = policy or _policy()
    return project_shadow_decision(
        selected_policy,
        item_ref,
        _response(selected_policy, weights=weights, model=model),
    )


def _feedback(
    item_ref: str = "msg-001",
    policy: ShadowPolicy | None = None,
    *,
    label: str | None = LABELS[0],
    disposition: Literal["labeled", "skipped"] = "labeled",
) -> ShadowFeedbackRecord:
    selected_policy = policy or _policy()
    return ShadowFeedbackRecord(
        schema_version="saracura-shadow-feedback.v1",
        item_ref=item_ref,
        policy_sha256=shadow_policy_digest(selected_policy),
        disposition=disposition,
        operator_label=label,
    )


def test_policy_is_frozen_closed_and_has_canonical_digest() -> None:
    policy = _policy()
    with pytest.raises(ValidationError):
        policy.id = "changed"
    with pytest.raises(ValidationError):
        ShadowPolicy.model_validate({**policy.model_dump(), "extra": "no"})
    raw = json.dumps(policy.model_dump(mode="json"), ensure_ascii=False)
    assert shadow_policy_digest(parse_shadow_policy_json(raw)) == shadow_policy_digest(policy)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("id", lambda value: value.__setitem__("id", "other-policy")),
        ("revision", lambda value: value.__setitem__("revision", "pilot.v2")),
        ("model", lambda value: value.__setitem__("model", "another.model.v1")),
        ("locale", lambda value: value.__setitem__("locale", "en")),
        ("domain", lambda value: value.__setitem__("domain", "finance")),
        (
            "workflow",
            lambda value: value.__setitem__(
                "workflow", {"id": "universal-choice", "revision": "phase4d-laya.v1"}
            ),
        ),
        ("state_keys", lambda value: value.__setitem__("state_keys", ["preview", "subject"])),
        (
            "question_id",
            lambda value: value.__setitem__("question_id", "other-question"),
        ),
        (
            "criterion id",
            lambda value: value["criteria"][0].__setitem__("id", "urgent_action"),
        ),
        (
            "criterion order",
            lambda value: value.__setitem__("criteria", list(reversed(value["criteria"]))),
        ),
        (
            "criterion description",
            lambda value: value["criteria"][0].__setitem__("description", "Nova descrição."),
        ),
        (
            "instruction",
            lambda value: value.__setitem__("instruction", "Outra instrução."),
        ),
    ],
)
def test_policy_digest_changes_for_every_bound_axis(field: str, mutate: object) -> None:
    original = _policy()
    payload = original.model_dump(mode="json")
    cast_mutate = mutate
    cast_mutate(payload)  # type: ignore[operator]
    if field == "workflow":
        original_payload = original.model_dump(mode="json")
        assert (
            sha256(canonical_json_bytes(payload)).digest()
            != sha256(canonical_json_bytes(original_payload)).digest()
        )
        with pytest.raises(ValidationError):
            ShadowPolicy.model_validate(payload)
        return
    changed = ShadowPolicy.model_validate(payload)
    assert shadow_policy_digest(changed) != shadow_policy_digest(original), field


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"saracura-shadow-policy.v1","schema_version":"x"}',
        '{"schema_version":"saracura-shadow-policy.v1","id":"bad/path"}',
        '{"schema_version":"saracura-shadow-policy.v1","id":"p","revision":"latest"}',
    ],
)
def test_policy_parser_rejects_duplicate_keys_alias_and_malformed_schema(payload: str) -> None:
    with pytest.raises(SaracuraError):
        parse_shadow_policy_json(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "latest"},
        {"revision": "master"},
        {"locale": "fr"},
        {"domain": "unknown_domain"},
        {"workflow": WorkflowReference(id="universal-choice", revision="unknown.v1")},
        {"state_keys": ("subject", "subject")},
        {"instruction": unicodedata.normalize("NFD", "ação")},
    ],
)
def test_policy_rejects_alias_unsupported_and_malformed_fields(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _policy(**change)


def test_policy_rejects_laya_control_workflow_revision() -> None:
    with pytest.raises(ValidationError):
        _policy(workflow=WorkflowReference(id="universal-choice", revision="phase4d-laya.v1"))


def test_policy_rejects_labels_with_duplicate_ids_and_oversized_text() -> None:
    policy = _policy()
    criteria = list(policy.criteria)
    criteria[1] = ChoiceCriterion(id=criteria[0].id, description=criteria[1].description)
    with pytest.raises(ValidationError):
        _policy(criteria=tuple(criteria))
    with pytest.raises(ValidationError):
        _policy(instruction="x" * 121)
    with pytest.raises(ValidationError):
        _policy(
            criteria=(
                ChoiceCriterion(id="a", description="ç" * 121),
                ChoiceCriterion(id="b", description="ok"),
            )
        )


def test_item_rejects_malformed_reference_and_accepts_hex_reference() -> None:
    assert _item(item_ref="a" * 64).item_ref == "a" * 64
    for item_ref in ("user@example.com", "folder/item", "../unsafe", "Upper"):
        with pytest.raises(ValidationError):
            _item(item_ref=item_ref)


def test_item_parser_rejects_non_nfc_and_non_string_state() -> None:
    decomposed = unicodedata.normalize("NFD", "prévia")
    raw = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "msg-1",
            "state": {"subject": "canary-subject", "preview": decomposed},
        },
        ensure_ascii=False,
    )
    with pytest.raises(SaracuraError) as captured:
        parse_shadow_item_json(raw)
    assert "canary-subject" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    with pytest.raises(ValidationError):
        _item(state={"subject": "valid", "preview": 123})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        ShadowItem(
            schema_version="saracura-shadow-item.v1",
            item_ref="msg-1",
            state={1: "not-a-string-key", "preview": "ok"},  # type: ignore[dict-item]
        )


def test_shadow_runner_owns_one_prepared_backend_and_preserves_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    backend = _FakeUniversalBackend()
    real_engine = cast(Any, shadow_runner_module).DecisionEngine
    include_timing_values: list[bool] = []

    class EngineSpy:
        def __init__(self, **kwargs: Any) -> None:
            self.delegate = real_engine(**kwargs)

        def decide(self, request: Any, *, include_timing: bool = True) -> Any:
            include_timing_values.append(include_timing)
            return self.delegate.decide(request, include_timing=include_timing)

    monkeypatch.setattr(shadow_runner_module, "DecisionEngine", EngineSpy)
    runner = ShadowRunner(policy, lambda: backend)
    with pytest.raises(SaracuraError):
        runner.decide(_item())

    with runner as active:
        records = [active.decide(_item(f"msg-{index:03d}")) for index in range(100)]
        assert [record.item_ref for record in records] == [f"msg-{i:03d}" for i in range(100)]
        assert all(record.status == "uncalibrated" for record in records)
        assert all(record.abstained and not record.automation_allowed for record in records)
    runner.close()

    assert backend.prepared == 1
    assert backend.closed == 1
    assert backend.decisions == 100
    assert include_timing_values == [False] * 100
    assert "subject" not in records[0].model_dump_json()


@pytest.mark.parametrize("failure", [ValueError("private-content-canary"), KeyboardInterrupt()])
def test_shadow_runner_redacts_engine_failures_and_closes_backend(failure: BaseException) -> None:
    backend = _FakeUniversalBackend(fail_with=failure)
    runner = ShadowRunner(_policy(), lambda: backend)
    caught: SaracuraError | None = None
    try:
        with runner as active:
            active.decide(_item())
    except SaracuraError as error:
        caught = error
    assert caught is not None
    assert caught.__cause__ is None
    assert caught.__context__ is None
    assert "private-content-canary" not in str(caught)
    assert backend.prepared == backend.closed == 1
    with pytest.raises(SaracuraError):
        runner.decide(_item())


def test_shadow_runner_preserves_only_sanitized_capacity_failure() -> None:
    private = SaracuraError(
        ErrorCode.CAPACITY_EXCEEDED,
        "private-input-canary at /private/input with 987654 tokens",
        "/private/input",
        details={"content": "private-content-canary"},
    )
    backend = _FakeUniversalBackend(fail_with=private)
    runner = ShadowRunner(_policy(), lambda: backend)
    caught: SaracuraError | None = None
    try:
        with runner as active:
            active.decide(_item())
    except SaracuraError as error:
        caught = error

    assert caught is not None
    assert caught.payload.code == ErrorCode.CAPACITY_EXCEEDED
    assert caught.payload.path == "/state"
    assert caught.payload.details == {}
    assert "private" not in str(caught)
    assert "987654" not in str(caught)
    assert caught.__cause__ is None
    assert caught.__context__ is None
    assert backend.prepared == backend.closed == 1
    with pytest.raises(SaracuraError):
        runner.decide(_item())


def test_shadow_runner_masks_non_capacity_contract_errors() -> None:
    backend = _FakeUniversalBackend(
        fail_with=SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "private-error-canary",
            "/private/input",
            details={"content": "private-content-canary"},
        )
    )
    runner = ShadowRunner(_policy(), lambda: backend)
    caught: SaracuraError | None = None
    try:
        with runner as active:
            active.decide(_item())
    except SaracuraError as error:
        caught = error

    assert caught is not None
    assert caught.payload.code == ErrorCode.BACKEND_UNAVAILABLE
    assert caught.payload.path == "/model"
    assert caught.payload.details == {}
    assert "private" not in str(caught)
    assert caught.__cause__ is None
    assert caught.__context__ is None
    assert backend.prepared == backend.closed == 1


def test_shadow_state_capacity_helper_owns_canonical_bounds_without_echo() -> None:
    validate_shadow_state_capacity({"subject": "ok", "preview": "ok"})

    with pytest.raises(SaracuraError) as overflow:
        validate_shadow_state_capacity({"subject": "x" * 201})
    assert overflow.value.payload.code == ErrorCode.CAPACITY_EXCEEDED
    assert overflow.value.payload.path == "/state"
    assert "x" * 20 not in str(overflow.value)
    assert overflow.value.__cause__ is None
    assert overflow.value.__context__ is None

    with pytest.raises(SaracuraError) as malformed:
        validate_shadow_state_capacity({"subject": 17})
    assert malformed.value.payload.code == ErrorCode.REQUEST_INVALID
    assert "17" not in str(malformed.value)


def test_shadow_runner_prepare_failure_closes_and_never_exposes_cause() -> None:
    class BrokenBackend(_FakeUniversalBackend):
        def prepare(self) -> None:
            self.prepared += 1
            raise ValueError("private-prepare-canary")

    backend = BrokenBackend()
    runner = ShadowRunner(_policy(), lambda: backend)
    caught: SaracuraError | None = None
    try:
        runner.__enter__()
    except SaracuraError as error:
        caught = error
    assert caught is not None
    assert caught.__cause__ is None
    assert caught.__context__ is None
    assert "private-prepare-canary" not in str(caught)
    assert backend.prepared == backend.closed == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"domain": "unregistered_domain"},
        {"workflow": WorkflowReference(id="universal-choice", revision="phase4d-laya.v1")},
    ],
)
def test_shadow_runner_revalidates_copied_policy_before_backend_factory(
    updates: dict[str, object],
) -> None:
    calls: list[bool] = []
    mutated_policy = _policy().model_copy(update=updates)

    def backend_factory() -> _FakeUniversalBackend:
        calls.append(True)
        return _FakeUniversalBackend()

    caught: SaracuraError | None = None
    try:
        runner = ShadowRunner(mutated_policy, backend_factory)
        runner.__enter__()
    except SaracuraError as error:
        caught = error
    assert caught is not None
    assert caught.payload.path == "/policy"
    assert caught.__cause__ is None
    assert caught.__context__ is None
    assert calls == []


def _run_optional_synthetic_torch_shadow_batch() -> None:
    torch = cast(Any, importlib.import_module("torch"))

    policy = _policy()

    class SyntheticTorchBackend(_FakeUniversalBackend):
        def prepare(self) -> None:
            self.prepared += 1
            self.layer = torch.nn.Linear(1, len(LABELS), bias=False)
            with torch.no_grad():
                self.layer.weight.copy_(torch.tensor([[1.0], [2.0], [3.0]]))

        def score_universal_choice(self, request: Any, question: Any, state: bytes) -> ScoredChoice:
            del state
            self.decisions += 1
            output = self.layer(torch.tensor([[1.0]])).detach().tolist()[0]
            return ScoredChoice(
                question_id=question.id,
                raw_scores=dict(
                    zip((criterion.id for criterion in question.criteria), output, strict=True)
                ),
                input_tokens=5,
            )

    backend = SyntheticTorchBackend()
    with ShadowRunner(policy, lambda: backend) as active:
        records = [active.decide(_item(f"syn-{index:02d}")) for index in range(10)]
    assert len(records) == 10
    assert backend.prepared == backend.closed == 1
    assert backend.decisions == 10


def test_optional_synthetic_torch_shadow_batch_loads_once() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("universal-local extra is not installed")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_shadow import _run_optional_synthetic_torch_shadow_batch; "
            "_run_optional_synthetic_torch_shadow_batch()",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ptbr_email_reference_examples_match_policy_and_stay_synthetic() -> None:
    root = Path(__file__).parents[1]
    policy = parse_shadow_policy_json(
        (root / "examples/ptbr-email-shadow-policy.json").read_bytes()
    )
    assert policy.model == (
        "phase4e-saracura-ranker.v1."
        "f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e"
    )
    assert policy.workflow.revision == "phase4e-saracura-ranker.v1"
    assert policy.domain == "email_triage"
    assert policy.locale == "pt-BR"
    assert policy.state_keys == ("subject", "preview")
    assert tuple(criterion.id for criterion in policy.criteria) == (
        "action_required",
        "financial_or_accounting",
        "legal_or_security",
        "project_or_operations",
        "marketing_or_newsletter",
        "low_value_or_spam",
        "manual_review",
    )
    assert len(policy.instruction) <= 120
    assert len(policy.instruction.encode("utf-8")) <= 480
    assert all(len(item.description) <= 120 for item in policy.criteria)
    assert all(len(item.description.encode("utf-8")) <= 480 for item in policy.criteria)

    input_path = root / "examples/ptbr-email-shadow-input.jsonl"
    input_lines = input_path.read_text(encoding="utf-8").splitlines()
    items = [parse_shadow_item_json(line) for line in input_lines]
    assert len(items) == 8
    for item in items:
        validate_shadow_item_state(policy, item)
        assert len(canonical_json_bytes(cast(Any, item.state)).decode("utf-8")) <= 200
        assert len(canonical_json_bytes(cast(Any, item.state))) <= 800
    refs = {item.item_ref for item in items}
    assert len(refs) == len(items)
    assert all("@" not in item.item_ref and "/" not in item.item_ref for item in items)
    assert not any("@" in line for line in input_lines)

    feedback_lines = (
        (root / "examples/ptbr-email-shadow-feedback.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    feedback = [parse_shadow_feedback_json(line) for line in feedback_lines]
    assert len(feedback) == 8
    assert {record.item_ref for record in feedback} == refs
    assert all(record.policy_sha256 == shadow_policy_digest(policy) for record in feedback)
    assert {record.operator_label for record in feedback if record.operator_label is not None} == {
        criterion.id for criterion in policy.criteria
    }
    assert sum(record.disposition == "skipped" for record in feedback) == 1


def test_shadow_json_parsers_enforce_closed_decision_and_feedback_schemas() -> None:
    decision = _decision()
    assert (
        parse_shadow_decision_json(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False))
        == decision
    )
    with pytest.raises(SaracuraError):
        parse_shadow_decision_json(
            json.dumps({**decision.model_dump(mode="json"), "raw_scores": {"secret": 1}})
        )
    feedback = _feedback()
    with pytest.raises(SaracuraError):
        parse_shadow_feedback_json(
            json.dumps({**feedback.model_dump(mode="json"), "extra": "canary"})
        )


@pytest.mark.parametrize(
    "state",
    [
        {"subject": "x"},
        {"subject": "x", "preview": "y", "private-extra-key": "canary"},
    ],
)
def test_policy_state_keys_must_match_exactly_without_echo(state: dict[str, str]) -> None:
    item = _item(state=state)
    with pytest.raises(SaracuraError) as captured:
        validate_shadow_item_state(_policy(), item)
    assert "private-extra-key" not in str(captured.value)
    assert "canary" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_state_non_string_and_nfc_values_fail_before_projection() -> None:
    decomposed = unicodedata.normalize("NFD", "ação")
    raw = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "msg-1",
            "state": {"subject": "ação", "preview": decomposed},
        },
        ensure_ascii=False,
    )
    with pytest.raises(SaracuraError):
        parse_shadow_item_json(raw)
    with pytest.raises(ValidationError):
        ShadowItem(
            schema_version="saracura-shadow-item.v1",
            item_ref="msg-1",
            state={"subject": "x", "preview": 1},  # type: ignore[dict-item]
        )


def test_decision_projection_is_ordered_content_free_and_research_only() -> None:
    policy = _policy()
    item = _item(state={"subject": "SUBJECT-CANARY", "preview": "PREVIEW-CANARY"})
    response = _response(
        policy,
        weights={LABELS[0]: 0.4, LABELS[1]: 0.4, LABELS[2]: 0.2},
    )
    record = project_shadow_decision(policy, item.item_ref, response)
    payload = record.model_dump(mode="json")
    assert record.suggested_label == LABELS[0]
    assert [entry.label for entry in record.ranking] == list(LABELS)
    assert record.status == "uncalibrated"
    assert record.score_semantics == "ranking_weights"
    assert record.abstained is True
    assert record.automation_allowed is False
    assert not {"SUBJECT-CANARY", "PREVIEW-CANARY"}.intersection(json.dumps(payload))
    assert "timing" not in payload and "usage" not in payload and "raw_scores" not in payload
    assert "confidence" not in json.dumps(payload).casefold()


@pytest.mark.parametrize(
    "change",
    [
        {
            "model": ModelReference(
                id="saracura/universal-ranker", revision="wrong.v1", checkpoint_sha256="a" * 64
            )
        },
        {"value": LABELS[1]},
        {"weights": {LABELS[0]: 1.0, "unapproved": 0.0, LABELS[2]: 0.0}},
    ],
)
def test_decision_projection_rejects_model_and_label_mismatch(change: dict[str, object]) -> None:
    policy = _policy()
    response = _response(policy, **change)  # type: ignore[arg-type]
    with pytest.raises(SaracuraError) as captured:
        project_shadow_decision(policy, "msg-1", response)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_decision_projection_sanitizes_invalid_reference_without_exception_context() -> None:
    with pytest.raises(SaracuraError) as captured:
        project_shadow_decision(_policy(), "canary/path", _response())
    assert "canary" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_decision_and_feedback_schemas_are_closed_and_dispositions_consistent() -> None:
    feedback = _feedback(disposition="skipped", label=None)
    assert feedback.operator_label is None
    with pytest.raises(ValidationError):
        ShadowFeedbackRecord.model_validate({**feedback.model_dump(), "extra": "no"})
    decision = _decision()
    with pytest.raises(ValidationError):
        ShadowDecisionRecord.model_validate({**decision.model_dump(), "extra": "no"})
    with pytest.raises(ValidationError):
        _feedback(disposition="skipped", label=LABELS[0])
    with pytest.raises(ValidationError):
        _feedback(disposition="labeled", label=None)
    with pytest.raises(SaracuraError):
        parse_shadow_feedback_json(
            '{"schema_version":"saracura-shadow-feedback.v1","item_ref":"msg-1","item_ref":"msg-2"}'
        )


def test_evaluator_calculates_coverage_agreement_and_closed_matrix() -> None:
    policy = _policy()
    decisions = [_decision(f"msg-{index}", policy) for index in range(4)]
    feedback = [
        _feedback("msg-0", policy, label=LABELS[0]),
        _feedback("msg-1", policy, label=LABELS[1]),
        _feedback("msg-2", policy, disposition="skipped", label=None),
    ]
    summary = evaluate_shadow_feedback(policy, decisions, feedback)
    assert summary.prediction_count == 4
    assert summary.feedback_count == 3
    assert summary.labeled_count == 2
    assert summary.skipped_count == 1
    assert summary.unreviewed_prediction_count == 1
    assert summary.feedback_coverage == 0.75
    assert summary.agreement_count == 1
    assert summary.agreement_rate == 0.5
    assert tuple(summary.confusion_matrix) == LABELS
    assert all(tuple(row) == LABELS for row in summary.confusion_matrix.values())
    assert summary.confusion_matrix[LABELS[0]][LABELS[0]] == 1
    assert summary.confusion_matrix[LABELS[0]][LABELS[1]] == 1
    assert sum(sum(row.values()) for row in summary.confusion_matrix.values()) == 2
    assert set(summary.model_dump(mode="json")) == {
        "schema_version",
        "policy_sha256",
        "prediction_count",
        "feedback_count",
        "labeled_count",
        "skipped_count",
        "unreviewed_prediction_count",
        "feedback_coverage",
        "agreement_count",
        "agreement_rate",
        "confusion_matrix",
    }
    assert not {"calibration", "training", "checkpoint", "confidence"}.intersection(
        summary.model_dump(mode="json")
    )
    with pytest.raises(ValidationError):
        type(summary).model_validate({**summary.model_dump(), "training_data": []})


def test_zero_labeled_feedback_has_null_agreement_rate() -> None:
    policy = _policy()
    summary = evaluate_shadow_feedback(
        policy,
        [_decision("msg-1", policy), _decision("msg-2", policy)],
        [_feedback("msg-1", policy, disposition="skipped", label=None)],
    )
    assert summary.labeled_count == 0
    assert summary.agreement_count == 0
    assert summary.agreement_rate is None
    assert summary.feedback_coverage == 0.5


def test_zero_predictions_has_zero_coverage_and_full_empty_matrix() -> None:
    policy = _policy()
    summary = evaluate_shadow_feedback(policy, [], [])
    assert summary.prediction_count == 0
    assert summary.feedback_coverage == 0.0
    assert summary.agreement_rate is None
    assert tuple(summary.confusion_matrix) == LABELS
    assert all(sum(row.values()) == 0 for row in summary.confusion_matrix.values())


def test_summary_rejects_agreement_count_that_disagrees_with_matrix_diagonal() -> None:
    policy = _policy()
    summary = evaluate_shadow_feedback(
        policy,
        [_decision("msg-1", policy)],
        [_feedback("msg-1", policy, label=LABELS[0])],
    )
    assert summary.confusion_matrix[LABELS[0]][LABELS[0]] == 1
    payload = summary.model_dump()
    payload["agreement_count"] = 0
    payload["agreement_rate"] = 0.0
    with pytest.raises(ValidationError, match="matrix agreement"):
        type(summary).model_validate(payload)


@pytest.mark.parametrize("duplicate_type", ("decision", "feedback"))
def test_evaluator_rejects_duplicate_references(duplicate_type: str) -> None:
    policy = _policy()
    decision = _decision("same", policy)
    feedback = _feedback("same", policy)
    decisions = [decision, decision] if duplicate_type == "decision" else [decision]
    feedback_records = [feedback, feedback] if duplicate_type == "feedback" else []
    with pytest.raises(SaracuraError, match="Duplicate"):
        evaluate_shadow_feedback(policy, decisions, feedback_records)


def test_evaluator_rejects_mixed_wrong_model_or_orphan_feedback() -> None:
    policy = _policy()
    first = _decision("msg-1", policy)
    mixed_model = ModelReference(
        id="saracura/universal-ranker", revision=policy.model, checkpoint_sha256="b" * 64
    )
    second = _decision("msg-2", policy, model=mixed_model)
    with pytest.raises(SaracuraError, match="mixed model"):
        evaluate_shadow_feedback(policy, [first, second], [])
    wrong_model = ModelReference(
        id="saracura/universal-ranker", revision="wrong.v1", checkpoint_sha256="a" * 64
    )
    with pytest.raises(SaracuraError, match="does not match"):
        evaluate_shadow_feedback(policy, [_decision("msg-3", policy, model=wrong_model)], [])
    with pytest.raises(SaracuraError, match="no prediction"):
        evaluate_shadow_feedback(policy, [first], [_feedback("orphan", policy)])


def test_evaluator_rejects_wrong_policy_digest_and_unknown_label() -> None:
    policy = _policy()
    decision = _decision("msg-1", policy)
    wrong_digest = _feedback("msg-1", policy).model_copy(update={"policy_sha256": "0" * 64})
    with pytest.raises(SaracuraError, match="does not match"):
        evaluate_shadow_feedback(policy, [decision], [wrong_digest])
    unknown_label = _feedback("msg-1", policy).model_copy(update={"operator_label": "unknown"})
    with pytest.raises(SaracuraError, match="outside the policy"):
        evaluate_shadow_feedback(policy, [decision], [unknown_label])


def test_evaluation_is_deterministic_and_matrix_is_full() -> None:
    policy = _policy()
    decisions = [_decision(f"msg-{index}", policy) for index in range(3)]
    feedback = [_feedback(f"msg-{index}", policy, label=LABELS[index]) for index in range(3)]
    first = evaluate_shadow_feedback(policy, decisions, feedback)
    second = evaluate_shadow_feedback(policy, decisions, feedback)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert set(first.confusion_matrix) == set(LABELS)
    assert all(set(row) == set(LABELS) for row in first.confusion_matrix.values())


def test_default_shadow_import_does_not_load_optional_ml_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import saracura.shadow; "
            "assert not any(name == 'torch' or name.startswith('torch.') or "
            "name == 'transformers' or name.startswith('transformers.') or "
            "name == 'tokenizers' or name.startswith('tokenizers.') or "
            "name == 'safetensors' or name.startswith('safetensors.') "
            "for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
