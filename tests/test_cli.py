from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import saracura.cli as cli
from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.cli import _run_decide
from saracura.contracts import (
    ChoiceCriterion,
    ErrorCode,
    ModelReference,
    SaracuraError,
    WorkflowReference,
)
from saracura.runtime import known_scaling_questions
from saracura.shadow import (
    ShadowDecisionRecord,
    ShadowFeedbackRecord,
    ShadowPolicy,
    ShadowRankingEntry,
    shadow_policy_digest,
)
from tests.helpers import decision_request

_SHADOW_MODEL_REVISION = "phase4e-saracura-ranker.v1.fixture"
_SHADOW_LABELS = ("action_required", "finance", "manual_review")


def _shadow_policy() -> ShadowPolicy:
    return ShadowPolicy(
        schema_version="saracura-shadow-policy.v1",
        id="email-triage",
        revision="pilot.v1",
        model=_SHADOW_MODEL_REVISION,
        locale="pt-BR",
        domain="email_triage",
        workflow=WorkflowReference(id="universal-choice", revision="phase4e-saracura-ranker.v1"),
        state_keys=("subject", "preview"),
        question_id="classification",
        instruction="Classifique o item em uma categoria.",
        criteria=(
            ChoiceCriterion(id=_SHADOW_LABELS[0], description="Exige uma ação do operador."),
            ChoiceCriterion(id=_SHADOW_LABELS[1], description="Trata de finanças."),
            ChoiceCriterion(id=_SHADOW_LABELS[2], description="Use quando houver ambiguidade."),
        ),
    )


def _decision_record(policy: ShadowPolicy, item_ref: str) -> ShadowDecisionRecord:
    return ShadowDecisionRecord(
        schema_version="saracura-shadow-decision.v1",
        item_ref=item_ref,
        policy_sha256=shadow_policy_digest(policy),
        model=ModelReference(
            id="saracura/universal-ranker",
            revision=policy.model,
            checkpoint_sha256="a" * 64,
        ),
        suggested_label=_SHADOW_LABELS[0],
        ranking=(
            ShadowRankingEntry(label=_SHADOW_LABELS[0], ranking_weight=0.6),
            ShadowRankingEntry(label=_SHADOW_LABELS[1], ranking_weight=0.3),
            ShadowRankingEntry(label=_SHADOW_LABELS[2], ranking_weight=0.1),
        ),
        status="uncalibrated",
        score_semantics="ranking_weights",
        abstained=True,
        reason="uncalibrated_research",
        automation_allowed=False,
    )


def test_invalid_laya_cli_path_does_not_import_optional_ml_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import saracura.cli as cli; "
                "assert cli.main(['decide', '--backend', 'laya-universal']) == 2; "
                "assert not any(name.split('.')[0] in {'torch', 'transformers', "
                "'tokenizers', 'safetensors', 'psutil', 'platformdirs'} for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_maps_io_failure_without_exposing_local_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).parents[1]
    calibration = root / "examples/ptbr-support-calibration.json"

    exit_code = _run_decide(tmp_path, calibration, include_timing=False)

    stderr = capsys.readouterr().err
    payload = json.loads(stderr)
    assert exit_code == 2
    assert payload["error"]["code"] == "INTERNAL_ERROR"
    assert str(tmp_path) not in stderr


def test_cli_never_falls_back_from_an_identifiable_invalid_v2_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeMiniLM:
        pass

    request = decision_request(known_scaling_questions(1))
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"schema_version":2}\n', encoding="utf-8")
    monkeypatch.setattr(cli, "MiniLMRoutingBackend", FakeMiniLM)
    monkeypatch.setattr(
        cli,
        "load_calibration_envelope",
        lambda _path: (_ for _ in ()).throw(
            SaracuraError(
                ErrorCode.CALIBRATION_INCOMPATIBLE,
                "Calibration artifact failed schema validation.",
                "/calibration",
            )
        ),
    )

    assert _run_decide(request_path, calibration, False, backend=cast(Any, FakeMiniLM())) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "CALIBRATION_INCOMPATIBLE"


def test_cli_request_fifo_is_rejected_without_blocking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public request reader must apply O_NONBLOCK before the regular-file check."""

    request = tmp_path / "request.json"
    os.mkfifo(request, 0o600)
    calibration = Path(__file__).parents[1] / "examples/ptbr-support-calibration.json"

    assert _run_decide(request, calibration, include_timing=False) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("failure", ("request", "v2-receipt"))
def test_decide_preflight_rejects_unsafe_inputs_before_minilm_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    """Neither a rejected request nor a rejected v2 receipt may touch MiniLM."""

    constructed: list[object] = []

    class ConstructorSentinel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(object())
            raise AssertionError("MiniLM constructor must not run during preflight")

    request_path = tmp_path / "request.json"
    calibration_path = tmp_path / "calibration.json"
    if failure == "request":
        request_path.write_text('{"api_version":"wrong"}', encoding="utf-8")
    else:
        request = decision_request(known_scaling_questions(1))
        request_path.write_text(request.model_dump_json(), encoding="utf-8")
    calibration_path.write_text('{"schema_version":2}\n', encoding="utf-8")
    calibration_path.chmod(0o600)
    monkeypatch.setattr(cli, "MiniLMRoutingBackend", ConstructorSentinel)

    exit_code = cli.main(
        [
            "decide",
            "--backend",
            "minilm-routing",
            "--request",
            str(request_path),
            "--calibration",
            str(calibration_path),
            "--encoder-snapshot",
            str(tmp_path / "snapshot"),
            "--training-manifest",
            str(tmp_path / "training-manifest.json"),
            "--checkpoint",
            str(tmp_path / "checkpoint.safetensors"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 2
    assert not constructed
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] in {
        ErrorCode.REQUEST_INVALID.value,
        ErrorCode.API_VERSION_UNSUPPORTED.value,
        ErrorCode.CALIBRATION_INCOMPATIBLE.value,
    }


@pytest.mark.parametrize("alias", ("latest", "main", "master"))
def test_decide_rejects_model_alias_before_minilm_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    alias: str,
) -> None:
    constructed: list[object] = []

    class ConstructorSentinel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(object())
            raise AssertionError("MiniLM constructor must not run for an alias")

    request = decision_request(known_scaling_questions(1)).model_copy(update={"model": alias})
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(cli, "MiniLMRoutingBackend", ConstructorSentinel)

    exit_code = cli.main(
        [
            "decide",
            "--backend",
            "minilm-routing",
            "--request",
            str(request_path),
            "--calibration",
            str(tmp_path / "not-read.json"),
            "--encoder-snapshot",
            str(tmp_path / "snapshot"),
            "--training-manifest",
            str(tmp_path / "training-manifest.json"),
            "--checkpoint",
            str(tmp_path / "checkpoint.safetensors"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 2
    assert not constructed
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.MODEL_ALIAS_FORBIDDEN


def test_laya_cli_prevalidates_then_emits_abstained_response_and_closes_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    instances: list[Any] = []

    class FakeLaya:
        def __init__(self, *, model_snapshot: Path, device: str) -> None:
            assert model_snapshot == tmp_path / "snapshot"
            assert device == "cpu"
            self.closed = False
            instances.append(self)

        @property
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(
                execution_tier="universal",
                decision_types=frozenset({"choice"}),
                max_questions=10,
                max_criteria=20,
                execution_boundary="test",
                cold_warm_semantics="test",
                quality_claims=False,
                dynamic_workflows=frozenset({("universal-choice", "phase4d-laya.v1")}),
            )

        @property
        def model(self) -> ModelReference:
            return ModelReference(
                id="convaiinnovations/laya-multilingual",
                revision="052592a15d198d9ad47da779604259b10b47b7aa",
                checkpoint_sha256="9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
            )

        def validate_request(self, request: Any) -> None:
            del request

        def score_universal_choice(self, request: Any, question: Any, state: bytes) -> ScoredChoice:
            del request, state
            return ScoredChoice(
                question_id=question.id,
                raw_scores={
                    criterion.id: float(index) for index, criterion in enumerate(question.criteria)
                },
                input_tokens=17,
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(cli, "_load_laya_backend", lambda: FakeLaya)
    example = Path(__file__).parents[1] / "examples/ptbr-universal-request.json"

    assert (
        cli.main(
            [
                "decide",
                "--backend",
                "laya-universal",
                "--request",
                str(example),
                "--model-snapshot",
                str(tmp_path / "snapshot"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["automation_allowed"] is False
    assert payload["answers"][0]["status"] == "uncalibrated"
    assert payload["answers"][0]["abstained"] is True
    assert payload["answers"][0]["calibration"] is None
    assert len(instances) == 1
    assert instances[0].closed is True


def test_laya_cli_rejects_bad_request_and_exclusive_arguments_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    constructed: list[object] = []

    class ConstructorSentinel:
        def __init__(self, **_kwargs: object) -> None:
            constructed.append(object())

    monkeypatch.setattr(cli, "_load_laya_backend", lambda: ConstructorSentinel)
    bad_request = tmp_path / "bad-request.json"
    bad_request.write_text('{"api_version":"wrong"}', encoding="utf-8")
    common = [
        "decide",
        "--backend",
        "laya-universal",
        "--request",
        str(bad_request),
        "--model-snapshot",
        str(tmp_path / "snapshot"),
        "--device",
        "cpu",
    ]

    assert cli.main(common) == 2
    assert not constructed
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.API_VERSION_UNSUPPORTED

    example = Path(__file__).parents[1] / "examples/ptbr-universal-request.json"
    assert (
        cli.main(
            [
                "decide",
                "--backend",
                "laya-universal",
                "--request",
                str(example),
                "--model-snapshot",
                str(tmp_path / "snapshot"),
                "--device",
                "cpu",
                "--calibration",
                "no.json",
            ]
        )
        == 2
    )
    assert not constructed
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.REQUEST_INVALID


def test_laya_describe_backend_reports_contract_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    closed: list[bool] = []

    class FakeLaya:
        def __init__(self, **_kwargs: object) -> None:
            pass

        @property
        def model(self) -> ModelReference:
            return ModelReference(
                id="convaiinnovations/laya-multilingual",
                revision="052592a15d198d9ad47da779604259b10b47b7aa",
                checkpoint_sha256="9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
            )

        def prepare(self) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(cli, "_load_laya_backend", lambda: FakeLaya)
    assert (
        cli.main(
            [
                "describe-backend",
                "--backend",
                "laya-universal",
                "--model-snapshot",
                str(tmp_path / "snapshot"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_tier"] == "universal"
    assert payload["workflow"] == {
        "criteria_per_question": {"maximum": 20, "minimum": 2},
        "id": "universal-choice",
        "locales": ["pt-BR", "en"],
        "question_type": "choice",
        "questions": {"maximum": 10, "minimum": 1},
        "revision": "phase4d-laya.v1",
    }
    assert payload["response"]["score_semantics"] == "ranking_weights"
    assert payload["runtime_disposition"] == "research_only_unresolved_provenance"
    assert closed == [True]


class _ShadowCliBackend:
    def __init__(self, *, fail_on_call: int | None = None, failure: BaseException | None = None):
        self.prepared = 0
        self.closed = 0
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.failure = failure

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=10,
            max_criteria=8,
            execution_boundary="shadow-cli-test",
            cold_warm_semantics="synthetic",
            quality_claims=False,
            dynamic_workflows=frozenset({("universal-choice", "phase4e-saracura-ranker.v1")}),
        )

    @property
    def model(self) -> ModelReference:
        return ModelReference(
            id="saracura/universal-ranker",
            revision=_SHADOW_MODEL_REVISION,
            checkpoint_sha256="a" * 64,
        )

    def prepare(self) -> None:
        self.prepared += 1

    def validate_request(self, request: Any) -> None:
        del request

    def score_universal_choice(self, request: Any, question: Any, state: bytes) -> ScoredChoice:
        del request, state
        self.calls += 1
        if self.fail_on_call == self.calls and self.failure is not None:
            raise self.failure
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


def _configure_shadow_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdin: bytes,
    backend: _ShadowCliBackend,
) -> tuple[ShadowPolicy, Path]:
    policy = _shadow_policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(stdin)))
    monkeypatch.setattr(cli, "load_saracura_candidate", lambda: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "verify_training_capsule",
        lambda _path, _candidate: SimpleNamespace(
            candidate=SimpleNamespace(model_revision=policy.model)
        ),
    )
    monkeypatch.setattr(cli, "_require_saracura_arguments", lambda *_args, **_kwargs: backend)
    return policy, policy_path


def _shadow_cli_args(policy_path: Path, tmp_path: Path) -> list[str]:
    return [
        "shadow-decide",
        "--policy",
        str(policy_path),
        "--encoder-snapshot",
        str(tmp_path / "snapshot"),
        "--training-capsule",
        str(tmp_path / "capsule"),
        "--device",
        "cpu",
    ]


def test_shadow_decide_cli_runs_100_items_in_order_with_one_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = [
        json.dumps(
            {
                "schema_version": "saracura-shadow-item.v1",
                "item_ref": f"msg-{index:03d}",
                "state": {"subject": "Fictional subject", "preview": "Fictional preview"},
            },
            ensure_ascii=False,
        )
        for index in range(100)
    ]
    backend = _ShadowCliBackend()
    policy, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, ("\n".join(lines) + "\n").encode(), backend
    )

    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 0

    output = capsys.readouterr().out.splitlines()
    records = [json.loads(line) for line in output]
    assert [record["item_ref"] for record in records] == [
        f"msg-{index:03d}" for index in range(100)
    ]
    assert all(record["policy_sha256"] == shadow_policy_digest(policy) for record in records)
    assert all(record["automation_allowed"] is False for record in records)
    assert backend.prepared == backend.closed == 1
    assert backend.calls == 100


def test_shadow_decide_cli_prevalidates_full_batch_and_emits_no_partial_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    valid = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "msg-1",
            "state": {"subject": "ok", "preview": "ok"},
        }
    )
    invalid = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "msg-2",
            "state": {"unknown-sensitive-key": "private-canary"},
        }
    )
    backend = _ShadowCliBackend()
    _policy_value, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, f"{valid}\n{invalid}\n".encode(), backend
    )

    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert captured.out == ""
    assert payload["error"]["details"]["line_index"] == 1
    assert "private-canary" not in captured.err
    assert "unknown-sensitive-key" not in captured.err
    assert backend.prepared == backend.closed == backend.calls == 0


@pytest.mark.parametrize("failure", [ValueError("private-runtime-canary"), KeyboardInterrupt()])
def test_shadow_decide_cli_runtime_failure_closes_and_discards_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
) -> None:
    lines = [
        json.dumps(
            {
                "schema_version": "saracura-shadow-item.v1",
                "item_ref": f"msg-{index}",
                "state": {"subject": "canary-subject", "preview": "canary-preview"},
            }
        )
        for index in range(4)
    ]
    backend = _ShadowCliBackend(fail_on_call=3, failure=failure)
    _policy_value, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, ("\n".join(lines) + "\n").encode(), backend
    )

    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert captured.out == ""
    assert payload["error"]["details"]["line_index"] == 2
    assert "canary" not in captured.err
    assert "private-runtime-canary" not in captured.err
    assert backend.prepared == backend.closed == 1
    assert backend.calls == 3


def test_shadow_decide_cli_prepare_failure_reports_first_line_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class BrokenPrepareBackend(_ShadowCliBackend):
        def prepare(self) -> None:
            self.prepared += 1
            raise ValueError("private-prepare-canary")

    item = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "msg-1",
            "state": {"subject": "ok", "preview": "ok"},
        }
    )
    backend = BrokenPrepareBackend()
    _policy_value, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, f"{item}\n".encode(), backend
    )
    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["details"]["line_index"] == 0
    assert "private-prepare-canary" not in captured.err
    assert backend.prepared == backend.closed == 1
    assert backend.calls == 0


def test_shadow_decide_cli_caps_jsonl_lines_and_rejects_duplicate_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    item = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "same-ref",
            "state": {"subject": "ok", "preview": "ok"},
        }
    )
    backend = _ShadowCliBackend()
    _policy_value, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, f"{item}\n{item}\n".encode(), backend
    )
    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["details"]["line_index"] == 1
    assert backend.prepared == 0


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"saracura-shadow-item.v1","item_ref":"x","state":{}}\xff\n',
        b'{"schema_version":"saracura-shadow-item.v1","item_ref":"x","state":{"subject":"s","preview":"p"}}'
        + b" " * 4000
        + b"\n",
    ],
)
def test_shadow_decide_cli_rejects_invalid_utf8_and_oversized_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes,
) -> None:
    backend = _ShadowCliBackend()
    _policy_value, policy_path = _configure_shadow_cli(tmp_path, monkeypatch, payload, backend)
    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["details"]["line_index"] == 0
    assert backend.prepared == 0


def test_shadow_decide_cli_rejects_501st_item_at_zero_based_line_index_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        json.dumps(
            {
                "schema_version": "saracura-shadow-item.v1",
                "item_ref": f"msg-{index}",
                "state": {"subject": "s", "preview": "p"},
            }
        )
        for index in range(501)
    ]
    backend = _ShadowCliBackend()
    _policy_value, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, ("\n".join(rows) + "\n").encode(), backend
    )
    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["details"]["line_index"] == 500
    assert backend.prepared == 0


def test_shadow_decide_cli_preflights_capsule_before_backend_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    item = json.dumps(
        {
            "schema_version": "saracura-shadow-item.v1",
            "item_ref": "msg-1",
            "state": {"subject": "ok", "preview": "ok"},
        }
    )
    backend = _ShadowCliBackend()
    policy, policy_path = _configure_shadow_cli(
        tmp_path, monkeypatch, f"{item}\n".encode(), backend
    )
    monkeypatch.setattr(
        cli,
        "verify_training_capsule",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("private-capsule-canary")),
    )

    assert cli.main(_shadow_cli_args(policy_path, tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private-capsule-canary" not in captured.err
    assert policy.model == _SHADOW_MODEL_REVISION
    assert backend.prepared == backend.closed == backend.calls == 0


def test_shadow_evaluate_cli_reads_only_local_closed_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _shadow_policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        _decision_record(policy, "msg-1").model_dump_json() + "\n", encoding="utf-8"
    )
    feedback_record = ShadowFeedbackRecord(
        schema_version="saracura-shadow-feedback.v1",
        item_ref="msg-1",
        policy_sha256=shadow_policy_digest(policy),
        disposition="labeled",
        operator_label=_SHADOW_LABELS[0],
    )
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(feedback_record.model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_require_saracura_arguments",
        lambda *_args, **_kwargs: pytest.fail("model backend must not be constructed"),
    )

    assert (
        cli.main(
            [
                "shadow-evaluate",
                "--policy",
                str(policy_path),
                "--decisions",
                str(decisions),
                "--feedback",
                str(feedback),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["prediction_count"] == 1
    assert summary["agreement_count"] == 1
    assert "calibration" not in summary


def test_shadow_evaluate_cli_rejects_symlinks_and_decide_rejects_other_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = _shadow_policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(decisions)
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text("", encoding="utf-8")

    assert (
        cli.main(
            [
                "shadow-evaluate",
                "--policy",
                str(policy_path),
                "--decisions",
                str(link),
                "--feedback",
                str(feedback),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.REQUEST_INVALID

    assert (
        cli.main(["shadow-decide", "--policy", str(policy_path), "--calibration", "no.json"]) == 2
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.REQUEST_INVALID


def test_shadow_evaluate_cli_rejects_duplicate_keys_and_oversized_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(_shadow_policy().model_dump_json(), encoding="utf-8")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        '{"schema_version":"saracura-shadow-decision.v1","schema_version":"bad"}\n',
        encoding="utf-8",
    )
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text("", encoding="utf-8")
    args = [
        "shadow-evaluate",
        "--policy",
        str(policy_path),
        "--decisions",
        str(decisions),
        "--feedback",
        str(feedback),
    ]
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == ErrorCode.REQUEST_INVALID

    decisions.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == ErrorCode.REQUEST_INVALID


def test_shadow_cli_help_and_invalid_arguments_do_not_import_optional_ml() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import saracura.cli as cli; "
            "assert cli.main(['shadow-decide', '--calibration', 'x']) == 2; "
            "assert not any(name.split('.')[0] in {'torch', 'transformers', 'tokenizers', "
            "'safetensors'} "
            "for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
