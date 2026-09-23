from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import saracura.cli as cli
from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.cli import _run_decide
from saracura.contracts import ErrorCode, ModelReference, SaracuraError
from saracura.runtime import known_scaling_questions
from tests.helpers import decision_request


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

    monkeypatch.setattr(cli, "LayaUniversalBackend", FakeLaya)
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

    monkeypatch.setattr(cli, "LayaUniversalBackend", ConstructorSentinel)
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

    monkeypatch.setattr(cli, "LayaUniversalBackend", FakeLaya)
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
