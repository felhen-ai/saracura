from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

import saracura.cli as cli
from saracura.cli import _run_decide
from saracura.contracts import ErrorCode, SaracuraError
from saracura.runtime import known_scaling_questions
from tests.helpers import decision_request


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
