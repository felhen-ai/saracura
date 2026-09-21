from __future__ import annotations

import json
from pathlib import Path

import pytest

from saracura.cli import _run_decide


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
