from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from saracura.backends import julia as julia_backend
from saracura.backends.julia import JULIA_MODEL_REVISION, JuliaBackend
from saracura.contracts import SaracuraError, parse_request_json
from saracura.runtime import DecisionEngine, default_workflows


class _FakeJuliaEngine:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        rows: object | None = None,
        questions: object | None = None,
        *,
        state: object | None = None,
    ) -> object:
        self.calls.append({"rows": rows, "questions": questions, "state": state})
        return self.result


def _request() -> Any:
    return parse_request_json(
        """
        {
          "api_version": "v1alpha1",
          "model": "a85b127321d580d65176c89ced8273f305745d85",
          "mode": "research",
          "locale": "pt-BR",
          "domain": "suporte",
          "workflow": {"id": "universal-choice", "revision": "phase5c-julia.v1"},
          "state": {"mensagem": "Fui cobrado duas vezes."},
          "questions": [{
            "id": "equipe",
            "type": "choice",
            "instruction": "Qual equipe deve tratar?",
            "criteria": [
              {"id": "cobranca", "description": "Cobranças e estornos"},
              {"id": "entrega", "description": "Entrega de pedidos"}
            ]
          }]
        }
        """
    )


def _backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JuliaBackend:
    content = b"pinned-test-weights"
    (tmp_path / "model.safetensors").write_bytes(content)
    monkeypatch.setattr(
        julia_backend,
        "JULIA_CHECKPOINT_SHA256",
        hashlib.sha256(content).hexdigest(),
    )
    return JuliaBackend(model_snapshot=tmp_path)


def test_julia_backend_runs_choice_without_importing_ml_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch)
    fake = _FakeJuliaEngine(
        {
            "answers": {
                "equipe": {
                    "type": "choice",
                    "probabilities": {"cobranca": 0.9, "entrega": 0.1},
                    "choice": "cobranca",
                    "max_probability": 0.9,
                }
            }
        }
    )
    backend._engine = fake
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(_request())

    assert response.model.revision == JULIA_MODEL_REVISION
    assert response.answers[0].value == "cobranca"
    assert response.answers[0].status == "uncalibrated"
    assert response.answers[0].abstained is True
    assert response.automation_allowed is False
    assert fake.calls[0]["state"] == {"mensagem": "Fui cobrado duas vezes."}


def test_julia_backend_rejects_wrong_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(SaracuraError):
        JuliaBackend(model_snapshot=tmp_path)


def test_julia_backend_rejects_unknown_output_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch)
    backend._engine = _FakeJuliaEngine(
        {
            "answers": {
                "equipe": {
                    "type": "choice",
                    "probabilities": {"cobranca": 0.9, "entrega": 0.1},
                    "choice": "cobranca",
                    "max_probability": 0.9,
                    "unexpected": True,
                }
            }
        }
    )
    with pytest.raises(SaracuraError):
        DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations={},
        ).decide(_request())


def test_julia_backend_is_choice_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    assert backend.capabilities.decision_types == frozenset({"choice"})
    assert backend.capabilities.max_criteria == 20
