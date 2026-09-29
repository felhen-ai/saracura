from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from saracura.backends import julia as julia_backend
from saracura.backends.julia import JULIA_MODEL_REVISION, JuliaBackend
from saracura.contracts import (
    ChoiceCriterion,
    SaracuraError,
    WorkflowReference,
    parse_request_json,
)
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


def _backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    position_ensemble: bool = False,
) -> JuliaBackend:
    content = b"pinned-test-weights"
    (tmp_path / "model.safetensors").write_bytes(content)
    monkeypatch.setattr(
        julia_backend,
        "JULIA_CHECKPOINT_SHA256",
        hashlib.sha256(content).hexdigest(),
    )
    return JuliaBackend(model_snapshot=tmp_path, position_ensemble=position_ensemble)


def _ensemble_request(count: int = 4, *, revision: str = "phase5d1-julia-cyclic-mean.v1") -> Any:
    request = _request()
    question = request.questions[0].model_copy(
        update={
            "criteria": tuple(
                ChoiceCriterion(id=f"choice-{index:02d}", description=f"Option {index}")
                for index in range(count)
            )
        }
    )
    return request.model_copy(
        update={
            "workflow": WorkflowReference(id="universal-choice", revision=revision),
            "questions": (question,),
        }
    )


class _PositionEngine:
    def __init__(self, *, fail_at: int | None = None, uniform: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_at = fail_at
        self.uniform = uniform

    def predict(
        self,
        rows: object | None = None,
        questions: object | None = None,
        *,
        state: object | None = None,
    ) -> object:
        assert isinstance(questions, dict)
        question_id, payload = next(iter(questions.items()))
        assert isinstance(payload, dict)
        criteria = payload["criteria"]
        assert isinstance(criteria, dict)
        ordered_ids = list(criteria)
        self.calls.append({"criteria": ordered_ids, "state": state})
        if self.fail_at == len(self.calls):
            return {"invalid": True}
        if self.uniform:
            probabilities = {criterion_id: 1 / len(ordered_ids) for criterion_id in ordered_ids}
        else:
            raw = [float(len(ordered_ids) - position) for position in range(len(ordered_ids))]
            total = sum(raw)
            probabilities = {
                criterion_id: value / total
                for criterion_id, value in zip(ordered_ids, raw, strict=True)
            }
        choice = max(ordered_ids, key=probabilities.__getitem__)
        maximum = probabilities[choice]
        return {
            "answers": {
                question_id: {
                    "type": "choice",
                    "probabilities": probabilities,
                    "choice": choice,
                    "max_probability": maximum,
                }
            }
        }


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
    assert len(fake.calls) == 1


@pytest.mark.parametrize("criterion_count", (2, 4, 20))
def test_position_ensemble_runs_each_cyclic_rotation_and_remaps_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    criterion_count: int,
) -> None:
    backend = _backend(tmp_path, monkeypatch, position_ensemble=True)
    backend._engine = _PositionEngine()
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(_ensemble_request(criterion_count))

    engine = backend._engine
    assert isinstance(engine, _PositionEngine)
    original = [
        criterion.id for criterion in _ensemble_request(criterion_count).questions[0].criteria
    ]
    assert [call["criteria"] for call in engine.calls] == [
        original[offset:] + original[:offset] for offset in range(criterion_count)
    ]
    assert len(engine.calls) == criterion_count
    answer = response.answers[0]
    assert list(answer.raw_scores) == original
    assert answer.raw_scores == answer.probabilities
    assert sum(answer.raw_scores.values()) == pytest.approx(1.0)
    assert answer.value == original[0]


def test_position_ensemble_breaks_exact_ties_by_original_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch, position_ensemble=True)
    backend._engine = _PositionEngine(uniform=True)
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(_ensemble_request(4))
    assert response.answers[0].value == "choice-00"


def test_position_ensemble_averages_by_original_criterion_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SequenceEngine:
        def __init__(self) -> None:
            self.calls = 0

        def predict(
            self,
            rows: object | None = None,
            questions: object | None = None,
            *,
            state: object | None = None,
        ) -> object:
            del rows, state
            assert isinstance(questions, dict)
            question_id = next(iter(questions))
            self.calls += 1
            probabilities = (
                {"choice-00": 0.8, "choice-01": 0.2}
                if self.calls == 1
                else {"choice-01": 0.7, "choice-00": 0.3}
            )
            choice = "choice-00" if self.calls == 1 else "choice-01"
            return {
                "answers": {
                    question_id: {
                        "type": "choice",
                        "probabilities": probabilities,
                        "choice": choice,
                        "max_probability": probabilities[choice],
                    }
                }
            }

    backend = _backend(tmp_path, monkeypatch, position_ensemble=True)
    backend._engine = SequenceEngine()
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(_ensemble_request(2))
    assert response.answers[0].raw_scores == pytest.approx({"choice-00": 0.55, "choice-01": 0.45})
    assert response.answers[0].probabilities == response.answers[0].raw_scores


def test_position_ensemble_fails_atomically_on_invalid_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch, position_ensemble=True)
    backend._engine = _PositionEngine(fail_at=2)
    with pytest.raises(SaracuraError, match="invalid typed-decision"):
        DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations={},
        ).decide(_ensemble_request(4))
    engine = backend._engine
    assert isinstance(engine, _PositionEngine)
    assert len(engine.calls) == 2


@pytest.mark.parametrize(
    ("position_ensemble", "revision"),
    [
        (False, "phase5d1-julia-cyclic-mean.v1"),
        (True, "phase5c-julia.v1"),
    ],
)
def test_julia_backend_binds_strategy_and_workflow_both_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    position_ensemble: bool,
    revision: str,
) -> None:
    backend = _backend(tmp_path, monkeypatch, position_ensemble=position_ensemble)
    backend._engine = _PositionEngine()
    with pytest.raises(SaracuraError) as error:
        DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations={},
        ).decide(_ensemble_request(4, revision=revision))
    assert error.value.payload.code.value == "WORKFLOW_UNSUPPORTED"


def test_position_ensemble_rejects_over_twenty_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch, position_ensemble=True)
    fake = _PositionEngine()
    backend._engine = fake
    with pytest.raises(SaracuraError) as error:
        DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations={},
        ).decide(_ensemble_request(21))
    assert error.value.payload.code.value == "CARDINALITY_EXCEEDED"
    assert error.value.payload.path == "/questions"
    assert fake.calls == []


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
