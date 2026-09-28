"""Public Choice fixture → pinned Kev wire-shape conversion.

Every public ChoiceQuestion becomes one entry under ``questions[question.id]``
with ``type: choice``, ``instructions`` equal to the public instruction and
``criteria`` mapped from criterion id to description. The request includes
``model: kev-latest``. Historical list-shaped fake responses are not evidence.
"""

from __future__ import annotations

from typing import Any

from saracura.contracts.models import DecisionRequest


def convert_fixture_to_wire(request: DecisionRequest) -> dict[str, Any]:
    """Convert a public DecisionRequest into the pinned Kev wire shape."""
    questions: dict[str, dict[str, Any]] = {}
    for question in request.questions:
        criteria_map: dict[str, str] = {
            criterion.id: criterion.description for criterion in question.criteria
        }
        questions[question.id] = {
            "type": "choice",
            "instructions": question.instruction,
            "criteria": criteria_map,
        }
    return {
        "model": "kev-latest",
        "questions": questions,
        "state": request.state,
    }
