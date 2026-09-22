"""Closed, source-bound recovery-plan validation and strict response parsing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from benchmarks.universal_remote.plan import ENDPOINT, MODEL, materialize_requests

ROOT = Path(__file__).parents[2]
PLAN_PATH = ROOT / "benchmarks/manifests/phase4c3c-resume-plan.v1.json"
PREDECESSOR_PLAN = ROOT / "benchmarks/manifests/universal-bakeoff-plan.v3.json"
FIRST_ORDINAL, LAST_ORDINAL = 10, 56
BODY_CEILING = 1_048_576
SOURCE_COMMIT = "a461a5eba12a2777e73fb1e0ccb259bebf8d4cb5"
PREDECESSOR_PLAN_SHA256 = "9e74bf6116670a0f394d7201bfe861b377321f6de43c42cb2d14173b1af4546e"
SOURCES = {
    "benchmarks/universal_remote/plan.py": (
        "b3e61795a3669d739692b18d340b282641c073030120bdfb5491b35d1dd9fbcd"
    ),
    "benchmarks/universal_remote/runner.py": (
        "757a9e5d52453e60c342bfc8495bb6c3704c73627a6e89802c93cfc64b4dc507"
    ),
    "benchmarks/universal_local/plan.py": (
        "1d6dcb486f4c41ffc3f852e113a52023c407e078e94b49f8fa2cda3799f865a5"
    ),
    "benchmarks/universal_local/registry.py": (
        "b1be2bbf0b7bef841a0af8b4b1e9b487a7b53ed153a5166d98ed4a85e9a22f0f"
    ),
    "benchmarks/universal_local/inputs.py": (
        "4d161ec8e03858fe4fa2c348ad67710d3704bd8bbdd23d005ef3e5c4493c69d2"
    ),
    "benchmarks/manifests/universal-bakeoff-plan.v1.json": (
        "535165cdf6eacc3bd73bafecaf2ef2c82050890d3525045667623177614be08d"
    ),
    "benchmarks/manifests/universal-bakeoff-plan.v2.json": (
        "b140377ab0c58d3f33e9c4c5a2598dfc58de3659ab9d5c980c8f7a7f57b64ae9"
    ),
    "benchmarks/manifests/decision-backend-candidates.v2.json": (
        "b7fae06d959334a605ceb663d09728bb03c2095d632ccbc1d54bfd56c0469ac1"
    ),
}
DIAGNOSTICS = frozenset(
    {
        "json_invalid",
        "root_shape",
        "answer_key_mismatch",
        "usage_shape",
        "usage_invalid",
        "answer_shape",
        "answer_type_mismatch",
        "choice_unknown",
        "probability_key_mismatch",
        "probability_invalid",
        "probability_sum_out_of_tolerance",
        "choice_not_maximal",
        "confidence_invalid",
        "unexpected_response_type",
        "validator_internal_error",
        "response_too_large",
    }
)


class ResponseError(ValueError):
    """Private validation exception whose public value is a closed diagnostic."""

    def __init__(self, category: str) -> None:
        if category not in DIAGNOSTICS and category != "model_drift":
            raise ValueError("invalid diagnostic")
        self.category = category
        super().__init__(category)


def _dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_dupes,
            parse_float=Decimal,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
    except (RecursionError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ResponseError("json_invalid") from error


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)
    )


def _git_bytes(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("pinned Git object unavailable") from error


def validate_plan(raw: bytes | None = None) -> dict[str, Any]:
    """Validate every immutable recovery binding, including frozen wire bytes."""
    try:
        value = json.loads(
            (PLAN_PATH.read_bytes() if raw is None else raw).decode("utf-8"),
            object_pairs_hook=_dupes,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid recovery plan JSON") from error
    required = {
        "schema_version",
        "plan_id",
        "evidence_lane",
        "predecessor_plan_sha256",
        "provider",
        "endpoint",
        "model",
        "first_request_ordinal",
        "last_request_ordinal",
        "continuation_request_count",
        "request_attempts",
        "timeout_seconds",
        "response_body_ceiling_bytes",
        "reservation_cost_usd",
        "computed_rate_usd_per_million",
        "cumulative_budget_usd",
        "probability_sum_absolute_tolerance",
        "source_commit",
        "sources",
        "required_partial_contract",
        "wire_sha256_by_ordinal",
        "diagnostic_categories",
        "quality_labels",
        "conclusion_authority",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("recovery plan root is not closed")
    literals = {
        "schema_version": "phase4c3c-resume-plan.v1",
        "plan_id": "phase4c3c-typesafe-response-contract-recovery",
        "evidence_lane": "synthetic_research",
        "predecessor_plan_sha256": PREDECESSOR_PLAN_SHA256,
        "provider": "typesafe_native",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "first_request_ordinal": FIRST_ORDINAL,
        "last_request_ordinal": LAST_ORDINAL,
        "continuation_request_count": 47,
        "request_attempts": 1,
        "timeout_seconds": 60,
        "response_body_ceiling_bytes": BODY_CEILING,
        "reservation_cost_usd": "0.0032",
        "computed_rate_usd_per_million": "0.042",
        "cumulative_budget_usd": "0.25",
        "probability_sum_absolute_tolerance": "0.01",
        "quality_labels": False,
        "conclusion_authority": "systems_compatibility_only",
        "source_commit": SOURCE_COMMIT,
        "sources": SOURCES,
    }
    if any(value[key] != expected for key, expected in literals.items()):
        raise ValueError("recovery plan binding")
    if digest(PREDECESSOR_PLAN) != PREDECESSOR_PLAN_SHA256:
        raise ValueError("predecessor plan drift")
    commit = value["source_commit"]
    upstream = "refs/remotes/origin/main"
    resolved_upstream = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", upstream],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if resolved_upstream.returncode != 0:
        raise ValueError("origin/main is unavailable")
    reachable = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, upstream],
        check=False,
    )
    if reachable.returncode != 0:
        raise ValueError("pinned commit is not reachable from main")
    for path, expected in SOURCES.items():
        target = ROOT / path
        if not target.is_file() or digest(target) != expected:
            raise ValueError("recovery source digest")
        if hashlib.sha256(_git_bytes(commit, path)).hexdigest() != expected:
            raise ValueError("pinned materializer digest")
    partial = value["required_partial_contract"]
    if partial != {
        "schema_version": "phase4c3-result.v1",
        "status": "partial",
        "failure_category": "response_invalid",
        "completed_request_count": 10,
        "completed_decision_count": 19,
    }:
        raise ValueError("partial contract")
    wires = value["wire_sha256_by_ordinal"]
    if (
        not isinstance(wires, dict)
        or set(wires) != {str(x) for x in range(57)}
        or not all(_is_digest(x) for x in wires.values())
    ):
        raise ValueError("wire digest map")
    materialized = materialize_requests()
    for item in materialized:
        if hashlib.sha256(item["wire"]).hexdigest() != wires[str(item["ordinal"])]:
            raise ValueError("materialized wire drift")
    if (
        not isinstance(value["diagnostic_categories"], list)
        or set(value["diagnostic_categories"]) != DIAGNOSTICS
        or len(value["diagnostic_categories"]) != len(DIAGNOSTICS)
    ):
        raise ValueError("diagnostic categories")
    return value


def _decimal_fraction(value: Any, category: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ResponseError(category)
    decimal = Decimal(value)
    digits = decimal.as_tuple().digits
    exponent = decimal.as_tuple().exponent
    if (
        not isinstance(exponent, int)
        or len(digits) > 1000
        or abs(exponent) > 1000
        or not decimal.is_finite()
        or not Decimal(0) <= decimal <= Decimal(1)
    ):
        raise ResponseError(category)
    return Fraction(decimal)


def validate_response(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Validate and reduce a response without returning provider content."""
    if not isinstance(value, dict):
        raise ResponseError("root_shape")
    if set(value) != {"model", "answers", "usage"}:
        raise ResponseError("root_shape")
    if not isinstance(value["model"], str):
        raise ResponseError("unexpected_response_type")
    if value["model"] != MODEL:
        raise ResponseError("model_drift")
    answers = value["answers"]
    if not isinstance(answers, dict):
        raise ResponseError("unexpected_response_type")
    if set(answers) != set(request["questions"]):
        raise ResponseError("answer_key_mismatch")
    usage = value["usage"]
    if not isinstance(usage, dict):
        raise ResponseError("unexpected_response_type")
    if set(usage) != {"input_tokens", "output_tokens"}:
        raise ResponseError("usage_shape")
    if any(
        isinstance(usage[key], bool)
        or not isinstance(usage[key], int)
        or usage[key] < 0
        or usage[key] > 2**53 - 1
        for key in usage
    ):
        raise ResponseError("usage_invalid")
    tolerance_used = False
    for key, question in request["questions"].items():
        answer = answers[key]
        if not isinstance(answer, dict):
            raise ResponseError("unexpected_response_type")
        if set(answer) != {"type", "choice", "probabilities", "confidence"}:
            raise ResponseError("answer_shape")
        if not isinstance(answer["type"], str):
            raise ResponseError("unexpected_response_type")
        if answer["type"] != "choice":
            raise ResponseError("answer_type_mismatch")
        if not isinstance(answer["choice"], str):
            raise ResponseError("unexpected_response_type")
        criteria = question["criteria"]
        if answer["choice"] not in criteria:
            raise ResponseError("choice_unknown")
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, dict):
            raise ResponseError("unexpected_response_type")
        if set(probabilities) != set(criteria):
            raise ResponseError("probability_key_mismatch")
        values = [_decimal_fraction(item, "probability_invalid") for item in probabilities.values()]
        _decimal_fraction(answer["confidence"], "confidence_invalid")
        error = abs(sum(values, Fraction(0)) - 1)
        if error > Fraction(1, 100):
            raise ResponseError("probability_sum_out_of_tolerance")
        tolerance_used = tolerance_used or error > Fraction(1, 1_000_000)
        selected = _decimal_fraction(probabilities[answer["choice"]], "probability_invalid")
        if selected < max(values):
            raise ResponseError("choice_not_maximal")
    return {
        "model": MODEL,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "probability_sum_tolerance_used": tolerance_used,
    }
