"""Closed plan-v2 validation and deterministic, self-authored text generation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, TypedDict, cast

from benchmarks.universal_local.inputs import TextChoice, TextState
from benchmarks.universal_local.registry import registry_digest

ROOT = Path(__file__).parents[2]
PLAN_PATH = ROOT / "benchmarks" / "manifests" / "universal-bakeoff-plan.v2.json"
V1_PLAN_PATH = ROOT / "benchmarks" / "manifests" / "universal-bakeoff-plan.v1.json"
V1_PLAN_DIGEST = "535165cdf6eacc3bd73bafecaf2ef2c82050890d3525045667623177614be08d"
SCENARIO_FIELDS = frozenset(
    {
        "id",
        "locale",
        "domain",
        "novelty",
        "N",
        "Q",
        "K",
        "input_mode",
        "input_seed",
        "state_length_profile",
        "choice_length_profile",
    }
)
EXPECTED_MATRIX = (
    (
        "scenario-01",
        "pt-BR",
        "support",
        "seen_taxonomy",
        1,
        1,
        5,
        "self_authored",
        "phase4c2-support-ptbr-01",
        "short",
        "short",
    ),
    (
        "scenario-02",
        "pt-BR",
        "support",
        "seen_taxonomy",
        8,
        1,
        5,
        "rule_generated",
        "phase4c2-support-ptbr-02",
        "medium",
        "short",
    ),
    (
        "scenario-03",
        "pt-BR",
        "support",
        "unseen_taxonomy",
        1,
        10,
        5,
        "rule_generated",
        "phase4c2-support-ptbr-03",
        "medium",
        "medium",
    ),
    (
        "scenario-04",
        "pt-BR",
        "inbox_triage",
        "unseen_taxonomy",
        8,
        10,
        20,
        "rule_generated",
        "phase4c2-inbox-ptbr-04",
        "medium",
        "medium",
    ),
    (
        "scenario-05",
        "en",
        "support",
        "seen_taxonomy",
        1,
        1,
        5,
        "self_authored",
        "phase4c2-support-en-05",
        "short",
        "short",
    ),
    (
        "scenario-06",
        "pt-BR",
        "support",
        "seen_taxonomy",
        32,
        1,
        2,
        "rule_generated",
        "phase4c2-support-ptbr-06",
        "long",
        "medium",
    ),
    (
        "scenario-07",
        "pt-BR",
        "inbox_triage",
        "unseen_taxonomy",
        1,
        50,
        20,
        "rule_generated",
        "phase4c2-inbox-ptbr-07",
        "long",
        "long",
    ),
    (
        "scenario-08",
        "pt-BR",
        "support",
        "seen_taxonomy",
        1,
        1,
        77,
        "rule_generated",
        "phase4c2-support-ptbr-08",
        "medium",
        "long",
    ),
)
REPORT_METRICS = frozenset(
    {
        "candidate_id",
        "model_id",
        "model_revision",
        "acquisition_contract_digest",
        "conformance_vector_sha256",
        "conformance_state",
        "architecture_attribution",
        "plan_sha256",
        "python_version",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
        "safetensors_version",
        "os_version",
        "architecture",
        "device_type",
        "execution_dtype",
        "source_weight_dtypes",
        "process_threads",
        "attention_implementation",
        "mps_fallback_enabled",
        "microbatch_size",
        "load_duration_ms",
        "warmup_duration_ms",
        "latency_samples_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "decisions_per_second",
        "peak_rss_bytes",
        "mps_current_allocated_bytes",
        "mps_driver_allocated_bytes",
        "input_token_count_min",
        "input_token_count_max",
        "state_tokens_original",
        "state_tokens_retained",
        "truncated_state_tokens",
        "choice_tokens_original",
        "choice_tokens_retained",
        "truncated_choice_tokens",
        "supported_count",
        "supported_with_state_truncation_count",
        "supported_with_choice_truncation_count",
        "supported_with_state_and_choice_truncation_count",
        "unsupported_count",
        "failed_count",
        "deterministic_repeat_match",
        "comparison_eligible",
        "error_code",
    }
)
RESERVED_FRAME = re.compile(r"[SBID](?:0|[1-9][0-9]*):")


class Scenario(TypedDict):
    id: str
    locale: str
    domain: str
    novelty: str
    N: int
    Q: int
    K: int
    input_mode: str
    input_seed: str
    state_length_profile: str
    choice_length_profile: str


class Plan(TypedDict):
    schema_version: str
    plan_id: str
    generator_revision: str
    evidence_lane: str
    candidate_registry_sha256: str
    predecessor_plan_sha256: str
    dataset_artifact: None
    remote_budget: dict[str, int]
    generator_contract: dict[str, Any]
    scenarios: list[Scenario]
    report_policy: dict[str, Any]


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate plan key")
        result[key] = value
    return result


def _raw(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid plan JSON") from error
    if not isinstance(value, dict):
        raise ValueError("plan root invalid")
    return value


def _values(scenario: Scenario) -> tuple[object, ...]:
    return (
        scenario["id"],
        scenario["locale"],
        scenario["domain"],
        scenario["novelty"],
        scenario["N"],
        scenario["Q"],
        scenario["K"],
        scenario["input_mode"],
        scenario["input_seed"],
        scenario["state_length_profile"],
        scenario["choice_length_profile"],
    )


def _profiles(value: Any, expected: dict[str, tuple[int, int]]) -> bool:
    return isinstance(value, dict) and value == {
        key: list(bounds) for key, bounds in expected.items()
    }


def load_plan(raw: bytes | None = None) -> Plan:
    if hashlib.sha256(V1_PLAN_PATH.read_bytes()).hexdigest() != V1_PLAN_DIGEST:
        raise ValueError("predecessor plan bytes mismatch")
    value = _raw(PLAN_PATH.read_bytes() if raw is None else raw)
    required = {
        "schema_version",
        "plan_id",
        "generator_revision",
        "evidence_lane",
        "candidate_registry_sha256",
        "predecessor_plan_sha256",
        "dataset_artifact",
        "remote_budget",
        "generator_contract",
        "scenarios",
        "report_policy",
    }
    if set(value) != required:
        raise ValueError("plan root is not closed")
    if (
        value["schema_version"] != "universal-bakeoff-plan.v2"
        or value["plan_id"] != "phase4c2-local-candidates"
        or value["generator_revision"] != "phase4c.2-text-fixture-generator.v1"
        or value["evidence_lane"] != "synthetic_research"
        or value["candidate_registry_sha256"] != registry_digest()
        or value["predecessor_plan_sha256"] != V1_PLAN_DIGEST
        or value["dataset_artifact"] is not None
        or value["remote_budget"] != {"requests": 0, "tokens": 0, "usd": 0}
    ):
        raise ValueError("plan binding invalid")
    contract = value["generator_contract"]
    if not isinstance(contract, dict) or set(contract) != {
        "reserved_labels",
        "state_byte_profiles",
        "choice_byte_profiles",
        "sentence_banks",
    }:
        raise ValueError("generator contract is not closed")
    if (
        contract["reserved_labels"] != ["S", "B", "I", "D"]
        or not _profiles(
            contract["state_byte_profiles"],
            {"short": (80, 240), "medium": (400, 900), "long": (1800, 2800)},
        )
        or not _profiles(
            contract["choice_byte_profiles"],
            {"short": (8, 40), "medium": (41, 96), "long": (97, 180)},
        )
    ):
        raise ValueError("generator profiles invalid")
    raw_scenarios = value["scenarios"]
    if not isinstance(raw_scenarios, list) or len(raw_scenarios) != len(EXPECTED_MATRIX):
        raise ValueError("plan scenario matrix invalid")
    scenarios: list[Scenario] = []
    for raw_scenario, expected in zip(raw_scenarios, EXPECTED_MATRIX, strict=True):
        if not isinstance(raw_scenario, dict) or set(raw_scenario) != SCENARIO_FIELDS:
            raise ValueError("scenario is not closed")
        scenario = cast(Scenario, raw_scenario)
        if _values(scenario) != expected:
            raise ValueError("scenario matrix mismatch")
        scenarios.append(scenario)
    policy = value["report_policy"]
    if (
        not isinstance(policy, dict)
        or set(policy) != {"status", "metric_authority", "conclusion_authority", "allowed_metrics"}
        or policy["status"] != "real_local_candidate"
        or policy["metric_authority"] != "measured_systems_only"
        or policy["conclusion_authority"] != "systems_compatibility_only"
        or not isinstance(policy["allowed_metrics"], list)
        or set(policy["allowed_metrics"]) != REPORT_METRICS
        or len(policy["allowed_metrics"]) != len(REPORT_METRICS)
    ):
        raise ValueError("report policy invalid")
    return cast(Plan, {**value, "scenarios": scenarios})


def _pick(bank: list[str], seed: str, index: int, label: str) -> str:
    if not bank or not all(isinstance(item, str) and item for item in bank):
        raise ValueError("invalid sentence bank")
    digest = hashlib.sha256(f"{seed}:{index}:{label}".encode()).digest()
    return bank[int.from_bytes(digest[:8], "big") % len(bank)]


def _fit(text: str, low: int, high: int) -> str:
    result = text
    while len(result.encode("utf-8")) < low:
        result += " " + text
    encoded = result.encode("utf-8")[:high]
    while True:
        try:
            return encoded.decode("utf-8").rstrip()
        except UnicodeDecodeError:
            encoded = encoded[:-1]


def materialize(
    plan: Plan, scenario: Scenario
) -> tuple[
    tuple[TextState, ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[TextChoice, ...], ...], ...],
]:
    roots = plan["generator_contract"]["sentence_banks"]
    if not isinstance(roots, dict):
        raise ValueError("invalid sentence banks")
    banks = roots.get(scenario["domain"], {}).get(scenario["locale"])
    if not isinstance(banks, dict) or set(banks) != {
        "subject",
        "body",
        "instruction",
        "description",
    }:
        raise ValueError("scenario sentence bank invalid")
    state_low, state_high = cast(
        list[int],
        plan["generator_contract"]["state_byte_profiles"][scenario["state_length_profile"]],
    )
    choice_low, choice_high = cast(
        list[int],
        plan["generator_contract"]["choice_byte_profiles"][scenario["choice_length_profile"]],
    )
    seed = scenario["input_seed"]
    states: list[TextState] = []
    for index in range(scenario["N"]):
        subject = _pick(cast(list[str], banks["subject"]), seed, index, "subject")
        body = _fit(
            _pick(cast(list[str], banks["body"]), seed, index, "body"),
            state_low - len(subject.encode()),
            state_high - len(subject.encode()),
        )
        if not state_low <= len((subject + body).encode()) <= state_high:
            raise ValueError("generated state profile invalid")
        states.append(TextState(subject, body))
    questions = tuple(
        tuple(
            _pick(
                cast(list[str], banks["instruction"]),
                seed,
                state * scenario["Q"] + question,
                "instruction",
            )
            for question in range(scenario["Q"])
        )
        for state in range(scenario["N"])
    )
    choices = tuple(
        tuple(
            tuple(
                TextChoice(
                    f"choice-{choice}",
                    _fit(
                        _pick(
                            cast(list[str], banks["description"]),
                            seed,
                            state * scenario["Q"] * scenario["K"]
                            + question * scenario["K"]
                            + choice,
                            "description",
                        ),
                        choice_low,
                        choice_high,
                    ),
                )
                for choice in range(scenario["K"])
            )
            for question in range(scenario["Q"])
        )
        for state in range(scenario["N"])
    )
    source_values = (
        [value for state in states for value in (state.subject, state.body)]
        + [item for row in questions for item in row]
        + [choice.description for states_row in choices for row in states_row for choice in row]
    )
    if any(unicodedata.normalize("NFC", value) != value for value in source_values):
        raise ValueError("generated text is not NFC")
    if any(RESERVED_FRAME.search(value) for value in source_values):
        raise ValueError("generated text contains a reserved framing label")
    if any(
        not choice_low <= len(choice.description.encode()) <= choice_high
        for states_row in choices
        for row in states_row
        for choice in row
    ):
        raise ValueError("generated choice profile invalid")
    return tuple(states), questions, choices


def validate_plan() -> Plan:
    plan = load_plan()
    for scenario in plan["scenarios"]:
        materialize(plan, scenario)
    return plan


def plan_digest() -> str:
    return hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
