"""Offline, fixture-only contracts and computation-shape harness for Phase 4C.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Never, TypeVar
from urllib.parse import urlparse

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

ROOT = Path(__file__).parents[1]
MANIFEST_DIR = ROOT / "benchmarks" / "manifests"
REGISTRY_PATH = MANIFEST_DIR / "decision-backend-candidates.v1.json"
PLAN_PATH = MANIFEST_DIR / "universal-bakeoff-plan.v1.json"
ARTIFACT_ROOT = ROOT / ".artifacts" / "universal-bakeoff"
TOOL_REVISION = "phase4c.1-reference-harness.v1"
FIXTURE_GENERATOR_REVISION = "phase4c.1-fixture-generator.v1"
METRICS = (
    "declared_sequence_count",
    "declared_attention_work",
    "declared_state_count",
    "declared_question_count",
    "declared_choice_count",
    "unsupported_count",
    "failed_count",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ContractModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Candidate(ContractModel):
    id: StrictStr
    display_name: StrictStr = Field(min_length=3, max_length=80)
    architecture_class: Literal[
        "compiled_encoder_head",
        "option_marker_encoder",
        "zero_shot_nli",
        "constrained_causal_lm",
        "native_remote",
        "masked_diffusion",
    ]
    role: Literal[
        "specialized_reference",
        "universal_candidate",
        "teacher_control",
        "product_control",
        "documented_exclusion",
    ]
    source_type: Literal["first_party", "open_source", "remote_service"]
    source_url: StrictStr
    source_revision: StrictStr | None
    source_license: Literal["Apache-2.0", "MIT", "unknown"]
    license_evidence_url: StrictStr
    model_id: StrictStr | None
    model_revision: StrictStr | None
    weight_format: Literal["not_reviewed", "not_applicable"]
    execution_boundary: Literal[
        "existing_verified_local", "review_required_local", "review_required_remote", "excluded"
    ]
    data_transmission: Literal["none", "provider_request", "not_applicable"]
    claimed_local_mac: StrictBool
    claimed_dynamic_choices: StrictBool
    claimed_ptbr: StrictBool
    claimed_question_conditioned: StrictBool
    claim_evidence_state: Literal["upstream_readme", "upstream_source_reviewed", "measured"]
    acquisition_state: Literal["existing", "not_reviewed", "excluded", "not_applicable"]
    exclusion_reason: StrictStr | None
    notes: StrictStr = Field(min_length=20, max_length=500)


class CandidateRegistry(ContractModel):
    schema_version: Literal["decision-backend-candidates.v1"]
    reviewed_at: StrictStr
    candidates: list[Candidate]


class Scenario(ContractModel):
    id: StrictStr
    locale: Literal["pt-BR", "en"]
    domain: Literal["support", "inbox_triage"]
    novelty: Literal["seen_taxonomy", "unseen_taxonomy", "unseen_domain"]
    N: StrictInt = Field(gt=0, le=32)
    Q: StrictInt = Field(gt=0, le=50)
    K: StrictInt = Field(gt=0, le=77)
    input_mode: Literal["self_authored", "rule_generated"]
    input_seed: StrictStr = Field(min_length=8, max_length=100)
    authorized_metrics: list[
        Literal[
            "declared_sequence_count",
            "declared_attention_work",
            "declared_state_count",
            "declared_question_count",
            "declared_choice_count",
            "unsupported_count",
            "failed_count",
        ]
    ]


class RemoteBudget(ContractModel):
    requests: StrictInt = Field(ge=0)
    usd: StrictInt = Field(ge=0)


class ReportPolicy(ContractModel):
    status: Literal["fixture_only"]
    metric_authority: Literal["declared_analytic_model"]
    conclusion_authority: Literal["none"]
    placeholder_label: Literal["deterministic_placeholder"]
    allowed_metrics: list[
        Literal[
            "declared_sequence_count",
            "declared_attention_work",
            "declared_state_count",
            "declared_question_count",
            "declared_choice_count",
            "unsupported_count",
            "failed_count",
        ]
    ]


class BakeoffPlan(ContractModel):
    schema_version: Literal["universal-bakeoff-plan.v1"]
    plan_id: Literal["universal-bakeoff-plan.v1"]
    generator_revision: Literal["phase4c.1-fixture-generator.v1"]
    evidence_lane: Literal["synthetic_research"]
    candidate_registry_sha256: StrictStr
    dataset_artifact: None
    remote_budget: RemoteBudget
    scenarios: list[Scenario]
    report_policy: ReportPolicy


ContractType = TypeVar("ContractType", bound=ContractModel)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
    )


def _model(model: type[ContractType], raw: bytes) -> ContractType:
    try:
        return model.model_validate(_load_json(raw))
    except (ValueError, ValidationError) as error:
        raise ValueError("invalid closed benchmark contract") from error


def _https_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and not re.search(r"/(?:latest|main|master|resolve/main|tree/main)(?:/|$)", parsed.path)
    )


EXPECTED_IDS = (
    "saracura-compiled",
    "laya-multilingual",
    "von-option-marker",
    "poorjev-nli",
    "qwen-system-one",
    "typesafe-jev",
    "diffusiongemma-openjev",
)
EXPECTED_SOURCE_REVISIONS = {
    "laya-multilingual": "573e5b62696ba441230cd6be71d593331b5d23af",
    "von-option-marker": "2656a69be2ef0ebf2f3058302e52791f99cbd8de",
    "poorjev-nli": "7e684e95db13b90238c63e6ab39e1a016263a168",
    "qwen-system-one": "ebde2a2db7067b920dfe51e9ce785613e66613d5",
    "diffusiongemma-openjev": "e04794ab36e4f7e6040c2547baecdb2737ce2e79",
}
EXPECTED_SOURCE_URLS = {
    "saracura-compiled": "https://github.com/felhen-ai/saracura",
    "laya-multilingual": "https://github.com/NandhaKishorM/laya",
    "von-option-marker": "https://github.com/wfzyx/von",
    "poorjev-nli": "https://github.com/rupeshpoojary9/poorjev",
    "qwen-system-one": "https://github.com/sgoedecke/system-one",
    "typesafe-jev": "https://typesafe.ai",
    "diffusiongemma-openjev": "https://github.com/razorback16/openjev",
}
EXPECTED_LICENSE_EVIDENCE_URLS = {
    "saracura-compiled": (
        "https://github.com/felhen-ai/saracura/blob/"
        "e9ff46e4e5aadf0d5dfa9d3c802b1f7a42d17b6a/LICENSE"
    ),
    "laya-multilingual": (
        "https://github.com/NandhaKishorM/laya/blob/"
        "573e5b62696ba441230cd6be71d593331b5d23af/LICENSE"
    ),
    "von-option-marker": (
        "https://github.com/wfzyx/von/blob/2656a69be2ef0ebf2f3058302e52791f99cbd8de/LICENSE"
    ),
    "poorjev-nli": (
        "https://github.com/rupeshpoojary9/poorjev/blob/"
        "7e684e95db13b90238c63e6ab39e1a016263a168/LICENSE"
    ),
    "qwen-system-one": (
        "https://github.com/sgoedecke/system-one/blob/"
        "ebde2a2db7067b920dfe51e9ce785613e66613d5/README.md"
    ),
    "typesafe-jev": "https://typesafe.ai",
    "diffusiongemma-openjev": (
        "https://github.com/razorback16/openjev/blob/"
        "e04794ab36e4f7e6040c2547baecdb2737ce2e79/README.md"
    ),
}
EXPECTED_CANDIDATE_SHAPES: dict[str, dict[str, Any]] = {
    "saracura-compiled": {
        "architecture_class": "compiled_encoder_head",
        "role": "specialized_reference",
        "source_type": "first_party",
        "source_license": "Apache-2.0",
        "execution_boundary": "review_required_local",
        "data_transmission": "none",
        "claimed_local_mac": True,
        "claimed_dynamic_choices": False,
        "claimed_ptbr": True,
        "claimed_question_conditioned": False,
        "claim_evidence_state": "upstream_source_reviewed",
        "acquisition_state": "not_reviewed",
        "weight_format": "not_reviewed",
    },
    "laya-multilingual": {
        "architecture_class": "option_marker_encoder",
        "role": "universal_candidate",
        "source_type": "open_source",
        "source_license": "Apache-2.0",
        "execution_boundary": "review_required_local",
        "data_transmission": "none",
        "claimed_local_mac": True,
        "claimed_dynamic_choices": True,
        "claimed_ptbr": False,
        "claimed_question_conditioned": True,
        "claim_evidence_state": "upstream_readme",
        "acquisition_state": "not_reviewed",
        "weight_format": "not_reviewed",
    },
    "von-option-marker": {
        "architecture_class": "option_marker_encoder",
        "role": "universal_candidate",
        "source_type": "open_source",
        "source_license": "Apache-2.0",
        "execution_boundary": "review_required_local",
        "data_transmission": "none",
        "claimed_local_mac": True,
        "claimed_dynamic_choices": True,
        "claimed_ptbr": False,
        "claimed_question_conditioned": True,
        "claim_evidence_state": "upstream_readme",
        "acquisition_state": "not_reviewed",
        "weight_format": "not_reviewed",
    },
    "poorjev-nli": {
        "architecture_class": "zero_shot_nli",
        "role": "universal_candidate",
        "source_type": "open_source",
        "source_license": "MIT",
        "execution_boundary": "review_required_local",
        "data_transmission": "none",
        "claimed_local_mac": True,
        "claimed_dynamic_choices": True,
        "claimed_ptbr": False,
        "claimed_question_conditioned": True,
        "claim_evidence_state": "upstream_readme",
        "acquisition_state": "not_reviewed",
        "weight_format": "not_reviewed",
    },
    "qwen-system-one": {
        "architecture_class": "constrained_causal_lm",
        "role": "teacher_control",
        "source_type": "open_source",
        "source_license": "unknown",
        "execution_boundary": "review_required_remote",
        "data_transmission": "provider_request",
        "claimed_local_mac": False,
        "claimed_dynamic_choices": True,
        "claimed_ptbr": False,
        "claimed_question_conditioned": True,
        "claim_evidence_state": "upstream_readme",
        "acquisition_state": "not_applicable",
        "weight_format": "not_reviewed",
    },
    "typesafe-jev": {
        "architecture_class": "native_remote",
        "role": "product_control",
        "source_type": "remote_service",
        "source_license": "unknown",
        "execution_boundary": "review_required_remote",
        "data_transmission": "provider_request",
        "claimed_local_mac": False,
        "claimed_dynamic_choices": True,
        "claimed_ptbr": False,
        "claimed_question_conditioned": True,
        "claim_evidence_state": "upstream_readme",
        "acquisition_state": "not_applicable",
        "weight_format": "not_applicable",
    },
    "diffusiongemma-openjev": {
        "architecture_class": "masked_diffusion",
        "role": "documented_exclusion",
        "source_type": "open_source",
        "source_license": "unknown",
        "execution_boundary": "excluded",
        "data_transmission": "not_applicable",
        "claimed_local_mac": False,
        "claimed_dynamic_choices": True,
        "claimed_ptbr": False,
        "claimed_question_conditioned": True,
        "claim_evidence_state": "upstream_readme",
        "acquisition_state": "excluded",
        "weight_format": "not_applicable",
    },
}


def load_candidate_registry(raw: bytes | None = None) -> CandidateRegistry:
    registry = _model(CandidateRegistry, REGISTRY_PATH.read_bytes() if raw is None else raw)
    if registry.reviewed_at != "2026-09-22":
        raise ValueError("invalid candidate review date")
    if tuple(candidate.id for candidate in registry.candidates) != EXPECTED_IDS:
        raise ValueError("invalid candidate identity set")
    if len({candidate.id for candidate in registry.candidates}) != len(registry.candidates):
        raise ValueError("duplicate candidate id")
    for candidate in registry.candidates:
        if (
            not ID.fullmatch(candidate.id)
            or not _https_url(candidate.source_url)
            or not _https_url(candidate.license_evidence_url)
            or candidate.source_url != EXPECTED_SOURCE_URLS[candidate.id]
            or candidate.license_evidence_url != EXPECTED_LICENSE_EVIDENCE_URLS[candidate.id]
        ):
            raise ValueError("invalid candidate identity")
        printable_values = [candidate.display_name, candidate.notes]
        if candidate.exclusion_reason is not None:
            printable_values.append(candidate.exclusion_reason)
        if any(not value.isprintable() for value in printable_values):
            raise ValueError("candidate text must be printable")
        if (
            candidate.exclusion_reason is not None
            and not 20 <= len(candidate.exclusion_reason) <= 300
        ):
            raise ValueError("invalid exclusion reason")
        if (
            candidate.id in EXPECTED_SOURCE_REVISIONS
            and candidate.source_revision != EXPECTED_SOURCE_REVISIONS[candidate.id]
        ):
            raise ValueError("candidate revision is not immutable")
        if candidate.id not in EXPECTED_SOURCE_REVISIONS and candidate.source_revision is not None:
            raise ValueError("unexpected candidate revision")
        if candidate.source_revision is not None and not HEX40.fullmatch(candidate.source_revision):
            raise ValueError("candidate revision is not immutable")
        if candidate.model_id is not None or candidate.model_revision is not None:
            raise ValueError("model acquisition is not allowed")
        if any(
            getattr(candidate, key) != value
            for key, value in EXPECTED_CANDIDATE_SHAPES[candidate.id].items()
        ):
            raise ValueError("candidate identity fields are not fixed")
        excluded = candidate.role == "documented_exclusion"
        if (candidate.exclusion_reason is not None) != excluded or (
            excluded and candidate.acquisition_state != "excluded"
        ):
            raise ValueError("invalid exclusion state")
        if excluded and (
            candidate.execution_boundary != "excluded"
            or candidate.weight_format != "not_applicable"
        ):
            raise ValueError("invalid exclusion state")
    return registry


def load_plan(raw: bytes | None = None, registry_raw: bytes | None = None) -> BakeoffPlan:
    registry_bytes = REGISTRY_PATH.read_bytes() if registry_raw is None else registry_raw
    load_candidate_registry(registry_bytes)
    plan = _model(BakeoffPlan, PLAN_PATH.read_bytes() if raw is None else raw)
    if (
        not HEX64.fullmatch(plan.candidate_registry_sha256)
        or hashlib.sha256(registry_bytes).hexdigest() != plan.candidate_registry_sha256
    ):
        raise ValueError("candidate registry digest mismatch")
    if plan.remote_budget.requests != 0 or plan.remote_budget.usd != 0 or len(plan.scenarios) != 8:
        raise ValueError("offline budget or scenario contract violated")
    if plan.report_policy.allowed_metrics != list(METRICS):
        raise ValueError("metric allowlist mismatch")
    expected = [
        (
            "scenario-01",
            "pt-BR",
            "support",
            "seen_taxonomy",
            1,
            1,
            5,
            "self_authored",
            "phase4c1-support-ptbr-01",
        ),
        (
            "scenario-02",
            "pt-BR",
            "support",
            "seen_taxonomy",
            8,
            1,
            5,
            "self_authored",
            "phase4c1-support-ptbr-02",
        ),
        (
            "scenario-03",
            "pt-BR",
            "support",
            "unseen_taxonomy",
            1,
            10,
            5,
            "self_authored",
            "phase4c1-support-ptbr-03",
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
            "phase4c1-inbox-ptbr-04",
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
            "phase4c1-support-en-05",
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
            "phase4c1-support-ptbr-06",
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
            "phase4c1-inbox-ptbr-07",
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
            "phase4c1-support-ptbr-08",
        ),
    ]
    actual = [
        (s.id, s.locale, s.domain, s.novelty, s.N, s.Q, s.K, s.input_mode, s.input_seed)
        for s in plan.scenarios
    ]
    if (
        actual != expected
        or len({s.id for s in plan.scenarios}) != 8
        or any(not s.input_seed.isprintable() for s in plan.scenarios)
        or any(s.authorized_metrics != list(METRICS) for s in plan.scenarios)
    ):
        raise ValueError("scenario matrix mismatch")
    return plan


def _canonical(value: Any) -> bytes:
    return rfc8785.dumps(value)


def _frame(*segments: bytes) -> bytes:
    return b"".join(len(segment).to_bytes(8, "big") + segment for segment in segments)


def _fixture_bytes(kind: str, scenario: Scenario, *indices: int) -> bytes:
    parts = (
        FIXTURE_GENERATOR_REVISION,
        scenario.id,
        scenario.locale,
        scenario.domain,
        scenario.novelty,
        scenario.input_mode,
        scenario.input_seed,
        kind,
        *(str(index) for index in indices),
    )
    return _frame(*(part.encode("utf-8") for part in parts))


def _materialize_scenario(
    scenario: Scenario,
) -> tuple[
    tuple[bytes, ...],
    tuple[tuple[bytes, ...], ...],
    tuple[tuple[tuple[bytes, ...], ...], ...],
]:
    states = tuple(_fixture_bytes("state", scenario, state) for state in range(scenario.N))
    questions = tuple(
        tuple(
            _fixture_bytes("question", scenario, state, question) for question in range(scenario.Q)
        )
        for state in range(scenario.N)
    )
    choices = tuple(
        tuple(
            tuple(
                _fixture_bytes("choice", scenario, state, question, choice)
                for choice in range(scenario.K)
            )
            for question in range(scenario.Q)
        )
        for state in range(scenario.N)
    )
    if (
        len(states) != scenario.N
        or any(len(state_questions) != scenario.Q for state_questions in questions)
        or any(len(question_choices) != scenario.Q for question_choices in choices)
        or any(
            len(choice_set) != scenario.K
            for question_choices in choices
            for choice_set in question_choices
        )
    ):
        raise ValueError("materialized scenario shape mismatch")
    return states, questions, choices


def _sequence_lengths(adapter: str, scenario: Scenario) -> tuple[int, ...]:
    states, questions, choices = _materialize_scenario(scenario)
    if adapter == "compiled_reference_adapter":
        return tuple(len(state) for state in states)
    if adapter == "joint_encoder_reference_adapter":
        return tuple(
            len(
                _frame(
                    b"state",
                    states[state],
                    b"question",
                    question,
                    b"choices",
                    *choices[state][question_index],
                )
            )
            for state, state_questions in enumerate(questions)
            for question_index, question in enumerate(state_questions)
        )
    if adapter == "pairwise_nli_reference_adapter":
        return tuple(
            len(
                _frame(
                    b"premise",
                    states[state],
                    b"hypothesis",
                    question,
                    b"choice",
                    choice,
                )
            )
            for state, state_questions in enumerate(questions)
            for question_index, question in enumerate(state_questions)
            for choice in choices[state][question_index]
        )
    if adapter == "constrained_lm_reference_adapter":
        return tuple(
            len(
                _frame(
                    b"decide",
                    b"state",
                    states[state],
                    b"question",
                    question,
                    b"allowed",
                    *choices[state][question_index],
                    b"answer",
                )
            )
            for state, state_questions in enumerate(questions)
            for question_index, question in enumerate(state_questions)
        )
    raise ValueError("unknown reference adapter")


def _shape(adapter: str, scenario: Scenario) -> tuple[int, int]:
    lengths = _sequence_lengths(adapter, scenario)
    return len(lengths), sum(length * length for length in lengths)


def _reference_results(plan: BakeoffPlan) -> list[dict[str, Any]]:
    adapters = (
        "compiled_reference_adapter",
        "joint_encoder_reference_adapter",
        "pairwise_nli_reference_adapter",
        "constrained_lm_reference_adapter",
    )
    rows: list[dict[str, Any]] = []
    for adapter in adapters:
        for scenario in plan.scenarios:
            sequence_count, attention_work = _shape(adapter, scenario)
            rows.append(
                {
                    "adapter": adapter,
                    "scenario_id": scenario.id,
                    "status": "fixture_only",
                    "metric_authority": "declared_analytic_model",
                    "placeholder_ranking": "deterministic_placeholder",
                    "metrics": {
                        "declared_sequence_count": sequence_count,
                        "declared_attention_work": attention_work,
                        "declared_state_count": scenario.N,
                        "declared_question_count": scenario.N * scenario.Q,
                        "declared_choice_count": scenario.N * scenario.Q * scenario.K,
                        "unsupported_count": 0,
                        "failed_count": 0,
                    },
                }
            )
    return rows


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_reference(output_name: str) -> Path:
    if not NAME.fullmatch(output_name):
        raise ValueError("invalid output name")
    registry_raw = REGISTRY_PATH.read_bytes()
    plan_raw = PLAN_PATH.read_bytes()
    plan = load_plan(raw=plan_raw, registry_raw=registry_raw)
    output = ARTIFACT_ROOT / output_name
    if ARTIFACT_ROOT.is_symlink() or ARTIFACT_ROOT.parent.is_symlink():
        raise ValueError("invalid artifact root")
    ARTIFACT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ARTIFACT_ROOT.is_symlink() or not ARTIFACT_ROOT.is_dir():
        raise ValueError("invalid artifact root")
    os.chmod(ARTIFACT_ROOT, 0o700)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("output already exists") from error
    os.chmod(output, 0o700)
    _fsync_directory(ARTIFACT_ROOT)
    result = {
        "schema_version": "universal-bakeoff-result.v1",
        "status": "fixture_only",
        "metric_authority": "declared_analytic_model",
        "conclusion_authority": "none",
        "candidate_registry_sha256": hashlib.sha256(registry_raw).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "tool_revision": TOOL_REVISION,
        "generator_revision": FIXTURE_GENERATOR_REVISION,
        "runtime": {"python": f"{sys.version_info.major}.{sys.version_info.minor}", "os": os.name},
        "adapters": _reference_results(plan),
    }
    result_raw = _canonical(result) + b"\n"
    report = (
        "# Universal bake-off reference run\n\n"
        "status: fixture_only\n"
        "metric_authority: declared_analytic_model\n"
        "conclusion_authority: none\n"
        f"candidate_registry_sha256: {result['candidate_registry_sha256']}\n"
        f"plan_sha256: {result['plan_sha256']}\n"
        f"tool_revision: {TOOL_REVISION}\n"
        f"generator_revision: {FIXTURE_GENERATOR_REVISION}\n"
        f"adapter_results: {len(result['adapters'])}\n"
    )
    report_raw = report.encode("utf-8")
    try:
        _write_exclusive(output / "result.json", result_raw)
        _write_exclusive(output / "report.md", report_raw)
        manifest = {
            "schema_version": "universal-bakeoff-artifact-manifest.v1",
            "files": {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in (("result.json", result_raw), ("report.md", report_raw))
            },
            "manifest_sha256": "",
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            _canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        ).hexdigest()
        _write_exclusive(output / "artifact-manifest.json", _canonical(manifest) + b"\n")
        _fsync_directory(output)
    except Exception:
        for name in ("artifact-manifest.json", "report.md", "result.json"):
            with suppress(FileNotFoundError):
                (output / name).unlink()
        with suppress(OSError):
            output.rmdir()
        raise
    return output


def _public_error(error: Exception) -> None:
    code = "contract_invalid" if isinstance(error, (ValueError, ValidationError)) else "run_failed"
    print(f"error:{code}", file=sys.stderr)


class RedactingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError("invalid command arguments")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = RedactingArgumentParser(prog="python -m benchmarks.universal_bakeoff")
        subparsers = parser.add_subparsers(dest="command", required=True)
        subparsers.add_parser("validate")
        run_parser = subparsers.add_parser("run-reference")
        run_parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        if args.command == "validate":
            load_plan()
            print("validated universal bake-off contracts")
        else:
            run_reference(args.output)
            print("created universal bake-off artifact")
        return 0
    except Exception as error:
        _public_error(error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
