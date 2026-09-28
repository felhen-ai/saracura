"""Closed manifests and public aggregate report models for Phase 5D."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HEX64 = r"^[0-9a-f]{64}$"


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


EXPECTED_MANIFEST: dict[str, Any] = {
    "schema_version": "phase5d-ptbr-faq-bacen-protocol.v1",
    "benchmark_id": "phase5d-ptbr-faq-bacen",
    "dataset": {
        "repository": "MTEB-BR/faq-bacen",
        "revision": "076d89a68a8b8d2f14e3161631c416ffe29b8463",
        "config": "default",
        "split": "test",
        "files": {
            "corpus/test-00000-of-00001.parquet": {
                "sha256": "378c43e9126a31419680d66c4e43a178a66c8da45a98f64b89af94afb6ecd3e0",
                "rows": 1673,
                "columns": ["_id", "title", "text"],
            },
            "queries/test-00000-of-00001.parquet": {
                "sha256": "5467fc92f2387b436526b609463cdb7f2251fb684667ac9fb4f589aba2e528c0",
                "rows": 373,
                "columns": ["_id", "text"],
            },
            "qrels/test-00000-of-00001.parquet": {
                "sha256": "17c173c566e5e021a4ead335717e97927f925df4f82994ccf4a987aeb8a6affc",
                "rows": 373,
                "columns": ["query-id", "corpus-id", "score"],
            },
        },
        "cardinalities": {
            "corpus": 1673,
            "queries": 373,
            "qrels": 373,
            "unique_gold_corpus_ids": 372,
        },
    },
    "rights": {
        "bcb_source_license": "ODbL-1.0",
        "upstream_repository_license": "Apache-2.0",
        "derivative_dataset_card_license": "not_declared",
        "redistribution": "rows_forbidden_aggregate_only",
    },
    "privacy": {"state": "publisher_open_data_statement_no_record_review"},
    "selection": {
        "seed": "saracura-phase5d-faq-bacen-v1",
        "normalization": "unicode_nfc_strip_collapse_whitespace",
        "snippet_codepoints": 120,
        "query_order": "utf8_id_lexicographic",
        "distractor_order": "sha256_u64be_length_prefixed_utf8_segments",
        "distractors": 3,
        "gold_position": "row_index_mod_4",
        "distinct_snippets_required": True,
    },
    "plan": {
        "fields": [
            "query_id",
            "query_sha256",
            "ordered_choice_ids",
            "ordered_choice_snippet_sha256",
            "gold_position",
        ],
        "canonicalization": "RFC8785",
        "sha256": "089a83874ed0cc57da45eb143f4b6f6494125dbfabddb660f8e7ca5105175f6f",
        "rows": 373,
        "gold_position_counts": [94, 93, 93, 93],
    },
    "request": {
        "api_version": "v1alpha1",
        "mode": "research",
        "locale": "pt-BR",
        "domain": "finance",
        "state_key": "pergunta",
        "question_id": "resposta",
        "instruction": "Escolha a alternativa que começa a responder corretamente à pergunta.",
        "choice_ids": ["a", "b", "c", "d"],
        "julia_workflow": "universal-choice@phase5c-julia.v1",
        "saracura_workflow": "universal-choice@phase4e-saracura-ranker.v1",
        "expected_saracura_state_capacity_rejects": 5,
    },
    "baselines": {
        "chance_accuracy": 0.25,
        "lexical": {
            "normalization": "NFC_casefold_unicode_alphanumeric_no_underscore",
            "minimum_token_codepoints": 4,
            "score": "unique_query_token_overlap",
            "tie_break": "earliest_choice",
            "correct": 265,
            "planned": 373,
            "accuracy": 0.710455764075067,
        },
    },
}


class ProtocolManifest(ClosedModel):
    schema_version: Literal["phase5d-ptbr-faq-bacen-protocol.v1"]
    benchmark_id: Literal["phase5d-ptbr-faq-bacen"]
    dataset: dict[str, Any]
    rights: dict[str, Any]
    privacy: dict[str, Any]
    selection: dict[str, Any]
    plan: dict[str, Any]
    request: dict[str, Any]
    baselines: dict[str, Any]

    @model_validator(mode="after")
    def exact_protocol(self) -> ProtocolManifest:
        if self.model_dump(mode="json") != EXPECTED_MANIFEST:
            raise ValueError("Phase 5D protocol manifest is immutable")
        return self


class OutcomeCounts(ClosedModel):
    planned: Literal[373]
    valid: int = Field(ge=0, le=373)
    correct: int = Field(ge=0, le=373)
    incorrect: int = Field(ge=0, le=373)
    rejected: int = Field(ge=0, le=373)
    error: int = Field(ge=0, le=373)


class PositionMetric(ClosedModel):
    position: int = Field(ge=0, le=3)
    planned: int = Field(ge=0)
    valid: int = Field(ge=0)
    correct: int = Field(ge=0)
    planned_top1_accuracy: float = Field(ge=0, le=1)


class LatencyMetrics(ClosedModel):
    cold_request_ms: float | None = Field(default=None, ge=0)
    warm_p50_ms: float | None = Field(default=None, ge=0)
    warm_p95_ms: float | None = Field(default=None, ge=0)
    throughput_rows_per_second: float = Field(ge=0)


class RejectionCounts(ClosedModel):
    request_capacity: int = Field(ge=0)
    backend_capacity: int = Field(ge=0)
    backend_unavailable: int = Field(ge=0)
    invalid_response: int = Field(ge=0)
    other_error: int = Field(ge=0)


class BenchmarkReport(ClosedModel):
    schema_version: Literal["phase5d-ptbr-faq-bacen-report.v1"]
    benchmark_id: Literal["phase5d-ptbr-faq-bacen"]
    generated_at_utc: str
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    release_code_sha256: str = Field(pattern=HEX64)
    manifest_sha256: str = Field(pattern=HEX64)
    protocol_plan_sha256: Literal[
        "089a83874ed0cc57da45eb143f4b6f6494125dbfabddb660f8e7ca5105175f6f"
    ]
    backend: Literal["julia", "saracura-universal"]
    model_id: str
    model_revision: str
    checkpoint_sha256: str = Field(pattern=HEX64)
    device: Literal["cpu"]
    platform: dict[str, str]
    dependencies: dict[str, str]
    counts: OutcomeCounts
    planned_top1_accuracy: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    valid_top1_accuracy: float | None = Field(default=None, ge=0, le=1)
    by_gold_position: list[PositionMetric] = Field(min_length=4, max_length=4)
    latency: LatencyMetrics
    chance_accuracy: float = Field(ge=0, le=1)
    lexical_baseline_correct: Literal[265]
    lexical_baseline_accuracy: float
    rejection_categories: RejectionCounts
    backend_choice_argmax_divergences: int = Field(ge=0)
    calibration_status: Literal["uncalibrated"]
    automation_allowed: Literal[False]
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_metrics(self) -> BenchmarkReport:
        counts = self.counts
        if self.chance_accuracy != 0.25:
            raise ValueError("chance baseline drifted")
        if counts.correct + counts.incorrect != counts.valid:
            raise ValueError("valid count must equal correct plus incorrect")
        if counts.valid + counts.rejected + counts.error != counts.planned:
            raise ValueError("outcome counts must cover every planned row")
        if sum(item.planned for item in self.by_gold_position) != 373:
            raise ValueError("position totals must cover the plan")
        if tuple(item.position for item in self.by_gold_position) != (0, 1, 2, 3):
            raise ValueError("position metrics must be ordered")
        expected_positions = (94, 93, 93, 93)
        if tuple(item.planned for item in self.by_gold_position) != expected_positions:
            raise ValueError("position totals drifted")
        for item in self.by_gold_position:
            if not 0 <= item.correct <= item.valid <= item.planned:
                raise ValueError("position counts are inconsistent")
            if not math.isclose(
                item.planned_top1_accuracy,
                item.correct / item.planned,
                abs_tol=1e-15,
            ):
                raise ValueError("position accuracy is inconsistent")
        if sum(item.valid for item in self.by_gold_position) != counts.valid:
            raise ValueError("position valid counts do not match the global count")
        if sum(item.correct for item in self.by_gold_position) != counts.correct:
            raise ValueError("position correct counts do not match the global count")
        if not math.isclose(self.planned_top1_accuracy, counts.correct / 373, abs_tol=1e-15):
            raise ValueError("planned accuracy is inconsistent")
        if not math.isclose(self.coverage, counts.valid / 373, abs_tol=1e-15):
            raise ValueError("coverage is inconsistent")
        expected_valid = None if counts.valid == 0 else counts.correct / counts.valid
        if expected_valid is None:
            if self.valid_top1_accuracy is not None:
                raise ValueError("valid accuracy must be null without valid rows")
        elif self.valid_top1_accuracy is None or not math.isclose(
            self.valid_top1_accuracy, expected_valid, abs_tol=1e-15
        ):
            raise ValueError("valid accuracy is inconsistent")
        if not math.isclose(self.lexical_baseline_accuracy, 265 / 373, abs_tol=1e-15):
            raise ValueError("lexical baseline drifted")
        categories = self.rejection_categories
        if (
            categories.request_capacity
            + categories.backend_capacity
            + categories.backend_unavailable
            + categories.invalid_response
            + categories.other_error
            != counts.rejected + counts.error
        ):
            raise ValueError("rejection categories are inconsistent")
        return self


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_protocol_manifest(raw: bytes | str) -> ProtocolManifest:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        return ProtocolManifest.model_validate(payload)
    except (ValueError, TypeError):
        raise ValueError("invalid Phase 5D protocol manifest") from None


def load_report(raw: bytes | str) -> BenchmarkReport:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        report = BenchmarkReport.model_validate(payload)
    except (ValueError, TypeError):
        raise ValueError("invalid Phase 5D aggregate report") from None
    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden = ("/Users/", "/Volumes/", "pergunta", "ordered_choice_ids", "query_id")
    if any(marker in serialized for marker in forbidden):
        raise ValueError("Phase 5D report contains forbidden row or path material")
    return report
