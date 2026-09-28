from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.ptbr_native.models import EXPECTED_MANIFEST, load_protocol_manifest, load_report
from benchmarks.ptbr_native.protocol import (
    build_plan,
    build_request,
    earliest_argmax,
    framed_hash,
    lexical_tokens,
    percentile,
)
from saracura.serialization import canonical_json_bytes

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "benchmarks/manifests/phase5d-ptbr-native.v1.json"


def fixture_rows() -> tuple[list[dict[str, Any]], ...]:
    corpus = [
        {"_id": "c0", "title": "", "text": "Resposta correta zero"},
        {"_id": "c1", "title": "", "text": "Resposta correta um"},
        {"_id": "c2", "title": "", "text": "Resposta alternativa dois"},
        {"_id": "c3", "title": "", "text": "Resposta alternativa três"},
    ]
    queries = [
        {"_id": "q1", "text": "Qual é a resposta um?"},
        {"_id": "q0", "text": "Qual é a resposta zero?"},
    ]
    qrels = [
        {"query-id": "q0", "corpus-id": "c0", "score": 1},
        {"query-id": "q1", "corpus-id": "c1", "score": 1},
    ]
    return corpus, queries, qrels


def valid_report() -> dict[str, Any]:
    return {
        "schema_version": "phase5d-ptbr-faq-bacen-report.v1",
        "benchmark_id": "phase5d-ptbr-faq-bacen",
        "generated_at_utc": "2026-09-28T20:00:00Z",
        "code_commit": "a" * 40,
        "release_code_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "protocol_plan_sha256": (
            "089a83874ed0cc57da45eb143f4b6f6494125dbfabddb660f8e7ca5105175f6f"
        ),
        "backend": "julia",
        "model_id": "supersoniclabs/julia-1",
        "model_revision": "revision",
        "checkpoint_sha256": "d" * 64,
        "device": "cpu",
        "platform": {"system": "Darwin", "machine": "arm64", "python": "3.13.0"},
        "dependencies": {"saracura": "0.1.0a2"},
        "counts": {
            "planned": 373,
            "valid": 373,
            "correct": 200,
            "incorrect": 173,
            "rejected": 0,
            "error": 0,
        },
        "planned_top1_accuracy": 200 / 373,
        "coverage": 1.0,
        "valid_top1_accuracy": 200 / 373,
        "by_gold_position": [
            {
                "position": position,
                "planned": planned,
                "valid": planned,
                "correct": 50,
                "planned_top1_accuracy": 50 / planned,
            }
            for position, planned in enumerate((94, 93, 93, 93))
        ],
        "latency": {
            "cold_request_ms": 10.0,
            "warm_p50_ms": 1.0,
            "warm_p95_ms": 2.0,
            "throughput_rows_per_second": 100.0,
        },
        "chance_accuracy": 0.25,
        "lexical_baseline_correct": 265,
        "lexical_baseline_accuracy": 265 / 373,
        "rejection_categories": {
            "request_capacity": 0,
            "backend_capacity": 0,
            "backend_unavailable": 0,
            "invalid_response": 0,
            "other_error": 0,
        },
        "backend_choice_argmax_divergences": 0,
        "calibration_status": "uncalibrated",
        "automation_allowed": False,
        "limitations": ["Aggregate benchmark only."],
    }


def test_bundled_manifest_is_exact_and_duplicate_keys_fail() -> None:
    parsed = load_protocol_manifest(MANIFEST.read_bytes())
    assert parsed.model_dump(mode="json") == EXPECTED_MANIFEST
    raw = MANIFEST.read_text().replace(
        '"schema_version": "phase5d-ptbr-faq-bacen-protocol.v1",',
        '"schema_version": "phase5d-ptbr-faq-bacen-protocol.v1",'
        '"schema_version": "phase5d-ptbr-faq-bacen-protocol.v1",',
        1,
    )
    with pytest.raises(ValueError, match="invalid Phase 5D"):
        load_protocol_manifest(raw)


def test_manifest_rejects_unknown_fields_and_protocol_drift() -> None:
    unknown = json.loads(MANIFEST.read_text())
    unknown["unexpected"] = True
    with pytest.raises(ValueError):
        load_protocol_manifest(json.dumps(unknown))
    drift = json.loads(MANIFEST.read_text())
    drift["selection"]["snippet_codepoints"] = 121
    with pytest.raises(ValueError):
        load_protocol_manifest(json.dumps(drift))


def test_plan_is_deterministic_balanced_and_uses_framed_hashing() -> None:
    rows = fixture_rows()
    first = build_plan(*rows, enforce_frozen_counts=False)
    second = build_plan(*rows, enforce_frozen_counts=False)
    assert first == second
    assert [row.query_id for row in first] == ["q0", "q1"]
    assert [row.gold_position for row in first] == [0, 1]
    assert len({*first[0].ordered_choice_snippets}) == 4
    assert framed_hash("ab", "c") != framed_hash("a", "bc")
    digest = hashlib.sha256(
        canonical_json_bytes([row.digest_record() for row in first])
    ).hexdigest()
    assert len(digest) == 64


def test_plan_rejects_shape_relation_and_cardinality_drift() -> None:
    corpus, queries, qrels = fixture_rows()
    with pytest.raises(ValueError, match="columns"):
        build_plan(
            [{**corpus[0], "extra": "x"}, *corpus[1:]],
            queries,
            qrels,
            enforce_frozen_counts=False,
        )
    with pytest.raises(ValueError, match="unknown"):
        build_plan(
            corpus,
            queries,
            [{**qrels[0], "corpus-id": "missing"}, qrels[1]],
            enforce_frozen_counts=False,
        )
    with pytest.raises(ValueError, match="cardinalities"):
        build_plan(corpus, queries, qrels)


def test_request_is_identical_except_for_backend_owned_identity() -> None:
    row = build_plan(*fixture_rows(), enforce_frozen_counts=False)[0]
    julia = build_request(row, model="julia-revision", backend="julia")
    owned = build_request(row, model="owned-revision", backend="saracura-universal")
    assert julia.state == owned.state
    assert julia.questions == owned.questions
    assert julia.workflow.revision == "phase5c-julia.v1"
    assert owned.workflow.revision == "phase4e-saracura-ranker.v1"


def test_lexical_token_argmax_and_percentile_rules() -> None:
    assert lexical_tokens("AÇÃO ação abc abcd foo_bar") == {"ação", "abcd"}
    assert earliest_argmax([2.0, 2.0, 1.0]) == 0
    with pytest.raises(ValueError):
        earliest_argmax([float("nan")])
    assert percentile([], 0.5) is None
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_report_recomputes_metrics_and_rejects_sensitive_content() -> None:
    payload = valid_report()
    assert load_report(json.dumps(payload)).counts.correct == 200
    drift = valid_report()
    drift["planned_top1_accuracy"] = 0.99
    with pytest.raises(ValueError):
        load_report(json.dumps(drift))
    sensitive = valid_report()
    sensitive["limitations"] = ["found at /Users/person/private"]
    with pytest.raises(ValueError, match="forbidden"):
        load_report(json.dumps(sensitive))


def test_report_accounts_rejects_and_errors_against_plan() -> None:
    payload = valid_report()
    payload["counts"] = {
        "planned": 373,
        "valid": 368,
        "correct": 200,
        "incorrect": 168,
        "rejected": 3,
        "error": 2,
    }
    payload["planned_top1_accuracy"] = 200 / 373
    payload["coverage"] = 368 / 373
    payload["valid_top1_accuracy"] = 200 / 368
    payload["rejection_categories"] = {
        "request_capacity": 3,
        "backend_capacity": 0,
        "backend_unavailable": 2,
        "invalid_response": 0,
        "other_error": 0,
    }
    assert load_report(json.dumps(payload)).counts.rejected == 3


def test_importing_protocol_does_not_import_pyarrow() -> None:
    source = (ROOT / "benchmarks/ptbr_native/protocol.py").read_text()
    assert "pyarrow" not in source
