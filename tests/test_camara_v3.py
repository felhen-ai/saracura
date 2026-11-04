from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from research.camara_v3 import (
    ALGORITHM_VERSION,
    BENCHMARK_BASE_URL,
    BENCHMARK_COMMIT,
    FROZEN_SOURCES,
    RETRIEVAL_DATE,
    SOURCE_FILENAMES,
    SOURCE_IDENTITY_RULES,
    SOURCE_IDENTITY_SCOPE,
    Candidate,
    SourceError,
    build_rows,
    score_predictions,
    select_rows,
    source_identity_sha256,
    stable_key,
    validate_derived_artifacts,
    validate_frozen_sources,
)


def _criteria() -> dict[str, str]:
    return {f"theme_{number:02d}": f"Theme {number}" for number in range(32)}


def _rows(labels: list[list[str]]) -> list[dict[str, Any]]:
    criteria = _criteria()
    return [
        {"id": f"row-{index}", "criteria": criteria, "accepted_labels": accepted}
        for index, accepted in enumerate(labels)
    ]


def test_perfect_score_is_100_and_multiple_accepted_prediction_is_valid() -> None:
    labels = [[f"theme_{index:02d}"] for index in range(32)] + [["theme_00", "theme_01"]]
    rows = _rows(labels)
    predictions = {row["id"]: row["accepted_labels"][0] for row in rows}
    score = score_predictions(rows, predictions)
    assert score["macro_valid_topic"] == 100.0
    assert score["raw_any_topic_hit"] == 100.0


def test_singleton_error_reduces_balanced_accuracy() -> None:
    rows = _rows([[f"theme_{index:02d}"] for index in range(32)])
    predictions = {row["id"]: "theme_00" for row in rows}
    score = score_predictions(rows, predictions)
    assert score["macro_valid_topic"] == pytest.approx(100 / 32)


def test_uniform_random_analytic_expectation_is_zero_before_clip() -> None:
    rows = _rows([[f"theme_{index:02d}"] for index in range(32)])
    predictions = {row["id"]: "theme_00" for row in rows}
    score = score_predictions(rows, predictions)
    assert score["uniform_random_chance"] == pytest.approx(100 / 32)
    assert score["normalized"] == 0.0


def test_zero_support_official_theme_is_disclosed_and_illegal_prediction_fails() -> None:
    rows = _rows([["theme_00"] for _ in range(32)])
    predictions = {row["id"]: "theme_00" for row in rows}
    score = score_predictions(rows, predictions)
    assert score["offered_theme_count"] == 32
    assert score["positive_support_theme_count"] == 1
    assert score["zero_support_themes"] == [f"theme_{number:02d}" for number in range(1, 32)]
    assert len(score["supports"]) == 32
    assert score["supports"]["theme_31"] == 0
    assert score["macro_valid_topic"] == 100.0
    assert score["uniform_random_chance"] == pytest.approx(100 / 32)
    assert score["sum_weights"] == pytest.approx(1.0)
    rows = _rows([[f"theme_{index:02d}"] for index in range(32)])
    predictions = {row["id"]: "not_official" for row in rows}
    with pytest.raises(SourceError, match="illegal"):
        score_predictions(rows, predictions)


def test_supported_class_macro_and_chance_use_different_denominators() -> None:
    rows = _rows([[f"theme_{index:02d}"] for index in range(31)])
    predictions = {row["id"]: "theme_00" for row in rows}
    score = score_predictions(rows, predictions)
    assert score["positive_support_theme_count"] == 31
    assert score["offered_theme_count"] == 32
    assert score["macro_valid_topic"] == pytest.approx(100 / 31)
    assert score["uniform_random_chance"] == pytest.approx(100 / 32)
    expected_signed = ((1 / 31) - (1 / 32)) / (1 - (1 / 32))
    assert score["normalized_signed_unclipped"] == pytest.approx(100 * expected_signed)


def test_score_rejects_non_corpus_prediction_ids_and_invalid_gold_sets() -> None:
    rows = _rows([[f"theme_{index:02d}"] for index in range(32)])
    predictions = {row["id"]: "theme_00" for row in rows}
    predictions["extra"] = "theme_00"
    with pytest.raises(SourceError, match="exactly"):
        score_predictions(rows, predictions)
    rows[0]["accepted_labels"] = ["theme_00", "theme_00"]
    with pytest.raises(SourceError, match="accepted labels"):
        score_predictions(rows, {row["id"]: "theme_00" for row in rows})


def test_score_rejects_non_discriminative_all_class_gold() -> None:
    rows = _rows([list(_criteria()) for _ in range(32)])
    predictions = {row["id"]: "theme_00" for row in rows}
    with pytest.raises(SourceError, match="non-discriminative"):
        score_predictions(rows, predictions)


def _candidate(source_id: str, labels: tuple[str, ...], text: str | None = None) -> Candidate:
    return Candidate(
        source_id,
        text or f"text {source_id}",
        labels,
        f"https://official/{source_id}",
        "2026-01-01",
    )


def test_selection_is_stable_and_exact_deduplication_rules() -> None:
    criteria = _criteria()
    candidates = {
        str(index): _candidate(str(index), (f"theme_{index % 32:02d}",)) for index in range(700)
    }
    legacy = [{"text": f"text {index}"} for index in range(529)]
    selected_one, audit_one = select_rows(candidates, legacy, {"text 650"}, {"651"}, criteria)
    selected_two, audit_two = select_rows(candidates, legacy, {"text 650"}, {"651"}, criteria)
    assert [item[0].source_id for item in selected_one] == [
        item[0].source_id for item in selected_two
    ]
    assert len(selected_one) == 599
    assert all(item[0].text != "text 650" and item[0].source_id != "651" for item in selected_one)
    assert audit_one == audit_two
    assert all(value > 0 for value in audit_one["supports"].values())
    assert stable_key("17")[0] == "5cbc838105800b62b67cd411b403999e61fb321a7f89210d463b524f24c66fcb"


def test_conflicting_text_is_never_guessed() -> None:
    criteria = _criteria()
    candidates = {
        str(index): _candidate(str(index), (f"theme_{index % 32:02d}",)) for index in range(600)
    }
    candidates["600"] = _candidate("600", ("theme_00",), "text 0")
    legacy = [{"text": f"text {index}"} for index in range(529)]
    with pytest.raises(SourceError, match="expected 529"):
        select_rows(candidates, legacy, set(), set(), criteria)


def test_empty_topic_id_keeps_legacy_and_replacement_texts_ambiguous() -> None:
    criteria = _criteria()
    candidates = {
        str(index): _candidate(str(index), (f"theme_{index % 32:02d}",)) for index in range(600)
    }
    legacy = [{"text": f"text {index}"} for index in range(529)]
    all_text_to_ids = {candidate.text: {candidate.source_id} for candidate in candidates.values()}
    all_text_to_ids["text 0"] = {"0", "unlabeled-legacy"}
    all_text_to_ids["text 529"] = {"529", "unlabeled-replacement"}
    with pytest.raises(SourceError, match="expected 529"):
        select_rows(candidates, legacy, set(), set(), criteria, all_text_to_ids)


def _write_frozen_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    contents = {
        "proposicoes-2026.csv": b"proposals",
        "proposicoesTemas-2026.csv": b"topics",
        "temas-reference.json": b"reference",
        "camara_tema.train.jsonl": b"train",
        "camara_tema.dev.jsonl": b"dev",
        "camara_tema.test.jsonl": b"legacy-test",
    }
    for filename, content in contents.items():
        (sources_dir / filename).write_bytes(content)
    expected = {
        filename: {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "url": f"https://example.test/{filename}",
        }
        for filename, content in contents.items()
    }
    return sources_dir, sources_dir / "camara_tema.test.jsonl", expected


def test_frozen_sources_have_exact_official_urls_and_reject_changed_bytes(tmp_path: Path) -> None:
    expected_benchmark_url = (
        "https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/resolve/"
    )
    assert expected_benchmark_url == BENCHMARK_BASE_URL
    assert FROZEN_SOURCES["camara_tema.train.jsonl"]["url"] == (
        f"{expected_benchmark_url}{BENCHMARK_COMMIT}/data/camara_tema/train.jsonl"
    )
    assert FROZEN_SOURCES["camara_tema.dev.jsonl"]["url"] == (
        f"{expected_benchmark_url}{BENCHMARK_COMMIT}/data/camara_tema/dev.jsonl"
    )
    assert FROZEN_SOURCES["camara_tema.test.jsonl"]["url"] == (
        f"{expected_benchmark_url}{BENCHMARK_COMMIT}/data/camara_tema/test.jsonl"
    )
    assert FROZEN_SOURCES["proposicoes-2026.csv"]["url"] == (
        "https://dadosabertos.camara.leg.br/arquivos/proposicoes/csv/proposicoes-2026.csv"
    )
    assert FROZEN_SOURCES["proposicoesTemas-2026.csv"]["url"] == (
        "https://dadosabertos.camara.leg.br/arquivos/proposicoesTemas/csv/proposicoesTemas-2026.csv"
    )
    assert FROZEN_SOURCES["temas-reference.json"]["url"] == (
        "https://dadosabertos.camara.leg.br/api/v2/referencias/proposicoes/codTema"
    )

    sources_dir, legacy_test, expected = _write_frozen_fixture(tmp_path)
    validate_frozen_sources(sources_dir, legacy_test, expected)
    for filename in (
        "temas-reference.json",
        "camara_tema.train.jsonl",
        "camara_tema.dev.jsonl",
        "camara_tema.test.jsonl",
    ):
        path = legacy_test if filename == "camara_tema.test.jsonl" else sources_dir / filename
        original = path.read_bytes()
        path.write_bytes(original + b"!")
        with pytest.raises(SourceError, match="mismatched bytes or sha256"):
            validate_frozen_sources(sources_dir, legacy_test, expected)
        path.write_bytes(original)


def test_selected_proposal_url_must_exactly_match_its_numeric_id() -> None:
    criteria = _criteria()
    candidate = Candidate(
        "123",
        "official text",
        ("theme_00",),
        "https://dadosabertos.camara.leg.br/api/v2/proposicoes/456",
        "2026-01-01",
    )
    with pytest.raises(SourceError, match="does not exactly identify"):
        build_rows([(candidate, None)], criteria, "a" * 64)


@pytest.fixture
def identity_artifacts(tmp_path: Path) -> Path:
    criteria = _criteria()
    codes_to_keys = {str(index): key for index, key in enumerate(criteria, start=1)}
    source_identity = {
        "schema": "camara-v3-source-identity-v1",
        "retrieval_date": RETRIEVAL_DATE,
        "raw_sources": [
            {"file": filename, **FROZEN_SOURCES[filename]} for filename in SOURCE_FILENAMES
        ],
        "official_themes": {
            "count": len(criteria),
            "codes_to_keys": codes_to_keys,
            "codes_to_names": {code: criteria[key] for code, key in codes_to_keys.items()},
        },
        "rules": dict(SOURCE_IDENTITY_RULES),
        "seed": "42",
        "target_rows": 599,
        "algorithm_version": ALGORITHM_VERSION,
    }
    identity_hash = source_identity_sha256(source_identity)
    rows = [
        {
            "id": f"row-{index}",
            "sourceID": str(index),
            "text": f"self-authored text {index}",
            "criteria": criteria,
            "accepted_labels": [f"theme_{index % 32:02d}"],
            "provenance": {"source_manifest_sha256": identity_hash},
        }
        for index in range(599)
    ]
    dataset = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    manifest = {
        "schema": "camara-v3-source-manifest-v2",
        "source_manifest_sha256": identity_hash,
        "source_manifest_sha256_scope": SOURCE_IDENTITY_SCOPE,
        "source_identity_sha256": identity_hash,
        "source_identity": source_identity,
        "derived": {
            "file": "camara_tema_v3.test.jsonl",
            "rows": len(rows),
            "sha256": hashlib.sha256(dataset).hexdigest(),
        },
    }
    (tmp_path / "camara_tema_v3.test.jsonl").write_bytes(dataset)
    (tmp_path / "source-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return tmp_path


def test_source_identity_links_rows_and_detects_tampering(
    identity_artifacts: Path, tmp_path: Path
) -> None:
    manifest = validate_derived_artifacts(identity_artifacts)
    identity_hash = manifest["source_manifest_sha256"]
    assert manifest["source_identity_sha256"] == identity_hash
    assert manifest["source_identity"]["rules"] == SOURCE_IDENTITY_RULES
    assert all(
        json.loads(line)["provenance"]["source_manifest_sha256"] == identity_hash
        for line in (identity_artifacts / "camara_tema_v3.test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    changed_rules = tmp_path / "changed-rules"
    shutil.copytree(identity_artifacts, changed_rules)
    rules_manifest = json.loads(
        (changed_rules / "source-manifest.json").read_text(encoding="utf-8")
    )
    rules_manifest["source_identity"]["rules"]["selection"] = "changed"
    (changed_rules / "source-manifest.json").write_text(
        json.dumps(rules_manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(SourceError, match="invalid source identity hash"):
        validate_derived_artifacts(changed_rules)

    changed_row = tmp_path / "changed-row"
    shutil.copytree(identity_artifacts, changed_row)
    rows = [
        json.loads(line)
        for line in (changed_row / "camara_tema_v3.test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows[0]["provenance"]["source_manifest_sha256"] = "0" * 64
    (changed_row / "camara_tema_v3.test.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(SourceError, match="not linked"):
        validate_derived_artifacts(changed_row)
