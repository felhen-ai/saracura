"""Build and score the official 32-topic Camara V3 source corpus.

This module deliberately has no model or network dependency.  It consumes frozen
official snapshots and writes only derived, reproducible artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlsplit

TARGET_ROWS = 599
SEED = "42"
INSTRUCTIONS = "Qual é um dos temas oficiais desta proposição?"
BENCHMARK_COMMIT = "00f5d6dcc6679751723289889b8b0b28b71cbcb9"
BENCHMARK_BASE_URL = "https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench/resolve/"
BENCHMARK_DATA_PATH = "data/camara_tema"
RETRIEVAL_DATE = "2026-11-03"
ALGORITHM_VERSION = "camara-v3-selection-v1"
SOURCE_IDENTITY_SCOPE = (
    "sha256(canonical UTF-8 JSON source_identity; sorted keys, compact separators)"
)
WHITESPACE = re.compile(r"\s+", re.UNICODE)
SOURCE_FILENAMES = (
    "proposicoes-2026.csv",
    "proposicoesTemas-2026.csv",
    "temas-reference.json",
    "camara_tema.train.jsonl",
    "camara_tema.dev.jsonl",
    "camara_tema.test.jsonl",
)


class FrozenSource(TypedDict):
    bytes: int
    sha256: str
    url: str


# This is deliberately copied from the immutable source snapshot expectation,
# not regenerated from the files supplied to a build.
FROZEN_SOURCES: dict[str, FrozenSource] = {
    "proposicoes-2026.csv": {
        "bytes": 41284863,
        "sha256": "2989c95992dc25a10a4c9886a5b57fff67bf195b8eae0b7c690d93cbc414940e",
        "url": "https://dadosabertos.camara.leg.br/arquivos/proposicoes/csv/proposicoes-2026.csv",
    },
    "proposicoesTemas-2026.csv": {
        "bytes": 3088726,
        "sha256": "5f6fbdfb4a50ede59be61dc79588ef83e60d473c8962ce7d202da1323d5f68f8",
        "url": "https://dadosabertos.camara.leg.br/arquivos/proposicoesTemas/csv/proposicoesTemas-2026.csv",
    },
    "temas-reference.json": {
        "bytes": 2642,
        "sha256": "e20d8aeb7b88d8c4668e21d9aca9c0d07dd6805da481e69b012a20d3a4ec4807",
        "url": "https://dadosabertos.camara.leg.br/api/v2/referencias/proposicoes/codTema",
    },
    "camara_tema.train.jsonl": {
        "bytes": 67172925,
        "sha256": "85eeb3082fc7027fc3f35925a974aa19dc9bde4ecc3331923ebad289b44e6cb7",
        "url": (f"{BENCHMARK_BASE_URL}{BENCHMARK_COMMIT}/{BENCHMARK_DATA_PATH}/train.jsonl"),
    },
    "camara_tema.dev.jsonl": {
        "bytes": 1438318,
        "sha256": "ae89b36e1dd86eb864f0d6366a0fd55709d4ba64e1312d8c318b25e0cb354cfd",
        "url": f"{BENCHMARK_BASE_URL}{BENCHMARK_COMMIT}/{BENCHMARK_DATA_PATH}/dev.jsonl",
    },
    "camara_tema.test.jsonl": {
        "bytes": 2896063,
        "sha256": "3885484f11bac50dccba40725a036688098390f4a3af231f0e3fab1a1fa0a944",
        "url": f"{BENCHMARK_BASE_URL}{BENCHMARK_COMMIT}/{BENCHMARK_DATA_PATH}/test.jsonl",
    },
}
SOURCE_URLS = {filename: source["url"] for filename, source in FROZEN_SOURCES.items()}
SOURCE_IDENTITY_RULES = {
    "normalization": "NFC + Unicode whitespace collapse + strip; exact comparison only",
    "legacy_retention": "unique official ementa-to-proposal-ID match with official labels",
    "deduplication": (
        "exclude published train/dev normalized texts and every official ID linked to them; "
        "replacements exclude every legacy test text"
    ),
    "selection": (
        "sha256(seed42:proposalID), greedy maximum deficient theme count to support 5 where "
        "eligible, then hash order"
    ),
}
RETAINED_EXAMPLES = {
    139: ("2631818", {"direitos_humanos_e_minorias", "saude"}),
    396: (
        "2633316",
        {"direitos_humanos_e_minorias", "educacao", "financas_publicas_e_orcamento"},
    ),
}


class SourceError(ValueError):
    """A frozen source cannot produce a defensible corpus or score."""


def normalize_text(value: str) -> str:
    """Use precisely the D1 comparison normalization; do not fuzzy-match text."""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def slug_key(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _validate_https_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SourceError(f"source URL is not a canonical HTTPS URL: {url!r}")


def _source_paths(sources_dir: Path, legacy_test: Path) -> list[tuple[str, Path]]:
    return [
        ("proposicoes-2026.csv", sources_dir / "proposicoes-2026.csv"),
        ("proposicoesTemas-2026.csv", sources_dir / "proposicoesTemas-2026.csv"),
        ("temas-reference.json", sources_dir / "temas-reference.json"),
        ("camara_tema.train.jsonl", sources_dir / "camara_tema.train.jsonl"),
        ("camara_tema.dev.jsonl", sources_dir / "camara_tema.dev.jsonl"),
        ("camara_tema.test.jsonl", legacy_test),
    ]


def validate_frozen_sources(
    sources_dir: Path,
    legacy_test: Path,
    expected: Mapping[str, Mapping[str, Any]] = FROZEN_SOURCES,
) -> None:
    """Reject any byte change before a source is parsed.

    ``expected`` exists only to make this validator independently testable. The
    production builder always calls it without an override and therefore uses
    the pinned ``FROZEN_SOURCES`` mapping above.
    """
    required = [filename for filename, _ in _source_paths(sources_dir, legacy_test)]
    if set(expected) != set(required):
        raise SourceError("frozen source expectation must contain exactly the six required files")
    for filename, path in _source_paths(sources_dir, legacy_test):
        source = expected[filename]
        expected_bytes = source.get("bytes")
        expected_sha256 = source.get("sha256")
        url = source.get("url")
        if (
            not isinstance(expected_bytes, int)
            or expected_bytes < 1
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or not isinstance(url, str)
        ):
            raise SourceError(f"invalid frozen source expectation: {filename}")
        _validate_https_url(url)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise SourceError(f"cannot read frozen source {filename}: {error}") from error
        if len(raw) != expected_bytes or sha256_bytes(raw) != expected_sha256:
            raise SourceError(f"frozen source has mismatched bytes or sha256: {filename}")


def _frozen_source_manifest() -> list[dict[str, Any]]:
    return [{"file": filename, **FROZEN_SOURCES[filename]} for filename in SOURCE_FILENAMES]


def _validate_proposal_url(source_id: str, proposal_url: str) -> None:
    if not source_id.isascii() or not source_id.isdecimal():
        raise SourceError(f"selected proposal has non-numeric source ID: {source_id!r}")
    expected = f"https://dadosabertos.camara.leg.br/api/v2/proposicoes/{source_id}"
    _validate_https_url(proposal_url)
    if proposal_url != expected:
        raise SourceError(
            f"selected proposal URL does not exactly identify {source_id}: {proposal_url!r}"
        )


def stable_key(source_id: str) -> tuple[str, int | str]:
    numeric_id: int | str = int(source_id) if source_id.isdigit() else source_id
    return (sha256_bytes(f"{SEED}:{source_id}".encode()), numeric_id)


@dataclass(frozen=True)
class Candidate:
    source_id: str
    text: str
    accepted_labels: tuple[str, ...]
    proposal_url: str
    presentation_date: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise SourceError(f"cannot read JSONL {path.name}: {error}") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
    except OSError as error:
        raise SourceError(f"cannot read CSV {path.name}: {error}") from error
    if not rows or not rows[0]:
        raise SourceError(f"official CSV {path.name} has no complete rows")
    return rows


def official_themes(reference_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
        entries = payload["dados"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SourceError(f"cannot read official theme reference: {error}") from error
    code_to_slug: dict[str, str] = {}
    criteria: dict[str, str] = {}
    for entry in entries:
        code = str(entry["cod"])
        name = normalize_text(str(entry["nome"]))
        key = slug_key(name)
        if not code or not name or key in criteria or code in code_to_slug:
            raise SourceError("official theme reference has duplicate or empty code/name")
        code_to_slug[code] = key
        criteria[key] = name
    if len(criteria) != 32:
        raise SourceError(f"expected exactly 32 official themes, got {len(criteria)}")
    return code_to_slug, dict(sorted(criteria.items()))


def load_candidates(
    sources_dir: Path,
) -> tuple[dict[str, Candidate], dict[str, str], dict[str, set[str]]]:
    code_to_slug, criteria = official_themes(sources_dir / "temas-reference.json")
    proposals = _read_csv(sources_dir / "proposicoes-2026.csv")
    topic_rows = _read_csv(sources_dir / "proposicoesTemas-2026.csv")
    topics: dict[str, set[str]] = defaultdict(set)
    for row in topic_rows:
        proposal_url = row.get("uriProposicao", "")
        code = row.get("codTema", "")
        if not proposal_url or code not in code_to_slug:
            continue
        source_id = proposal_url.rstrip("/").rsplit("/", 1)[-1]
        topics[source_id].add(code_to_slug[code])
    candidates: dict[str, Candidate] = {}
    all_text_to_ids: dict[str, set[str]] = defaultdict(set)
    all_ids: set[str] = set()
    for row in proposals:
        source_id = row.get("id", "")
        text = normalize_text(row.get("ementa", ""))
        if not source_id or not text:
            continue
        if source_id in all_ids:
            raise SourceError(f"duplicate official proposal ID {source_id}")
        all_ids.add(source_id)
        all_text_to_ids[text].add(source_id)
        if source_id not in topics:
            continue
        candidates[source_id] = Candidate(
            source_id=source_id,
            text=text,
            accepted_labels=tuple(sorted(topics[source_id])),
            proposal_url=row.get("uri", ""),
            presentation_date=row.get("dataApresentacao", ""),
        )
    if not candidates:
        raise SourceError("no official ementa with an official theme")
    return candidates, criteria, all_text_to_ids


def _published_exclusions(
    all_text_to_ids: Mapping[str, set[str]], train_path: Path, dev_path: Path
) -> tuple[set[str], set[str]]:
    published_texts: set[str] = set()
    for path in (train_path, dev_path):
        for row in _read_jsonl(path):
            text = row.get("text")
            if isinstance(text, str):
                published_texts.add(normalize_text(text))
    known_ids = set().union(*(all_text_to_ids.get(text, set()) for text in published_texts))
    return published_texts, known_ids


def select_rows(
    candidates: Mapping[str, Candidate],
    legacy_rows: Sequence[Mapping[str, Any]],
    excluded_texts: set[str],
    excluded_ids: set[str],
    criteria: Mapping[str, str],
    all_text_to_ids: Mapping[str, set[str]] | None = None,
) -> tuple[list[tuple[Candidate, int | None]], dict[str, Any]]:
    """Retain exact uniquely attributable legacy rows, then apply D1's greedy fill."""
    if all_text_to_ids is None:
        text_to_ids: Mapping[str, set[str]] = defaultdict(set)
        for candidate in candidates.values():
            text_to_ids[candidate.text].add(candidate.source_id)
    else:
        text_to_ids = all_text_to_ids
    legacy_texts = {normalize_text(str(row.get("text", ""))) for row in legacy_rows}
    selected: list[tuple[Candidate, int | None]] = []
    selected_ids: set[str] = set()
    exclusions: Counter[str] = Counter()
    for index, row in enumerate(legacy_rows):
        text = normalize_text(str(row.get("text", "")))
        ids = text_to_ids.get(text, set())
        if text in excluded_texts:
            exclusions["legacy_train_dev_text"] += 1
        elif len(ids) != 1:
            exclusions["legacy_not_uniquely_attributable"] += 1
        else:
            source_id = next(iter(ids))
            if source_id not in candidates:
                exclusions["legacy_no_official_labels"] += 1
            elif source_id in excluded_ids:
                exclusions["legacy_known_train_dev_id"] += 1
            else:
                selected.append((candidates[source_id], index))
                selected_ids.add(source_id)
    if len(selected) != 529:
        raise SourceError(f"expected 529 eligible retained legacy rows, got {len(selected)}")

    ambiguous_replacement_texts = {
        candidate.text
        for candidate in candidates.values()
        if len(text_to_ids.get(candidate.text, set())) != 1
    }
    available = [
        candidate
        for candidate in candidates.values()
        if candidate.source_id not in selected_ids
        and candidate.source_id not in excluded_ids
        and candidate.text not in excluded_texts
        and candidate.text not in legacy_texts
        and candidate.text not in ambiguous_replacement_texts
    ]
    available.sort(key=lambda candidate: stable_key(candidate.source_id))
    supports = Counter(label for candidate, _ in selected for label in candidate.accepted_labels)
    remaining = list(available)
    while len(selected) < TARGET_ROWS:
        deficient = {label for label in criteria if supports[label] < 5}
        if not deficient:
            break
        ranked = sorted(
            remaining,
            key=lambda candidate: (
                -len(deficient.intersection(candidate.accepted_labels)),
                stable_key(candidate.source_id),
            ),
        )
        if not ranked or not deficient.intersection(ranked[0].accepted_labels):
            break
        chosen = ranked[0]
        selected.append((chosen, None))
        remaining.remove(chosen)
        supports.update(chosen.accepted_labels)
    for candidate in remaining:
        if len(selected) == TARGET_ROWS:
            break
        selected.append((candidate, None))
        supports.update(candidate.accepted_labels)
    if len(selected) != TARGET_ROWS:
        raise SourceError(f"cannot reach {TARGET_ROWS} rows; only {len(selected)} eligible")
    zero_support = [label for label in criteria if supports[label] == 0]
    audit = {
        "rules": dict(SOURCE_IDENTITY_RULES),
        "counts": {
            "legacy_total": len(legacy_rows),
            "legacy_retained": len(selected_ids),
            "replacements": TARGET_ROWS - len(selected_ids),
            "published_text_exclusions": len(excluded_texts),
            "known_published_id_exclusions": len(excluded_ids),
            "candidate_pool_after_exclusions": len(available),
            "replacement_ambiguous_text_exclusions": len(ambiguous_replacement_texts),
            "legacy_exclusions": dict(sorted(exclusions.items())),
        },
        "supports": {label: supports[label] for label in criteria},
        "offered_theme_count": len(criteria),
        "positive_support_theme_count": len(criteria) - len(zero_support),
        "zero_support_themes": zero_support,
        "minimum_five_best_effort": [label for label in criteria if supports[label] < 5],
        "limitation": "Only exact normalized text and known proposal IDs are excluded; unavailable upstream IDs do not establish semantic disjointness.",
    }
    return selected, audit


def build_rows(
    selected: Iterable[tuple[Candidate, int | None]],
    criteria: Mapping[str, str],
    source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, legacy_index in selected:
        _validate_proposal_url(candidate.source_id, candidate.proposal_url)
        row: dict[str, Any] = {
            "id": f"camara-v3-{candidate.source_id}",
            "sourceID": candidate.source_id,
            "textHash": text_hash(candidate.text),
            "text": candidate.text,
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": dict(criteria),
            "accepted_labels": list(candidate.accepted_labels),
            "provenance": {
                "source_url": candidate.proposal_url,
                "retrieval_date": "2026-11-03",
                "raw_source": "proposicoes-2026.csv + proposicoesTemas-2026.csv + temas-reference.json",
                "presentation_date": candidate.presentation_date,
                "source_manifest_sha256": source_manifest_sha256,
            },
        }
        if legacy_index is not None:
            row["legacy_frozen_index"] = legacy_index
        rows.append(row)
    return rows


def score_predictions(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, str]
) -> dict[str, Any]:
    """Return D2 balanced valid-topic top-1 metrics; this is not multilabel recall."""
    if not rows:
        raise SourceError("cannot score zero rows")
    criteria = rows[0].get("criteria")
    if not isinstance(criteria, Mapping) or len(criteria) != 32:
        raise SourceError("rows must carry exactly 32 official criteria")
    criteria_dict = dict(criteria)
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in criteria_dict.items()
    ):
        raise SourceError("criteria must be string keys and official names")
    if not isinstance(predictions, Mapping):
        raise SourceError("predictions must map every row id to one official label")
    official = set(criteria_dict)
    supports: Counter[str] = Counter()
    parsed: list[tuple[str, set[str]]] = []
    row_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        labels = row.get("accepted_labels")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise SourceError("rows must have unique immutable string ids")
        row_ids.add(row_id)
        row_criteria = row.get("criteria")
        if not isinstance(row_criteria, Mapping) or dict(row_criteria) != criteria_dict:
            raise SourceError("criteria must be identical across rows")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise SourceError("row lacks immutable id or accepted labels")
        accepted = set(labels)
        if not accepted <= official:
            raise SourceError(f"row {row_id} has non-official accepted labels")
        if row_id not in predictions:
            raise SourceError(f"missing prediction for {row_id}")
        prediction = predictions[row_id]
        if not isinstance(prediction, str) or prediction not in official:
            raise SourceError(f"illegal prediction for {row_id}: {prediction!r}")
        supports.update(accepted)
        parsed.append((prediction, accepted))
    if set(predictions) != row_ids:
        raise SourceError("predictions must have exactly the corpus row ids")
    positive_support = sorted(label for label in official if supports[label] > 0)
    if not positive_support:
        raise SourceError("coverage has no positive-support official themes")
    positive_count = len(positive_support)
    offered_count = len(official)
    class_hits = {label: 0.0 for label in official}
    raw_hits = 0
    chance = 0.0
    weight_sum = 0.0
    for prediction, accepted in parsed:
        hit = prediction in accepted
        raw_hits += int(hit)
        weight = sum(1.0 / supports[label] for label in accepted) / positive_count
        weight_sum += weight
        chance += weight * len(accepted) / offered_count
        for label in accepted:
            class_hits[label] += float(hit) / supports[label]
    macro = sum(class_hits[label] for label in positive_support) / positive_count
    constants = {
        label: sum(
            float(label in accepted)
            * sum(1.0 / supports[item] for item in accepted)
            / positive_count
            for _, accepted in parsed
        )
        for label in official
    }
    best_label = max(sorted(constants), key=constants.__getitem__)
    if chance >= 1.0:
        raise SourceError("non-discriminative accepted labels make chance one")
    signed_unclipped = (macro - chance) / (1.0 - chance)
    normalized = min(1.0, max(0.0, signed_unclipped))
    best_constant_score = constants[best_label]
    best_constant_signed_unclipped = (best_constant_score - chance) / (1.0 - chance)
    best_constant_normalized = min(1.0, max(0.0, best_constant_signed_unclipped))
    return {
        "metric": "balanced valid-topic top-1 accuracy (not multilabel recall)",
        "raw_any_topic_hit": 100.0 * raw_hits / len(parsed),
        "macro_valid_topic": 100.0 * macro,
        "normalized": 100.0 * normalized,
        "normalized_signed_unclipped": 100.0 * signed_unclipped,
        "uniform_random_chance": 100.0 * chance,
        "best_constant": {
            "label": best_label,
            "score": 100.0 * best_constant_score,
            "normalized": 100.0 * best_constant_normalized,
            "normalized_signed_unclipped": 100.0 * best_constant_signed_unclipped,
        },
        "offered_theme_count": offered_count,
        "positive_support_theme_count": positive_count,
        "zero_support_themes": sorted(official.difference(positive_support)),
        "sum_weights": weight_sum,
        "supports": {label: supports[label] for label in sorted(official)},
    }


def _artifact_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _write_idempotent(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise SourceError(f"refusing to overwrite non-identical derived artifact {path.name}")
        return
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            if path.read_bytes() != data:
                raise SourceError(
                    f"refusing to overwrite non-identical derived artifact {path.name}"
                ) from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _source_identity(
    code_to_slug: Mapping[str, str], criteria: Mapping[str, str]
) -> dict[str, Any]:
    codes_to_keys = dict(sorted(code_to_slug.items()))
    codes_to_names = {code: criteria[key] for code, key in codes_to_keys.items()}
    return {
        "schema": "camara-v3-source-identity-v1",
        "retrieval_date": RETRIEVAL_DATE,
        "raw_sources": _frozen_source_manifest(),
        "official_themes": {
            "count": len(criteria),
            "codes_to_keys": codes_to_keys,
            "codes_to_names": codes_to_names,
        },
        "rules": dict(SOURCE_IDENTITY_RULES),
        "seed": SEED,
        "target_rows": TARGET_ROWS,
        "algorithm_version": ALGORITHM_VERSION,
    }


def source_identity_sha256(source_identity: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json_bytes(source_identity))


def _selection_material(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    material: list[dict[str, Any]] = []
    for row in rows:
        source_id = row.get("sourceID")
        row_id = row.get("id")
        text = row.get("text")
        labels = row.get("accepted_labels")
        if (
            not isinstance(source_id, str)
            or not isinstance(row_id, str)
            or not isinstance(text, str)
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
        ):
            raise SourceError("derived selection has malformed id, text, or accepted labels")
        material.append(
            {
                "id": row_id,
                "sourceID": source_id,
                "text": text,
                "accepted_labels": sorted(labels),
            }
        )
    return material


def _validate_unchanged_selection(rows: Sequence[Mapping[str, Any]], prior_derived: Path) -> None:
    prior_rows = _read_jsonl(prior_derived)
    if len(prior_rows) != TARGET_ROWS:
        raise SourceError("prior derived artifact does not contain exactly 599 rows")
    if _selection_material(rows) != _selection_material(prior_rows):
        raise SourceError(
            "selected IDs, texts, or accepted label sets changed from prior derived artifact"
        )


def _validate_retained_examples(rows: Sequence[Mapping[str, Any]]) -> None:
    found: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        legacy_index = row.get("legacy_frozen_index")
        if isinstance(legacy_index, int) and legacy_index in RETAINED_EXAMPLES:
            found[legacy_index] = row
    for legacy_index, (source_id, labels) in RETAINED_EXAMPLES.items():
        retained_row = found.get(legacy_index)
        if retained_row is None:
            raise SourceError(
                f"required retained example is absent: legacy_frozen_index={legacy_index}"
            )
        if (
            retained_row.get("sourceID") != source_id
            or set(retained_row.get("accepted_labels", [])) != labels
        ):
            raise SourceError(
                f"required retained example changed: legacy_frozen_index={legacy_index}"
            )


def _validate_source_identity_contents(
    source_identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    if source_identity.get("schema") != "camara-v3-source-identity-v1":
        raise SourceError("source identity has an unsupported schema")
    if source_identity.get("retrieval_date") != RETRIEVAL_DATE:
        raise SourceError("source identity retrieval date changed")
    if source_identity.get("raw_sources") != _frozen_source_manifest():
        raise SourceError("source identity raw sources differ from the pinned six sources")
    if source_identity.get("rules") != SOURCE_IDENTITY_RULES:
        raise SourceError("source identity selection rules changed")
    if (
        source_identity.get("seed") != SEED
        or source_identity.get("target_rows") != TARGET_ROWS
        or source_identity.get("algorithm_version") != ALGORITHM_VERSION
    ):
        raise SourceError("source identity selection parameters changed")
    if not rows:
        raise SourceError("derived dataset has no rows")
    criteria = rows[0].get("criteria")
    if not isinstance(criteria, Mapping) or len(criteria) != 32:
        raise SourceError("derived dataset does not carry 32 official criteria")
    themes = source_identity.get("official_themes")
    if not isinstance(themes, Mapping) or themes.get("count") != len(criteria):
        raise SourceError("source identity official theme count differs from rows")
    codes_to_keys = themes.get("codes_to_keys")
    codes_to_names = themes.get("codes_to_names")
    if (
        not isinstance(codes_to_keys, Mapping)
        or not isinstance(codes_to_names, Mapping)
        or len(codes_to_keys) != len(criteria)
        or set(codes_to_keys) != set(codes_to_names)
        or set(codes_to_keys.values()) != set(criteria)
    ):
        raise SourceError("source identity official code/key mappings are invalid")
    for code, key in codes_to_keys.items():
        if (
            not isinstance(code, str)
            or not isinstance(key, str)
            or codes_to_names[code] != criteria[key]
        ):
            raise SourceError("source identity official code/name mappings are invalid")


def validate_derived_artifacts(output: Path) -> dict[str, Any]:
    """Verify the non-circular source identity and the derived JSONL it authorizes."""
    try:
        manifest = json.loads((output / "source-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceError(f"cannot read derived source manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "camara-v3-source-manifest-v2":
        raise SourceError("derived source manifest has an unsupported schema")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise SourceError("derived source manifest lacks source identity")
    identity_hash = source_identity_sha256(source_identity)
    if (
        manifest.get("source_manifest_sha256") != identity_hash
        or manifest.get("source_identity_sha256") != identity_hash
        or manifest.get("source_manifest_sha256_scope") != SOURCE_IDENTITY_SCOPE
    ):
        raise SourceError("derived source manifest has an invalid source identity hash")
    try:
        dataset = (output / "camara_tema_v3.test.jsonl").read_bytes()
        rows = [json.loads(line) for line in dataset.splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise SourceError(f"cannot read derived dataset: {error}") from error
    if len(rows) != TARGET_ROWS:
        raise SourceError("derived dataset must contain exactly 599 rows")
    _validate_source_identity_contents(source_identity, rows)
    for row in rows:
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("source_manifest_sha256") != identity_hash
        ):
            raise SourceError("derived row is not linked to the source identity")
    derived = manifest.get("derived")
    if (
        not isinstance(derived, Mapping)
        or derived.get("file") != "camara_tema_v3.test.jsonl"
        or derived.get("rows") != len(rows)
        or derived.get("sha256") != sha256_bytes(dataset)
    ):
        raise SourceError("derived dataset hash or row count does not match the manifest")
    return manifest


def build(
    sources_dir: Path,
    legacy_test: Path,
    output: Path,
    prior_derived: Path | None = None,
) -> dict[str, Any]:
    validate_frozen_sources(sources_dir, legacy_test)
    candidates, criteria, all_text_to_ids = load_candidates(sources_dir)
    legacy_rows = _read_jsonl(legacy_test)
    excluded_texts, excluded_ids = _published_exclusions(
        all_text_to_ids,
        sources_dir / "camara_tema.train.jsonl",
        sources_dir / "camara_tema.dev.jsonl",
    )
    selected, audit = select_rows(
        candidates, legacy_rows, excluded_texts, excluded_ids, criteria, all_text_to_ids
    )
    code_to_slug, _ = official_themes(sources_dir / "temas-reference.json")
    source_identity = _source_identity(code_to_slug, criteria)
    source_manifest_sha256 = source_identity_sha256(source_identity)
    rows = build_rows(selected, criteria, source_manifest_sha256)
    _validate_retained_examples(rows)
    previous = prior_derived or sources_dir.parent / "derived" / "camara_tema_v3.test.jsonl"
    _validate_unchanged_selection(rows, previous)
    dataset = _jsonl_bytes(rows)
    if len(rows) != TARGET_ROWS or len(dataset.splitlines()) != TARGET_ROWS:
        raise SourceError("derived dataset must contain exactly 599 compact JSONL rows")
    manifest = {
        "schema": "camara-v3-source-manifest-v2",
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_sha256_scope": SOURCE_IDENTITY_SCOPE,
        "source_identity_sha256": source_manifest_sha256,
        "source_identity": source_identity,
        "derived": {
            "file": "camara_tema_v3.test.jsonl",
            "rows": len(rows),
            "sha256": sha256_bytes(dataset),
        },
    }
    try:
        parsed_rows = [json.loads(line) for line in dataset.splitlines()]
    except json.JSONDecodeError as error:
        raise SourceError(f"derived dataset JSONL roundtrip failed: {error}") from error
    if len(parsed_rows) != TARGET_ROWS:
        raise SourceError("derived dataset JSONL roundtrip row count is incorrect")
    _write_idempotent(output / "camara_tema_v3.test.jsonl", dataset)
    _write_idempotent(output / "source-audit.json", _artifact_bytes(audit))
    _write_idempotent(output / "source-manifest.json", _artifact_bytes(manifest))
    return validate_derived_artifacts(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--legacy-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = build(arguments.sources_dir, arguments.legacy_test, arguments.output)
    except SourceError as error:
        print(f"source build failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"rows": manifest["derived"]["rows"], "sha256": manifest["derived"]["sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
