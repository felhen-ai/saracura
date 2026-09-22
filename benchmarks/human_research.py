"""Phase 4B.1 governance plus sanitized dispatch for the sealed 4B.2 stages.

Model construction, embedding extraction, and CPU-head training remain in
``benchmarks.human_training``.  This module has no calibration, blind
evaluation, runtime-calibration, network, or artifact discovery behavior.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never, cast

from benchmarks.first_party_gate import validate_protocol_bytes
from benchmarks.first_party_packet import (
    FAMILY_ID,
    GUIDE_PATH,
    LABELS,
    MANIFEST_PATH,
    SPLITS,
    STATE_ID,
    parse_states_bytes,
    validate_plan,
)
from saracura.research_trust import ResearchTrustError, canonical, verify_release_receipt
from saracura.verified_bytes import (
    VerifiedBytesError,
    open_verified_directory,
    read_verified_directory,
    read_verified_external_file,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HUMAN_ID = re.compile(r"^human_[0-9a-f]{16}$")
MAINTAINER_ID = re.compile(r"^maintainer_[0-9a-f]{16}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SAFE_CHILD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ANNOTATION_OUTCOMES = ("ambiguous", "out_of_scope", "rejected")
EXCLUSION_REASONS = (
    "backend_input_limit",
    "review_rejected",
    "review_disagreement_excluded",
    "adjudication_excluded",
)
CONTROL_IDS = (
    "artifact_manifest",
    "byte_hashes",
    "record_provenance",
    "human_author_identity",
    "independent_human_review",
    "privacy_record_review",
    "rights_record_review",
    "grouped_split",
    "cross_locale_family_link",
    "duplicate_scan",
    "contamination_scan",
    "shortcut_audit",
    "takedown_lineage",
)
INTAKE_LIMITS = {
    "base-states.jsonl": 16 * 1024 * 1024,
    "split-plan.json": 8 * 1024 * 1024,
    "contributors.json": 1 * 1024 * 1024,
    "annotations.jsonl": 32 * 1024 * 1024,
    "adjudications.jsonl": 16 * 1024 * 1024,
    "controls.json": 1 * 1024 * 1024,
    "takedown-ledger.jsonl": 8 * 1024 * 1024,
    "intake-manifest.json": 1 * 1024 * 1024,
}
RELEASE_INPUT_LIMITS = {
    "train.jsonl": 16 * 1024 * 1024,
    "dev.jsonl": 16 * 1024 * 1024,
    "calibration.jsonl": 16 * 1024 * 1024,
    "blind-test.jsonl": 16 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-request.json": 1 * 1024 * 1024,
    "controls.json": 1 * 1024 * 1024,
    "takedown-ledger.jsonl": 8 * 1024 * 1024,
    "release-input-manifest.json": 1 * 1024 * 1024,
}
VIEW_NAMES = {
    "train": "train.jsonl",
    "dev": "dev.jsonl",
    "calibration": "calibration.jsonl",
    "blind_test": "blind-test.jsonl",
}


class HumanResearchError(ValueError):
    """A bounded, non-secret Phase 4B.1 validation failure."""


class _SanitizedArgumentParser(argparse.ArgumentParser):
    """Do not echo operator paths or malformed values in research CLI errors."""

    def error(self, _message: str) -> Never:
        raise HumanResearchError("command arguments")


@dataclass(frozen=True, slots=True)
class Contributor:
    contributor_id: str
    identity_attestation_sha256: str
    contribution_grant_sha256: str
    review_grant_sha256: str
    grant_scopes: tuple[str, ...]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise HumanResearchError("duplicate JSON key")
        value[key] = child
    return value


def _nfc(value: object) -> None:
    if isinstance(value, str):
        if value != unicodedata.normalize("NFC", value):
            raise HumanResearchError("non-NFC string")
    elif isinstance(value, list):
        for child in value:
            _nfc(child)
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key != unicodedata.normalize("NFC", key):
                raise HumanResearchError("non-NFC key")
            _nfc(child)


def _json(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if (
        not raw
        or len(raw) > maximum
        or raw.startswith(b"\xef\xbb\xbf")
        or b"\r" in raw
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        raise HumanResearchError("canonical JSON framing")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(HumanResearchError("non-finite JSON")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, HumanResearchError) as exc:
        if isinstance(exc, HumanResearchError):
            raise
        raise HumanResearchError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise HumanResearchError("JSON object required")
    _nfc(value)
    if canonical(value) != raw[:-1]:
        raise HumanResearchError("noncanonical JSON")
    return value


def _jsonl(raw: bytes, *, maximum: int, allow_empty: bool = False) -> list[dict[str, Any]]:
    if allow_empty and raw == b"":
        return []
    if not raw or len(raw) > maximum or b"\r" in raw or not raw.endswith(b"\n"):
        raise HumanResearchError("canonical JSONL framing")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise HumanResearchError("canonical JSONL line")
        rows.append(_json(line, maximum=maximum))
    return rows


def _hash(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _integer(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _member(value: object, permitted: tuple[str, ...] | set[str]) -> bool:
    return isinstance(value, str) and value in permitted


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or TIMESTAMP.fullmatch(value) is None:
        raise HumanResearchError("timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise HumanResearchError("timestamp") from exc
    if parsed.tzinfo != UTC or parsed.microsecond:
        raise HumanResearchError("timestamp")
    return parsed


def _self_digest(value: dict[str, Any], field: str) -> str:
    if not _hash(value.get(field)):
        raise HumanResearchError("self digest")
    unsigned = dict(value)
    del unsigned[field]
    if digest(canonical(unsigned)) != value[field]:
        raise HumanResearchError("self digest")
    return cast(str, value[field])


def _ledger(values: dict[str, bytes], names: tuple[str, ...] | set[str]) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": len(values[name]), "sha256": digest(values[name])}
        for name in sorted(names)
    ]


def _validate_ledger(
    value: object, values: dict[str, bytes], names: set[str], *, allow_zero: bool = False
) -> None:
    if not isinstance(value, list) or len(value) != len(names):
        raise HumanResearchError("file ledger")
    if [item.get("path") for item in value if isinstance(item, dict)] != sorted(names):
        raise HumanResearchError("file ledger order")
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or not isinstance(item["path"], str)
            or item["path"] not in names
            or not _integer(item["bytes"], minimum=0 if allow_zero else 1)
            or not _hash(item["sha256"])
            or item["bytes"] != len(values[item["path"]])
            or item["sha256"] != digest(values[item["path"]])
        ):
            raise HumanResearchError("file ledger")


def _read_capsule(path: Path, limits: dict[str, int]) -> dict[str, bytes]:
    try:
        return dict(read_verified_directory(path, expected_files=limits))
    except VerifiedBytesError as exc:
        raise HumanResearchError("unsafe capsule input") from exc


def _known_protocol() -> tuple[dict[str, object], str]:
    protocol, _, protocol_digest = validate_protocol_bytes(MANIFEST_PATH.read_bytes())
    return protocol, protocol_digest


def _validate_contributors(raw: bytes, protocol_digest: str) -> dict[str, Contributor]:
    value = _json(raw, maximum=INTAKE_LIMITS["contributors.json"])
    expected = {"schema_version", "protocol_sha256", "contributors", "registry_sha256"}
    if (
        set(value) != expected
        or value["schema_version"] != "support-routing-contributors.v1"
        or value["protocol_sha256"] != protocol_digest
        or not isinstance(value["contributors"], list)
        or not value["contributors"]
    ):
        raise HumanResearchError("contributors schema")
    _self_digest(value, "registry_sha256")
    contributors: dict[str, Contributor] = {}
    ids: list[str] = []
    for item in value["contributors"]:
        fields = {
            "contributor_id",
            "identity_attestation_sha256",
            "contribution_grant_sha256",
            "review_grant_sha256",
            "grant_scopes",
            "verified_by",
            "verified_at",
            "revoked",
        }
        if not isinstance(item, dict) or set(item) != fields:
            raise HumanResearchError("contributor shape")
        scopes = item["grant_scopes"]
        _timestamp(item["verified_at"])
        verified_by = item["verified_by"]
        if (
            not isinstance(item["contributor_id"], str)
            or HUMAN_ID.fullmatch(item["contributor_id"]) is None
            or any(
                not _hash(item[name])
                for name in (
                    "identity_attestation_sha256",
                    "contribution_grant_sha256",
                    "review_grant_sha256",
                )
            )
            or not isinstance(scopes, list)
            or not scopes
            or any(type(scope) is not str for scope in scopes)
            or any(
                scope not in {"human_original_authoring", "independent_annotation", "adjudication"}
                for scope in scopes
            )
            or scopes != sorted(set(scopes))
            or not isinstance(verified_by, str)
            or MAINTAINER_ID.fullmatch(verified_by) is None
            or item["revoked"] is not False
        ):
            raise HumanResearchError("contributor values")
        contributor_id = item["contributor_id"]
        if contributor_id in contributors:
            raise HumanResearchError("duplicate contributor")
        ids.append(contributor_id)
        contributors[contributor_id] = Contributor(
            contributor_id,
            item["identity_attestation_sha256"],
            item["contribution_grant_sha256"],
            item["review_grant_sha256"],
            tuple(scopes),
        )
    if ids != sorted(ids):
        raise HumanResearchError("contributors order")
    return contributors


def _validate_annotations(
    raw: bytes,
    states: dict[str, dict[str, Any]],
    contributors: dict[str, Contributor],
    guide_digest: str,
) -> dict[str, list[dict[str, Any]]]:
    rows = _jsonl(raw, maximum=INTAKE_LIMITS["annotations.jsonl"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    prior: tuple[str, str] | None = None
    for row in rows:
        fields = {
            "schema_version",
            "state_id",
            "family_id",
            "content_digest",
            "reviewer_id",
            "reviewer_attestation_sha256",
            "review_grant_sha256",
            "annotation_guide_sha256",
            "reviewer_label",
            "privacy_review",
            "rights_review",
            "reviewed_at",
            "annotation_digest",
        }
        if (
            set(row) != fields
            or row["schema_version"] != "support-routing-independent-annotation.v1"
            or not isinstance(row["state_id"], str)
            or not isinstance(row["family_id"], str)
            or not isinstance(row["reviewer_id"], str)
            or HUMAN_ID.fullmatch(row["reviewer_id"]) is None
            or not _hash(row["content_digest"])
            or not _hash(row["reviewer_attestation_sha256"])
            or not _hash(row["review_grant_sha256"])
            or row["annotation_guide_sha256"] != guide_digest
            or not _member(row["reviewer_label"], {*LABELS, *ANNOTATION_OUTCOMES})
            or not _member(row["privacy_review"], {"approved_no_personal_data", "rejected"})
            or not _member(row["rights_review"], {"approved_first_party", "rejected"})
        ):
            raise HumanResearchError("annotation schema")
        _timestamp(row["reviewed_at"])
        _self_digest(row, "annotation_digest")
        state = states.get(row["state_id"])
        contributor = contributors.get(row["reviewer_id"])
        if (
            state is None
            or row["family_id"] != state["family_id"]
            or row["content_digest"] != state["content_digest"]
            or row["reviewer_id"] == state["author_id"]
            or contributor is None
            or "independent_annotation" not in contributor.grant_scopes
            or row["reviewer_attestation_sha256"] != contributor.identity_attestation_sha256
            or row["review_grant_sha256"] != contributor.review_grant_sha256
        ):
            raise HumanResearchError("annotation binding")
        order = (row["state_id"], row["reviewer_id"])
        if prior is not None and order <= prior:
            raise HumanResearchError("annotation order")
        prior = order
        grouped.setdefault(row["state_id"], []).append(row)
    return grouped


def _validate_adjudications(
    raw: bytes,
    states: dict[str, dict[str, Any]],
    contributors: dict[str, Contributor],
    guide_digest: str,
) -> dict[str, dict[str, Any]]:
    rows = _jsonl(raw, maximum=INTAKE_LIMITS["adjudications.jsonl"], allow_empty=True)
    result: dict[str, dict[str, Any]] = {}
    prior = ""
    for row in rows:
        fields = {
            "schema_version",
            "state_id",
            "family_id",
            "content_digest",
            "author_id",
            "candidate_label",
            "adjudicator_id",
            "adjudicator_attestation_sha256",
            "adjudication_grant_sha256",
            "annotation_guide_sha256",
            "annotation_digests",
            "selected_label",
            "adjudicated_at",
            "adjudication_digest",
        }
        if (
            set(row) != fields
            or row["schema_version"] != "support-routing-adjudication.v1"
            or not isinstance(row["state_id"], str)
            or not isinstance(row["family_id"], str)
            or not isinstance(row["author_id"], str)
            or not isinstance(row["adjudicator_id"], str)
            or HUMAN_ID.fullmatch(row["adjudicator_id"]) is None
            or not _member(row["candidate_label"], LABELS)
            or not _member(row["selected_label"], {*LABELS, *ANNOTATION_OUTCOMES})
            or not _hash(row["content_digest"])
            or not _hash(row["adjudicator_attestation_sha256"])
            or not _hash(row["adjudication_grant_sha256"])
            or row["annotation_guide_sha256"] != guide_digest
            or not isinstance(row["annotation_digests"], list)
            or not row["annotation_digests"]
            or any(not _hash(item) for item in row["annotation_digests"])
            or row["annotation_digests"] != sorted(set(row["annotation_digests"]))
        ):
            raise HumanResearchError("adjudication schema")
        _timestamp(row["adjudicated_at"])
        _self_digest(row, "adjudication_digest")
        state = states.get(row["state_id"])
        contributor = contributors.get(row["adjudicator_id"])
        if (
            row["state_id"] <= prior
            or state is None
            or row["family_id"] != state["family_id"]
            or row["content_digest"] != state["content_digest"]
            or row["author_id"] != state["author_id"]
            or row["candidate_label"] != state["candidate_label"]
            or row["adjudicator_id"] == state["author_id"]
            or contributor is None
            or "adjudication" not in contributor.grant_scopes
            or row["adjudicator_attestation_sha256"] != contributor.identity_attestation_sha256
            or row["adjudication_grant_sha256"] != contributor.review_grant_sha256
        ):
            raise HumanResearchError("adjudication binding")
        prior = row["state_id"]
        result[row["state_id"]] = row
    return result


def _validate_author_bindings(
    states: dict[str, dict[str, Any]], contributors: dict[str, Contributor]
) -> None:
    for state in states.values():
        author = contributors.get(state["author_id"])
        if (
            author is None
            or "human_original_authoring" not in author.grant_scopes
            or state["author_attestation_sha256"] != author.identity_attestation_sha256
            or not _hash(author.contribution_grant_sha256)
        ):
            raise HumanResearchError("author contributor binding")


def _validate_reference_sets(
    states: dict[str, dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    adjudications: dict[str, dict[str, Any]],
    controls: list[dict[str, Any]],
    contributors: dict[str, Contributor],
) -> None:
    referenced_contributors = {state["author_id"] for state in states.values()}
    referenced_contributors.update(
        review["reviewer_id"] for rows in annotations.values() for review in rows
    )
    referenced_contributors.update(row["adjudicator_id"] for row in adjudications.values())
    referenced_contributors.update(row["reviewer_id"] for row in controls)
    if set(contributors) != referenced_contributors:
        raise HumanResearchError("unreferenced contributor")


def _validate_controls(
    raw: bytes, *, policy_digest: str, inputs: dict[str, str], contributors: dict[str, Contributor]
) -> None:
    value = _json(raw, maximum=INTAKE_LIMITS["controls.json"])
    fields = {
        "schema_version",
        "policy_registry_sha256",
        "packet_input_digests",
        "controls",
        "controls_sha256",
    }
    if (
        set(value) != fields
        or value["schema_version"] != "support-routing-artifact-controls.v1"
        or value["policy_registry_sha256"] != policy_digest
        or value["packet_input_digests"] != inputs
        or not isinstance(value["controls"], list)
    ):
        raise HumanResearchError("controls schema")
    _self_digest(value, "controls_sha256")
    controls = value["controls"]
    if [item.get("control_id") for item in controls if isinstance(item, dict)] != list(CONTROL_IDS):
        raise HumanResearchError("controls order")
    for item in controls:
        required = {
            "control_id",
            "status",
            "scope",
            "method_id",
            "evidence_sha256",
            "reviewer_id",
            "reviewed_at",
        }
        reviewer = item.get("reviewer_id") if isinstance(item, dict) else None
        contributor = contributors.get(reviewer) if isinstance(reviewer, str) else None
        if (
            not isinstance(item, dict)
            or set(item) != required
            or item["status"] != "completed"
            or item["scope"] != "packet"
            or not isinstance(item["method_id"], str)
            or not item["method_id"].endswith(".v1")
            or item["method_id"].startswith("simulated-")
            or not _hash(item["evidence_sha256"])
            or not isinstance(reviewer, str)
            or HUMAN_ID.fullmatch(reviewer) is None
            or contributor is None
            or "independent_annotation" not in contributor.grant_scopes
        ):
            raise HumanResearchError("control entry")
        _timestamp(item["reviewed_at"])
    if controls[8]["method_id"] != "ptbr-only-no-cross-locale-families.v1":
        raise HumanResearchError("cross-locale control")


def _validate_takedown(raw: bytes) -> None:
    rows = _jsonl(raw, maximum=INTAKE_LIMITS["takedown-ledger.jsonl"])
    if len(rows) != 1:
        raise HumanResearchError("takedown event count")
    row = rows[0]
    fields = {
        "schema_version",
        "event",
        "previous_event_digest",
        "source_atom_id",
        "affected_state_ids",
        "affected_family_ids",
        "artifact_digests",
        "reason",
        "occurred_at",
        "disposition",
        "event_digest",
    }
    if (
        set(row) != fields
        or row["schema_version"] != "support-routing-takedown-event.v1"
        or row["event"] != "genesis"
        or row["previous_event_digest"] is not None
        or row["source_atom_id"] is not None
        or row["affected_state_ids"] != []
        or row["affected_family_ids"] != []
        or row["artifact_digests"] != []
        or row["reason"] != "genesis"
        or row["disposition"] != "active"
    ):
        raise HumanResearchError("takedown genesis")
    _timestamp(row["occurred_at"])
    _self_digest(row, "event_digest")


def _reconcile(
    states: dict[str, dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    adjudications: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    views: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    excluded: Counter[str] = Counter()
    for state_id, state in states.items():
        assignment = assignments.get(state_id)
        if assignment is None:
            raise HumanResearchError("split assignment")
        adjudication = adjudications.get(state_id)
        if len(state["text"]) > 320 or len(state["text"].encode("utf-8")) > 1280:
            if adjudication is not None:
                raise HumanResearchError("premature adjudication")
            excluded["backend_input_limit"] += 1
            continue
        reviews = annotations.get(state_id, [])
        expected = 1 if assignment["split"] == "train" else 2
        if len(reviews) != expected or len({item["reviewer_id"] for item in reviews}) != expected:
            raise HumanResearchError("annotation cardinality")
        if any(
            review["privacy_review"] != "approved_no_personal_data"
            or review["rights_review"] != "approved_first_party"
            or review["reviewer_label"] in ANNOTATION_OUTCOMES
            for review in reviews
        ):
            if adjudication is not None:
                raise HumanResearchError("premature adjudication")
            excluded["review_rejected"] += 1
            continue
        labels = [review["reviewer_label"] for review in reviews]
        disagrees = any(label != state["candidate_label"] for label in labels)
        if not disagrees:
            if adjudication is not None:
                raise HumanResearchError("unnecessary adjudication")
        else:
            if adjudication is None:
                excluded["review_disagreement_excluded"] += 1
                continue
            if adjudication["adjudicator_id"] in {
                state["author_id"],
                *(review["reviewer_id"] for review in reviews),
            } or adjudication["annotation_digests"] != sorted(
                review["annotation_digest"] for review in reviews
            ):
                raise HumanResearchError("adjudication independence")
            if adjudication["selected_label"] != state["candidate_label"]:
                excluded["adjudication_excluded"] += 1
                continue
        views[assignment["split"]].append(
            {
                "state_id": state_id,
                "family_id": state["family_id"],
                "locale": "pt-BR",
                "candidate_label": state["candidate_label"],
                "text": state["text"],
                "content_digest": state["content_digest"],
            }
        )
    if set(adjudications) - set(states):
        raise HumanResearchError("orphan adjudication")
    return views, excluded


def _minimums(
    views: dict[str, list[dict[str, Any]]],
    states: dict[str, dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
) -> None:
    needed = {"train": 200, "dev": 100, "calibration": 100, "blind_test": 100}
    if sum(len(rows) for rows in views.values()) < 500 or any(
        len(views[split]) < count for split, count in needed.items()
    ):
        raise HumanResearchError("accepted packet minimum")
    if any(
        sum(row["candidate_label"] == label for row in views[split]) < 10
        for split in SPLITS
        for label in LABELS
    ):
        raise HumanResearchError("accepted label minimum")
    accepted_ids = {row["state_id"] for rows in views.values() for row in rows}
    authors = Counter(states[state_id]["author_id"] for state_id in accepted_ids)
    reviewers = Counter(
        review["reviewer_id"] for state_id in accepted_ids for review in annotations[state_id]
    )
    if (
        len(authors) < 2
        or len(reviewers) < 3
        or any(count < 2 for count in authors.values())
        or any(count < 2 for count in reviewers.values())
    ):
        raise HumanResearchError("independence minimum")


def _view_bytes(views: dict[str, list[dict[str, Any]]]) -> dict[str, bytes]:
    return {
        VIEW_NAMES[split]: b"".join(
            canonical(row) + b"\n" for row in sorted(rows, key=lambda item: item["state_id"])
        )
        for split, rows in views.items()
    }


def _packet_manifest(
    values: dict[str, bytes],
    views: dict[str, list[dict[str, Any]]],
    excluded: Counter[str],
    protocol_digest: str,
    policy_digest: str,
) -> dict[str, Any]:
    view_raw = _view_bytes(views)
    counts = {split: len(views[split]) for split in SPLITS}
    labels = {
        split: {
            label: sum(row["candidate_label"] == label for row in views[split]) for label in LABELS
        }
        for split in SPLITS
    }
    manifest: dict[str, Any] = {
        "schema_version": "support-routing-human-packet.v1",
        "protocol_sha256": protocol_digest,
        "policy_registry_sha256": policy_digest,
        "states_sha256": digest(values["base-states.jsonl"]),
        "split_plan_sha256": digest(values["split-plan.json"]),
        "contributors_sha256": digest(values["contributors.json"]),
        "annotations_sha256": digest(values["annotations.jsonl"]),
        "adjudications_sha256": digest(values["adjudications.jsonl"]),
        "controls_sha256": digest(values["controls.json"]),
        "takedown_ledger_sha256": digest(values["takedown-ledger.jsonl"]),
        "files": {
            ("blind_test" if name == "blind-test.jsonl" else name.removesuffix(".jsonl")): {
                "bytes": len(raw),
                "sha256": digest(raw),
            }
            for name, raw in view_raw.items()
        },
        "accepted_counts": {
            "total": sum(counts.values()),
            "by_split": counts,
            "by_split_label": labels,
        },
        "excluded_counts": {
            "total": sum(excluded.values()),
            "by_reason": {reason: excluded[reason] for reason in EXCLUSION_REASONS},
        },
        "labels": list(LABELS),
        "split_algorithm": "support-routing-stratified-sha256.v1",
        "source_type": "human_original",
        "locale": "pt-BR",
        "synthetic_only": False,
        "authorizations": {
            "training": False,
            "calibration": False,
            "blind_test": False,
            "automation": False,
            "broad_quality_claims": False,
        },
    }
    manifest["packet_sha256"] = digest(canonical(manifest))
    return manifest


def _validate_packet(packet_raw: bytes, values: dict[str, bytes]) -> dict[str, Any]:
    packet = _json(packet_raw, maximum=RELEASE_INPUT_LIMITS["packet-manifest.json"])
    required = {
        "schema_version",
        "protocol_sha256",
        "policy_registry_sha256",
        "states_sha256",
        "split_plan_sha256",
        "contributors_sha256",
        "annotations_sha256",
        "adjudications_sha256",
        "controls_sha256",
        "takedown_ledger_sha256",
        "files",
        "accepted_counts",
        "excluded_counts",
        "labels",
        "split_algorithm",
        "source_type",
        "locale",
        "synthetic_only",
        "authorizations",
        "packet_sha256",
    }
    if (
        set(packet) != required
        or packet["schema_version"] != "support-routing-human-packet.v1"
        or any(
            not _hash(packet[name])
            for name in required
            if name.endswith("_sha256") and name != "packet_sha256"
        )
        or packet["labels"] != list(LABELS)
        or packet["split_algorithm"] != "support-routing-stratified-sha256.v1"
        or packet["source_type"] != "human_original"
        or packet["locale"] != "pt-BR"
        or packet["synthetic_only"] is not False
        or packet["authorizations"]
        != {
            "training": False,
            "calibration": False,
            "blind_test": False,
            "automation": False,
            "broad_quality_claims": False,
        }
    ):
        raise HumanResearchError("packet manifest schema")
    _self_digest(packet, "packet_sha256")
    protocol, protocol_digest = _known_protocol()
    if (
        packet["protocol_sha256"] != protocol_digest
        or packet["policy_registry_sha256"] != protocol["policy_registry_sha256"]
        or packet["controls_sha256"] != digest(values["controls.json"])
        or packet["takedown_ledger_sha256"] != digest(values["takedown-ledger.jsonl"])
    ):
        raise HumanResearchError("packet protocol binding")
    _validate_takedown(values["takedown-ledger.jsonl"])
    files = packet["files"]
    if not isinstance(files, dict) or set(files) != set(SPLITS):
        raise HumanResearchError("packet view ledger")
    for split in SPLITS:
        file_name = VIEW_NAMES[split]
        item = files[split]
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "sha256"}
            or not _integer(item["bytes"], minimum=1)
            or not _hash(item["sha256"])
            or item["bytes"] != len(values[file_name])
            or item["sha256"] != digest(values[file_name])
        ):
            raise HumanResearchError("packet view ledger")
    counts = packet["accepted_counts"]
    excluded = packet["excluded_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"total", "by_split", "by_split_label"}
        or not isinstance(excluded, dict)
        or set(excluded) != {"total", "by_reason"}
    ):
        raise HumanResearchError("packet counts")
    if (
        not _integer(counts["total"], minimum=1)
        or not isinstance(counts["by_split"], dict)
        or not isinstance(counts["by_split_label"], dict)
        or set(counts["by_split"]) != set(SPLITS)
        or set(counts["by_split_label"]) != set(SPLITS)
    ):
        raise HumanResearchError("packet counts")
    total = 0
    state_ids: set[str] = set()
    family_ids: set[str] = set()
    for split in SPLITS:
        rows = _jsonl(values[VIEW_NAMES[split]], maximum=RELEASE_INPUT_LIMITS[VIEW_NAMES[split]])
        view_state_ids: list[str] = []
        for row in rows:
            state_id = row.get("state_id")
            if not isinstance(state_id, str):
                raise HumanResearchError("view order")
            view_state_ids.append(state_id)
        if view_state_ids != sorted(view_state_ids):
            raise HumanResearchError("view order")
        by_label: Counter[str] = Counter()
        families: set[str] = set()
        for row in rows:
            fields = {
                "state_id",
                "family_id",
                "locale",
                "candidate_label",
                "text",
                "content_digest",
            }
            if (
                set(row) != fields
                or row["locale"] != "pt-BR"
                or not _member(row["candidate_label"], LABELS)
                or not isinstance(row["state_id"], str)
                or STATE_ID.fullmatch(row["state_id"]) is None
                or not isinstance(row["family_id"], str)
                or FAMILY_ID.fullmatch(row["family_id"]) is None
                or not isinstance(row["text"], str)
                or len(row["text"]) > 320
                or len(row["text"].encode("utf-8")) > 1280
                or not _hash(row["content_digest"])
                or row["family_id"] in families
                or row["state_id"] in state_ids
                or row["family_id"] in family_ids
            ):
                raise HumanResearchError("view schema")
            families.add(row["family_id"])
            state_ids.add(row["state_id"])
            family_ids.add(row["family_id"])
            by_label[row["candidate_label"]] += 1
        if counts["by_split"][split] != len(rows) or counts["by_split_label"][split] != {
            label: by_label[label] for label in LABELS
        }:
            raise HumanResearchError("packet count reconciliation")
        total += len(rows)
    if (
        counts["total"] != total
        or total < 500
        or any(not _integer(counts["by_split"][split], minimum=1) for split in SPLITS)
        or any(
            counts["by_split"][split] < minimum
            for split, minimum in zip(SPLITS, (200, 100, 100, 100), strict=True)
        )
        or any(
            not isinstance(counts["by_split_label"][split], dict)
            or set(counts["by_split_label"][split]) != set(LABELS)
            or any(
                not _integer(counts["by_split_label"][split][label], minimum=10) for label in LABELS
            )
            for split in SPLITS
        )
        or not _integer(excluded["total"])
        or not isinstance(excluded["by_reason"], dict)
        or set(excluded["by_reason"]) != set(EXCLUSION_REASONS)
        or any(not _integer(value) for value in excluded["by_reason"].values())
        or excluded["total"] != sum(excluded["by_reason"].values())
    ):
        raise HumanResearchError("packet count reconciliation")
    return packet


def validate_packet_view_subset(packet_raw: bytes, views: dict[str, bytes]) -> dict[str, Any]:
    """Validate signed-packet claims for an explicitly supplied view subset.

    Released stage capsules deliberately do not contain siblings from the other
    three splits.  This validator consequently proves the *complete* closed
    packet manifest and proves the byte ledger, ordering, labels, family
    isolation, and counters for the supplied views.  The signed receipt binds
    the remaining packet inputs; callers must verify that receipt separately.
    """

    if not views or set(views) - {"train", "dev", "calibration", "blind_test"}:
        raise HumanResearchError("packet view subset")
    packet = _json(packet_raw, maximum=RELEASE_INPUT_LIMITS["packet-manifest.json"])
    required = {
        "schema_version",
        "protocol_sha256",
        "policy_registry_sha256",
        "states_sha256",
        "split_plan_sha256",
        "contributors_sha256",
        "annotations_sha256",
        "adjudications_sha256",
        "controls_sha256",
        "takedown_ledger_sha256",
        "files",
        "accepted_counts",
        "excluded_counts",
        "labels",
        "split_algorithm",
        "source_type",
        "locale",
        "synthetic_only",
        "authorizations",
        "packet_sha256",
    }
    digest_fields = {
        "protocol_sha256",
        "policy_registry_sha256",
        "states_sha256",
        "split_plan_sha256",
        "contributors_sha256",
        "annotations_sha256",
        "adjudications_sha256",
        "controls_sha256",
        "takedown_ledger_sha256",
    }
    if (
        set(packet) != required
        or packet["schema_version"] != "support-routing-human-packet.v1"
        or any(not _hash(packet[name]) for name in digest_fields)
        or packet["labels"] != list(LABELS)
        or packet["split_algorithm"] != "support-routing-stratified-sha256.v1"
        or packet["source_type"] != "human_original"
        or packet["locale"] != "pt-BR"
        or packet["synthetic_only"] is not False
        or packet["authorizations"]
        != {
            "training": False,
            "calibration": False,
            "blind_test": False,
            "automation": False,
            "broad_quality_claims": False,
        }
    ):
        raise HumanResearchError("packet manifest schema")
    _self_digest(packet, "packet_sha256")
    files = packet["files"]
    counts = packet["accepted_counts"]
    excluded = packet["excluded_counts"]
    if (
        not isinstance(files, dict)
        or set(files) != set(SPLITS)
        or not isinstance(counts, dict)
        or set(counts) != {"total", "by_split", "by_split_label"}
        or not _integer(counts["total"], minimum=500)
        or not isinstance(counts["by_split"], dict)
        or set(counts["by_split"]) != set(SPLITS)
        or not isinstance(counts["by_split_label"], dict)
        or set(counts["by_split_label"]) != set(SPLITS)
        or not isinstance(excluded, dict)
        or set(excluded) != {"total", "by_reason"}
        or not _integer(excluded["total"])
        or not isinstance(excluded["by_reason"], dict)
        or set(excluded["by_reason"]) != set(EXCLUSION_REASONS)
        or any(not _integer(item) for item in excluded["by_reason"].values())
        or excluded["total"] != sum(excluded["by_reason"].values())
    ):
        raise HumanResearchError("packet counts")
    claimed_total = 0
    for split in SPLITS:
        ledger = files[split]
        labels = counts["by_split_label"][split]
        count = counts["by_split"][split]
        if (
            not isinstance(ledger, dict)
            or set(ledger) != {"bytes", "sha256"}
            or not _integer(ledger["bytes"], minimum=1)
            or not _hash(ledger["sha256"])
            or not _integer(count, minimum=(200 if split == "train" else 100))
            or not isinstance(labels, dict)
            or set(labels) != set(LABELS)
            or any(not _integer(labels[label], minimum=10) for label in LABELS)
            or sum(cast(int, labels[label]) for label in LABELS) != count
        ):
            raise HumanResearchError("packet counts")
        claimed_total += cast(int, count)
    if claimed_total != counts["total"]:
        raise HumanResearchError("packet count reconciliation")

    state_ids: set[str] = set()
    family_ids: set[str] = set()
    for split, raw in views.items():
        file_name = VIEW_NAMES[split]
        if files[split]["bytes"] != len(raw) or files[split]["sha256"] != digest(raw):
            raise HumanResearchError("packet view ledger")
        rows = _jsonl(raw, maximum=RELEASE_INPUT_LIMITS[file_name])
        expected_labels: Counter[str] = Counter()
        prior = ""
        for row in rows:
            fields = {
                "state_id",
                "family_id",
                "locale",
                "candidate_label",
                "text",
                "content_digest",
            }
            if (
                set(row) != fields
                or not isinstance(row["state_id"], str)
                or STATE_ID.fullmatch(row["state_id"]) is None
                or row["state_id"] <= prior
                or not isinstance(row["family_id"], str)
                or FAMILY_ID.fullmatch(row["family_id"]) is None
                or row["state_id"] in state_ids
                or row["family_id"] in family_ids
                or row["locale"] != "pt-BR"
                or not _member(row["candidate_label"], LABELS)
                or not isinstance(row["text"], str)
                or not row["text"]
                or len(row["text"]) > 320
                or len(row["text"].encode("utf-8")) > 1280
                or not _hash(row["content_digest"])
            ):
                raise HumanResearchError("packet view schema")
            prior = row["state_id"]
            state_ids.add(row["state_id"])
            family_ids.add(row["family_id"])
            expected_labels[row["candidate_label"]] += 1
        if (
            len(rows) != counts["by_split"][split]
            or {label: expected_labels[label] for label in LABELS}
            != counts["by_split_label"][split]
        ):
            raise HumanResearchError("packet view reconciliation")
    return packet


def _safe_name(value: str) -> str:
    if SAFE_CHILD.fullmatch(value) is None or value in {".", ".."}:
        raise HumanResearchError("unsafe output child")
    return value


def _write_at(directory_fd: int, name: str, raw: bytes) -> None:
    _safe_name(name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise HumanResearchError("immutable output conflict") from exc
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise HumanResearchError("output write")
            view = view[written:]
        os.fsync(fd)
    except OSError as exc:
        raise HumanResearchError("output write") from exc
    finally:
        os.close(fd)


def _mkdir_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        child_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(child_fd, 0o700)
            return child_fd
        except OSError:
            os.close(child_fd)
            raise
    except OSError as exc:
        raise HumanResearchError("immutable output conflict") from exc


def _fsync(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        raise HumanResearchError("output fsync") from exc


def _rename_no_replace(parent_fd: int, stage: str, final: str) -> None:
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.renameatx_np(parent_fd, stage.encode(), parent_fd, final.encode(), 0x00000004)
        if result != 0:
            raise HumanResearchError("immutable output conflict")
        return
    raise HumanResearchError("atomic no-replace directory install unavailable")


def _stage(parent: Path, final_name: str) -> tuple[int, str, int]:
    parent_fd = open_verified_directory(parent)
    stage_name = f".{final_name}.stage-{os.urandom(8).hex()}"
    try:
        stage_fd = _mkdir_at(parent_fd, stage_name)
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, stage_name, stage_fd


def _finish_stage(parent_fd: int, stage_name: str, stage_fd: int, final_name: str) -> None:
    _fsync(stage_fd)
    _fsync(parent_fd)
    _rename_no_replace(parent_fd, stage_name, final_name)
    _fsync(parent_fd)


def _intake(
    values: dict[str, bytes],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, object],
    str,
    str,
]:
    manifest = _json(values["intake-manifest.json"], maximum=INTAKE_LIMITS["intake-manifest.json"])
    required = {
        "schema_version",
        "files",
        "protocol_sha256",
        "policy_registry_sha256",
        "evidence_mode",
        "manifest_sha256",
    }
    if (
        set(manifest) != required
        or manifest["schema_version"] != "support-routing-human-intake.v1"
        or manifest["evidence_mode"] != "real_human"
    ):
        raise HumanResearchError("real-human intake required")
    _self_digest(manifest, "manifest_sha256")
    _validate_ledger(
        manifest["files"],
        values,
        set(INTAKE_LIMITS) - {"intake-manifest.json"},
        allow_zero=True,
    )
    protocol, protocol_digest = _known_protocol()
    policy_digest_value = protocol["policy_registry_sha256"]
    if not isinstance(policy_digest_value, str) or not _hash(policy_digest_value):
        raise HumanResearchError("protocol policy digest")
    policy_digest = policy_digest_value
    if (
        manifest["protocol_sha256"] != protocol_digest
        or manifest["policy_registry_sha256"] != policy_digest
    ):
        raise HumanResearchError("intake protocol binding")
    states_tuple = parse_states_bytes(values["base-states.jsonl"])
    states = {state.state_id: state.value for state in states_tuple}
    plan = _json(values["split-plan.json"], maximum=INTAKE_LIMITS["split-plan.json"])
    validate_plan(
        states_tuple,
        plan,
        plan.get("seed", ""),
        protocol_digest,
        digest(values["base-states.jsonl"]),
    )
    assignments: dict[str, dict[str, Any]] = {}
    for item in plan["assignments"]:
        if not isinstance(item, dict) or not isinstance(item.get("state_id"), str):
            raise HumanResearchError("split assignment")
        assignments[item["state_id"]] = item
    if set(assignments) != set(states) or len(assignments) != len(states):
        raise HumanResearchError("split assignment")
    contributors = _validate_contributors(values["contributors.json"], protocol_digest)
    _validate_author_bindings(states, contributors)
    guide_digest = digest(GUIDE_PATH.read_bytes())
    annotations = _validate_annotations(
        values["annotations.jsonl"], states, contributors, guide_digest
    )
    adjudications = _validate_adjudications(
        values["adjudications.jsonl"], states, contributors, guide_digest
    )
    inputs = {
        "protocol": protocol_digest,
        "states": digest(values["base-states.jsonl"]),
        "split_plan": digest(values["split-plan.json"]),
        "contributors": digest(values["contributors.json"]),
        "annotations": digest(values["annotations.jsonl"]),
        "adjudications": digest(values["adjudications.jsonl"]),
        "takedown_ledger": digest(values["takedown-ledger.jsonl"]),
    }
    _validate_controls(
        values["controls.json"],
        policy_digest=policy_digest,
        inputs=inputs,
        contributors=contributors,
    )
    controls = _json(values["controls.json"], maximum=INTAKE_LIMITS["controls.json"])
    _validate_takedown(values["takedown-ledger.jsonl"])
    _validate_reference_sets(
        states,
        annotations,
        adjudications,
        controls["controls"],
        contributors,
    )
    return states, assignments, annotations, adjudications, protocol, protocol_digest, policy_digest


def materialize_packet(intake_capsule: Path, output_parent: Path, packet_name: str) -> Path:
    """Materialize a real-human unsigned release-input capsule, never a release."""

    final_name = _safe_name(packet_name)
    values = _read_capsule(intake_capsule, INTAKE_LIMITS)
    states, assignments, annotations, adjudications, _, protocol_digest, policy_digest = _intake(
        values
    )
    views, excluded = _reconcile(states, assignments, annotations, adjudications)
    _minimums(views, states, annotations)
    view_raw = _view_bytes(views)
    packet = _packet_manifest(values, views, excluded, protocol_digest, policy_digest)
    packet_raw = canonical(packet) + b"\n"
    request = {
        "schema_version": "packet-release-request.v1",
        "packet_sha256": packet["packet_sha256"],
        "evidence_mode": "real_human",
    }
    request["request_sha256"] = digest(canonical(request))
    output = dict(view_raw)
    output.update(
        {
            "packet-manifest.json": packet_raw,
            "packet-release-request.json": canonical(request) + b"\n",
            "controls.json": values["controls.json"],
            "takedown-ledger.jsonl": values["takedown-ledger.jsonl"],
        }
    )
    release = {
        "schema_version": "packet-release-input.v1",
        "packet_sha256": packet["packet_sha256"],
        "files": _ledger(output, set(output)),
        "manifest_sha256": "",
    }
    release["manifest_sha256"] = digest(
        canonical({key: value for key, value in release.items() if key != "manifest_sha256"})
    )
    output["release-input-manifest.json"] = canonical(release) + b"\n"
    parent_fd, stage_name, stage_fd = _stage(output_parent, final_name)
    try:
        for name in sorted(output):
            _write_at(stage_fd, name, output[name])
        _finish_stage(parent_fd, stage_name, stage_fd, final_name)
    finally:
        os.close(stage_fd)
        os.close(parent_fd)
    return output_parent / final_name


def _release_input(values: dict[str, bytes]) -> dict[str, Any]:
    release = _json(
        values["release-input-manifest.json"],
        maximum=RELEASE_INPUT_LIMITS["release-input-manifest.json"],
    )
    required = {"schema_version", "packet_sha256", "files", "manifest_sha256"}
    if (
        set(release) != required
        or release["schema_version"] != "packet-release-input.v1"
        or not _hash(release["packet_sha256"])
    ):
        raise HumanResearchError("release input schema")
    _self_digest(release, "manifest_sha256")
    _validate_ledger(
        release["files"], values, set(RELEASE_INPUT_LIMITS) - {"release-input-manifest.json"}
    )
    request = _json(
        values["packet-release-request.json"],
        maximum=RELEASE_INPUT_LIMITS["packet-release-request.json"],
    )
    if (
        set(request) != {"schema_version", "packet_sha256", "evidence_mode", "request_sha256"}
        or request["schema_version"] != "packet-release-request.v1"
        or request["evidence_mode"] != "real_human"
        or request["packet_sha256"] != release["packet_sha256"]
    ):
        raise HumanResearchError("release request")
    _self_digest(request, "request_sha256")
    packet = _validate_packet(values["packet-manifest.json"], values)
    if packet["packet_sha256"] != release["packet_sha256"]:
        raise HumanResearchError("release packet binding")
    return packet


def _bind_receipt(receipt: dict[str, Any], packet: dict[str, Any]) -> None:
    evidence = receipt["evidence"]
    expected = {
        "protocol": packet["protocol_sha256"],
        "policy_registry": packet["policy_registry_sha256"],
        "states": packet["states_sha256"],
        "split_plan": packet["split_plan_sha256"],
        "contributors": packet["contributors_sha256"],
        "annotations": packet["annotations_sha256"],
        "adjudications": packet["adjudications_sha256"],
        "controls": packet["controls_sha256"],
        "takedown_ledger": packet["takedown_ledger_sha256"],
        "packet": packet["packet_sha256"],
    }
    if receipt["scope"] != "packet_training" or evidence != expected:
        raise HumanResearchError("receipt evidence binding")


def _released_descriptor(
    kind: str,
    packet: dict[str, Any],
    receipt_raw: bytes,
    values: dict[str, bytes],
    names: tuple[str, ...],
) -> bytes:
    payload = {name: values[name] for name in names if name != "capsule-descriptor.json"}
    return _released_descriptor_from_ledger(
        kind, packet, receipt_raw, _ledger(payload, set(payload))
    )


def _released_descriptor_from_ledger(
    kind: str,
    packet: dict[str, Any],
    receipt_raw: bytes,
    files: list[dict[str, Any]],
) -> bytes:
    """Use the single released-descriptor schema for both creation and proof.

    The train-head stage deliberately has no train/dev view bytes.  It can
    nevertheless re-create their descriptor because the signed packet commits
    their byte ledgers and the receipt is carried verbatim by the downstream
    capsule.  Keeping this helper below ``_released_descriptor`` makes the
    canonical shape impossible to drift between the writer and that proof.
    """

    descriptor: dict[str, Any] = {
        "schema_version": "human-capsule.v1",
        "capsule_kind": kind,
        "packet_sha256": packet["packet_sha256"],
        "packet_release_receipt_sha256": digest(receipt_raw),
        "files": files,
        "descriptor_sha256": "",
    }
    descriptor["descriptor_sha256"] = digest(
        canonical({key: value for key, value in descriptor.items() if key != "descriptor_sha256"})
    )
    return canonical(descriptor) + b"\n"


def reconstruct_train_dev_descriptor(
    packet: dict[str, Any], packet_raw: bytes, receipt_raw: bytes
) -> bytes:
    """Rebuild exactly the original released train/dev descriptor, without views.

    ``packet_raw`` and ``receipt_raw`` are the exact downstream bytes.  The two
    unavailable view entries are reconstructed only from the signed packet
    ledger, never from paths, discovery, or a cache.
    """

    files = packet.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != {"train", "dev", "calibration", "blind_test"}
        or any(
            not isinstance(files[split], dict)
            or set(files[split]) != {"bytes", "sha256"}
            or type(files[split]["bytes"]) is not int
            or files[split]["bytes"] <= 0
            or not isinstance(files[split]["sha256"], str)
            or not HEX64.fullmatch(files[split]["sha256"])
            for split in ("train", "dev")
        )
        or not isinstance(packet.get("packet_sha256"), str)
        or digest(
            canonical({key: value for key, value in packet.items() if key != "packet_sha256"})
        )
        != packet["packet_sha256"]
        or canonical(packet) + b"\n" != packet_raw
    ):
        raise HumanResearchError("train/dev descriptor reconstruction")
    ledger = [
        {"path": "dev.jsonl", **files["dev"]},
        {
            "path": "packet-manifest.json",
            "bytes": len(packet_raw),
            "sha256": digest(packet_raw),
        },
        {
            "path": "packet-release-receipt.json",
            "bytes": len(receipt_raw),
            "sha256": digest(receipt_raw),
        },
        {"path": "train.jsonl", **files["train"]},
    ]
    return _released_descriptor_from_ledger("train-dev", packet, receipt_raw, ledger)


def install_packet_release(
    release_input_capsule: Path, receipt: Path, output_parent: Path, release_name: str
) -> Path:
    """Verify a bundled-key receipt then atomically install three released capsules."""

    final_name = _safe_name(release_name)
    values = _read_capsule(release_input_capsule, RELEASE_INPUT_LIMITS)
    packet = _release_input(values)
    try:
        receipt_raw = read_verified_external_file(receipt, maximum=64 * 1024)
        verified = verify_release_receipt(receipt_raw)
    except (VerifiedBytesError, ResearchTrustError) as exc:
        raise HumanResearchError("release receipt") from exc
    _bind_receipt(verified, packet)
    parent_fd, stage_name, stage_fd = _stage(output_parent, final_name)
    try:
        groups = {
            "train-dev": (
                "train.jsonl",
                "dev.jsonl",
                "packet-manifest.json",
                "packet-release-receipt.json",
                "capsule-descriptor.json",
            ),
            "calibration": (
                "calibration.jsonl",
                "packet-manifest.json",
                "packet-release-receipt.json",
                "capsule-descriptor.json",
            ),
            "blind": (
                "blind-test.jsonl",
                "packet-manifest.json",
                "packet-release-receipt.json",
                "capsule-descriptor.json",
            ),
        }
        for kind, names in groups.items():
            child_fd = _mkdir_at(stage_fd, kind)
            try:
                payload = {name: values[name] for name in names if name in values}
                payload["packet-release-receipt.json"] = receipt_raw
                payload["capsule-descriptor.json"] = _released_descriptor(
                    kind, packet, receipt_raw, payload, names
                )
                for name in names:
                    _write_at(child_fd, name, payload[name])
                _fsync(child_fd)
            finally:
                os.close(child_fd)
        _finish_stage(parent_fd, stage_name, stage_fd, final_name)
    finally:
        os.close(stage_fd)
        os.close(parent_fd)
    return output_parent / final_name


def main(argv: list[str] | None = None) -> int:
    parser = _SanitizedArgumentParser(prog="python -m benchmarks.human_research")
    commands = parser.add_subparsers(dest="command", parser_class=_SanitizedArgumentParser)
    materialize = commands.add_parser("materialize-packet")
    materialize.add_argument("--intake-capsule", type=Path, required=True)
    materialize.add_argument("--output-parent", type=Path, required=True)
    materialize.add_argument("--packet-name", required=True)
    install = commands.add_parser("install-packet-release")
    install.add_argument("--release-input-capsule", type=Path, required=True)
    install.add_argument("--receipt", type=Path, required=True)
    install.add_argument("--output-parent", type=Path, required=True)
    install.add_argument("--release-name", required=True)
    extract = commands.add_parser("extract-train-dev-embeddings")
    extract.add_argument("--train-dev-capsule", type=Path, required=True)
    extract.add_argument("--encoder-snapshot", type=Path, required=True)
    extract.add_argument("--output-parent", type=Path, required=True)
    extract.add_argument("--output-name", required=True)
    train = commands.add_parser("train-head")
    train.add_argument("--embedding-capsule", type=Path, required=True)
    train.add_argument("--encoder-snapshot", type=Path, required=True)
    train.add_argument("--output-parent", type=Path, required=True)
    train.add_argument("--output-name", required=True)
    try:
        args = parser.parse_args(argv)
        if args.command == "materialize-packet":
            materialize_packet(args.intake_capsule, args.output_parent, args.packet_name)
        elif args.command == "install-packet-release":
            install_packet_release(
                args.release_input_capsule, args.receipt, args.output_parent, args.release_name
            )
        elif args.command == "extract-train-dev-embeddings":
            from benchmarks.human_training import extract_train_dev_embeddings

            extract_train_dev_embeddings(
                args.train_dev_capsule,
                args.encoder_snapshot,
                args.output_parent,
                args.output_name,
            )
        elif args.command == "train-head":
            from benchmarks.human_training import train_head

            train_head(
                args.embedding_capsule,
                args.encoder_snapshot,
                args.output_parent,
                args.output_name,
            )
        else:
            parser.print_help()
            return 2
    except Exception:
        print("GOVERNANCE_INVALID", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
