"""Phase 4B.1 governance plus sanitized dispatch for the sealed 4B.2 stages.

Model construction, embedding extraction, and CPU-head training remain in
``benchmarks.human_training``.  This module has no calibration, blind
evaluation, runtime-calibration, network, or artifact discovery behavior.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from contextlib import suppress
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
BLIND_STATE_FILES = ("01-precommitted.json", "02-running.json", "03-sealed.json")


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


def blind_custody_key(packet_manifest_sha256: str, blind_capsule_sha256: str) -> str:
    """Bind a one-time exposure to packet and blind capsule only.

    Model, fit, timestamp, and code deliberately do not participate, so an
    operator cannot compare a second candidate against one blind release.
    """

    if not _hash(packet_manifest_sha256) or not _hash(blind_capsule_sha256):
        raise HumanResearchError("blind custody binding")
    return digest(
        canonical(
            {
                "blind_capsule_sha256": blind_capsule_sha256,
                "packet_manifest_sha256": packet_manifest_sha256,
            }
        )
    )


def blind_exposure_state(
    *,
    state: str,
    custody_key: str,
    packet_sha256: str,
    packet_manifest_bytes_sha256: str,
    packet_release_receipt_sha256: str,
    blind_capsule_sha256: str,
    training_manifest_sha256: str,
    checkpoint_sha256: str,
    fit_sha256: str,
    previous_state_sha256: str | None,
    created_at: str,
    blind_report_sha256: str | None = None,
    calibration_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Construct one immutable closed blind-exposure transition."""

    base: dict[str, Any] = {
        "schema_version": "blind-exposure-state.v1",
        "state": state,
        "custody_key": custody_key,
        "packet_sha256": packet_sha256,
        "packet_manifest_bytes_sha256": packet_manifest_bytes_sha256,
        "packet_release_receipt_sha256": packet_release_receipt_sha256,
        "blind_capsule_sha256": blind_capsule_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "fit_sha256": fit_sha256,
        "previous_state_sha256": previous_state_sha256,
        "created_at": created_at,
        "state_sha256": "",
    }
    if state == "sealed":
        if not _hash(blind_report_sha256) or not _hash(calibration_candidate_sha256):
            raise HumanResearchError("blind sealed state binding")
        base["blind_report_sha256"] = blind_report_sha256
        base["calibration_candidate_sha256"] = calibration_candidate_sha256
    elif state not in {"precommitted", "running"} or (
        blind_report_sha256 is not None or calibration_candidate_sha256 is not None
    ):
        raise HumanResearchError("blind exposure state")
    _validate_blind_state_value(base)
    base["state_sha256"] = digest(
        canonical({key: value for key, value in base.items() if key != "state_sha256"})
    )
    return base


def _validate_blind_state_value(value: dict[str, Any]) -> None:
    common = {
        "schema_version",
        "state",
        "custody_key",
        "packet_sha256",
        "packet_manifest_bytes_sha256",
        "packet_release_receipt_sha256",
        "blind_capsule_sha256",
        "training_manifest_sha256",
        "checkpoint_sha256",
        "fit_sha256",
        "previous_state_sha256",
        "created_at",
        "state_sha256",
    }
    if value.get("state") == "sealed":
        common |= {"blind_report_sha256", "calibration_candidate_sha256"}
    if (
        set(value) != common
        or value.get("schema_version") != "blind-exposure-state.v1"
        or value.get("state") not in {"precommitted", "running", "sealed"}
        or any(
            not _hash(value.get(name))
            for name in common
            - {"schema_version", "state", "created_at", "previous_state_sha256", "state_sha256"}
        )
        or (value["state"] == "precommitted" and value["previous_state_sha256"] is not None)
        or (value["state"] != "precommitted" and not _hash(value["previous_state_sha256"]))
    ):
        raise HumanResearchError("blind exposure state")
    _timestamp(value["created_at"])
    recorded = value["state_sha256"]
    if recorded not in {
        "",
        digest(canonical({key: item for key, item in value.items() if key != "state_sha256"})),
    }:
        raise HumanResearchError("blind exposure state digest")


def validate_blind_exposure_state(raw: bytes, *, expected_state: str) -> dict[str, Any]:
    value = _json(raw, maximum=64 * 1024)
    if value.get("state") != expected_state:
        raise HumanResearchError("blind exposure order")
    _validate_blind_state_value(value)
    if not _hash(value["state_sha256"]):
        raise HumanResearchError("blind exposure state digest")
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
    # Never open the requested name before publication.  A caller can replace
    # that pathname between mkdir/stat/open; a private precommit name lets the
    # descriptor identity be established before the final no-replace rename.
    precommit = f"{name}.mkdir-{os.urandom(8).hex()}"
    created_identity: tuple[int, int] | None = None
    child_fd: int | None = None
    try:
        os.mkdir(precommit, 0o700, dir_fd=parent_fd)
        created_stat = os.stat(precommit, dir_fd=parent_fd, follow_symlinks=False)
        created_identity = (created_stat.st_dev, created_stat.st_ino)
        child_fd = os.open(
            precommit,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened_stat = os.fstat(child_fd)
        if (opened_stat.st_dev, opened_stat.st_ino) != created_identity:
            raise OSError("mkdir precommit identity changed")
        os.fchmod(child_fd, 0o700)
        _rename_no_replace(parent_fd, precommit, name)
        final_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (final_stat.st_dev, final_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            raise OSError("mkdir publication identity changed")
        result = child_fd
        child_fd = None
        return result
    except BaseException as exc:
        if child_fd is not None:
            with suppress(OSError):
                os.close(child_fd)
        # Only remove the private name when its pathname still names the inode
        # opened by this invocation.  Never remove the requested name after a
        # publication identity ambiguity, and never remove a pre-existing or
        # replacement directory.
        if created_identity is not None:
            with suppress(OSError):
                current = os.stat(precommit, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.rmdir(precommit, dir_fd=parent_fd)
        if isinstance(exc, HumanResearchError):
            raise
        if isinstance(exc, OSError):
            raise HumanResearchError("immutable output conflict") from exc
        raise


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
    if sys.platform.startswith("linux"):
        # renameat2(RENAME_NOREPLACE) is the Linux kernel primitive that makes
        # publication of a sealed directory one operation.  A check-then-
        # rename fallback would reintroduce the custody overwrite race, so an
        # unavailable primitive is deliberately fail-closed.
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            if renameat2(parent_fd, stage.encode(), parent_fd, final.encode(), 1) == 0:
                return
        raise HumanResearchError("immutable output conflict")
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


def _publish_blind_precommit(registry_fd: int, key: str, precommit_raw: bytes) -> int:
    """Atomically publish a nonempty ``key.active`` while holding custody.

    The temporary sibling has no custody name.  Consequently a failed write
    cannot leave an empty/partial keyed claim that blocks a later valid begin.
    The registry lock remains held by ``registry_fd`` until seal/close.
    """

    stage_fd: int | None = None
    stage_name = f".{key}.precommit-{os.urandom(8).hex()}"
    published = False
    try:
        fcntl.flock(registry_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in (key, f"{key}.active"):
            try:
                os.stat(name, dir_fd=registry_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise HumanResearchError("blind custody already used")
        stage_fd = _mkdir_at(registry_fd, stage_name)
        _write_at(stage_fd, "01-precommitted.json", precommit_raw)
        _fsync(stage_fd)
        _fsync(registry_fd)
        _rename_no_replace(registry_fd, stage_name, f"{key}.active")
        published = True
        _fsync(registry_fd)
        result = stage_fd
        stage_fd = None
        return result
    except OSError as exc:
        raise HumanResearchError("blind custody claim") from exc
    finally:
        if stage_fd is not None:
            if not published:
                with suppress(OSError):
                    os.unlink("01-precommitted.json", dir_fd=stage_fd)
            try:
                os.close(stage_fd)
            finally:
                if not published:
                    with suppress(OSError):
                        os.rmdir(stage_name, dir_fd=registry_fd)


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


CALIBRATION_CAPSULE_LIMITS = {
    "calibration.jsonl": 16 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-receipt.json": 64 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}
FIT_CAPSULE_LIMITS = {
    "temperature-fit.json": 1 * 1024 * 1024,
    # This is a descriptor-led, immutable replay witness, not a loose input.
    # It lets a later blind precommit rederive the exact candidate from the
    # closed calibration rows before any blind record is opened.
    "calibration.jsonl": 16 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-receipt.json": 64 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}
BLIND_CAPSULE_LIMITS = {
    "blind-test.jsonl": 16 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-receipt.json": 64 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}
BLIND_RUN_LIMITS = {
    "01-precommitted.json": 64 * 1024,
    "02-running.json": 64 * 1024,
    "03-sealed.json": 64 * 1024,
    "blind-report.json": 1 * 1024 * 1024,
    "calibration-candidate.json": 1 * 1024 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}


def _trusted_view_capsule(
    path: Path, *, kind: str, view_name: str, limits: dict[str, int]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Open exactly one signed view capsule; never accept a loose view file."""

    from benchmarks.human_training import _descriptor

    values = _read_capsule(path, limits)
    descriptor = _descriptor(values, kind=kind)
    packet = validate_packet_view_subset(
        values["packet-manifest.json"],
        {view_name: values[f"{view_name.replace('_', '-')}.jsonl"]},
    )
    try:
        receipt = verify_release_receipt(values["packet-release-receipt.json"])
    except Exception as exc:
        raise HumanResearchError("packet release receipt") from exc
    _bind_receipt(receipt, packet)
    if descriptor["packet_sha256"] != packet["packet_sha256"] or descriptor[
        "packet_release_receipt_sha256"
    ] != digest(values["packet-release-receipt.json"]):
        raise HumanResearchError("capsule governance binding")
    return values, packet


def _blind_descriptor_precommit(raw: bytes) -> dict[str, Any]:
    """Validate descriptor syntax without opening any blind view bytes."""

    value = _json(raw, maximum=1 * 1024 * 1024)
    expected = {
        "schema_version",
        "capsule_kind",
        "packet_sha256",
        "packet_release_receipt_sha256",
        "files",
        "descriptor_sha256",
    }
    expected_files = {
        "blind-test.jsonl",
        "packet-manifest.json",
        "packet-release-receipt.json",
    }
    if (
        set(value) != expected
        or value["schema_version"] != "human-capsule.v1"
        or value["capsule_kind"] != "blind"
        or any(
            not _hash(value[name]) for name in ("packet_sha256", "packet_release_receipt_sha256")
        )
        or not isinstance(value["files"], list)
        or [item.get("path") for item in value["files"] if isinstance(item, dict)]
        != sorted(expected_files)
    ):
        raise HumanResearchError("blind capsule descriptor")
    for item in value["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or item["path"] not in expected_files
            or type(item["bytes"]) is not int
            or item["bytes"] <= 0
            or not _hash(item["sha256"])
        ):
            raise HumanResearchError("blind capsule descriptor")
    _self_digest(value, "descriptor_sha256")
    return value


def _temperature_fit(raw: bytes) -> dict[str, Any]:
    from saracura.calibration.models import CalibrationContext

    value = _json(raw, maximum=1 * 1024 * 1024)
    expected_context = set(CalibrationContext.model_fields)
    expected = {
        "schema_version",
        *expected_context,
        "packet_manifest_sha256",
        "packet_release_receipt_sha256",
        "calibration_view_sha256",
        "training_manifest_sha256",
        "checkpoint_sha256",
        "golden_resource_sha256",
        "algorithm_revision",
        "temperature",
        "fit_sample_count",
        "fit_independent_state_count",
        "per_label_counts",
        "metrics_before",
        "metrics_after",
        "created_at",
        "fit_sha256",
    }
    if set(value) != expected or value.get("schema_version") != "temperature-fit.v1":
        raise HumanResearchError("temperature fit schema")
    try:
        CalibrationContext.model_validate(
            {name: value[name] for name in CalibrationContext.model_fields}
        )
        temperature = value["temperature"]
        if type(temperature) not in {int, float} or not 0.5 <= float(temperature) <= 5.0:
            raise ValueError
        if value["algorithm_revision"] != "bounded-log-temperature-golden-v1":
            raise ValueError
        if value["per_label_counts"] is None or set(value["per_label_counts"]) != set(LABELS):
            raise ValueError
        if any(
            type(count) is not int or count < 10 for count in value["per_label_counts"].values()
        ):
            raise ValueError
        if type(value["fit_sample_count"]) is not int or value["fit_sample_count"] < 100:
            raise ValueError
        if (
            type(value["fit_independent_state_count"]) is not int
            or value["fit_independent_state_count"] < 100
            or value["fit_independent_state_count"] > value["fit_sample_count"]
        ):
            raise ValueError
        if any(not _hash(value[name]) for name in expected if name.endswith("sha256")):
            raise ValueError
        if any(
            set(value[name]) != {"nll", "brier", "ece_15"}
            or any(
                type(item) not in {int, float} or not __import__("math").isfinite(item) or item < 0
                for item in value[name].values()
            )
            or value[name]["ece_15"] > 1
            for name in ("metrics_before", "metrics_after")
        ):
            raise ValueError
        _timestamp(value["created_at"])
        unsigned = dict(value)
        recorded = unsigned.pop("fit_sha256")
        if recorded != digest(canonical(unsigned)):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise HumanResearchError("temperature fit schema") from exc
    return value


def _blind_report(raw: bytes) -> dict[str, Any]:
    """Validate the entire aggregate-only blind report before finalization."""

    from saracura.calibration.models import BlindMetrics

    value = _json(raw, maximum=1 * 1024 * 1024)
    expected = {
        "schema_version",
        "model_manifest_sha256",
        "packet_manifest_sha256",
        "fit_sha256",
        "blind_view_sha256",
        "exposure_run_sha256",
        "golden_resource_sha256",
        "metrics_before",
        "metrics_after",
        "created_at",
        "report_sha256",
    }
    try:
        if (
            set(value) != expected
            or value["schema_version"] != "blind-report.v1"
            or any(not _hash(value[name]) for name in expected if name.endswith("sha256"))
        ):
            raise ValueError
        BlindMetrics.model_validate(value["metrics_before"])
        BlindMetrics.model_validate(value["metrics_after"])
        _timestamp(value["created_at"])
        unsigned = dict(value)
        recorded = unsigned.pop("report_sha256")
        if recorded != digest(canonical(unsigned)):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise HumanResearchError("blind report schema") from exc
    return value


def _scores_for_view(
    backend: Any, rows: list[dict[str, Any]]
) -> tuple[list[list[float]], list[int]]:
    """Keep record-level scores in memory only, in deterministic state order."""

    from saracura.runtime.workflows import MINILM_ROUTING_QUESTION
    from saracura.serialization import (
        SERIALIZER_VERSION,
        canonical_json_bytes,
        frame_segments,
        serialize_question,
    )

    logits: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        payload = frame_segments(
            SERIALIZER_VERSION, "pt-BR", "support", canonical_json_bytes({"text": row["text"]})
        )
        encoded = backend.encode_state(payload)
        outcome = backend.score_choice(
            encoded, MINILM_ROUTING_QUESTION, serialize_question(MINILM_ROUTING_QUESTION)
        )
        logits.append([float(outcome.raw_scores[label]) for label in LABELS])
        labels.append(LABELS.index(row["candidate_label"]))
    return logits, labels


def fit_temperature_stage(
    encoder_snapshot: Path,
    training_capsule: Path,
    calibration_capsule: Path,
    output_parent: Path,
    output_name: str,
) -> Path:
    """Fit the sole bounded temperature from calibration records only."""

    from benchmarks.human_calibration import GOLDEN_RESOURCE, canonical_timestamp, fit_artifact
    from benchmarks.human_training import (
        TRAINING_LIMITS,
        _load_rows,
        _make_descriptor,
        _packet_only,
        _trusted_capsule,
    )
    from saracura.backends.minilm import MiniLMRoutingBackend
    from saracura.calibration.models import CalibrationContext
    from saracura.serialization import SERIALIZER_VERSION

    training_values, _ = _trusted_capsule(training_capsule, TRAINING_LIMITS, kind="training")
    calibration_values, packet = _trusted_view_capsule(
        calibration_capsule,
        kind="calibration",
        view_name="calibration",
        limits=CALIBRATION_CAPSULE_LIMITS,
    )
    training_packet = _packet_only(training_values["packet-manifest.json"])
    if (
        training_values["packet-manifest.json"] != calibration_values["packet-manifest.json"]
        or training_values["packet-release-receipt.json"]
        != calibration_values["packet-release-receipt.json"]
        or training_packet["packet_sha256"] != packet["packet_sha256"]
    ):
        raise HumanResearchError("fit capsule packet binding")
    rows = _load_rows(calibration_values["calibration.jsonl"], "calibration")
    backend = MiniLMRoutingBackend(
        encoder_snapshot=encoder_snapshot, training_capsule=training_capsule, device="mps"
    )
    logits, labels = _scores_for_view(backend, rows)
    profile_packet = packet["packet_sha256"]
    context = CalibrationContext(
        model_id=backend.model.id,
        model_revision=backend.model.revision,
        checkpoint_sha256=backend.model.checkpoint_sha256,
        architecture_config_sha256=backend.calibration_metadata.architecture_config_sha256,
        serializer_version=SERIALIZER_VERSION,
        tokenizer_revision=backend.calibration_metadata.tokenizer_revision,
        truncation_policy_id=backend.calibration_metadata.truncation_policy_id,
        precision=backend.calibration_metadata.precision,
        quantization=backend.calibration_metadata.quantization,
        output_transform=backend.calibration_metadata.output_transform,
        workflow_id="support-routing",
        workflow_revision="phase4a-local-minilm-routing.v1",
        question_id="department",
        dataset_id="support-routing-human-ptbr",
        dataset_revision=f"human-ptbr-v1.{profile_packet[:32]}",
        split_manifest_sha256=profile_packet,
        locale="pt-BR",
        domain="support",
        head="choice",
        cardinality_bucket="2-5",
        risk_policy=None,
    )
    golden = Path(__file__).with_name("fixtures") / GOLDEN_RESOURCE
    fit = fit_artifact(
        context,
        packet_manifest_sha256=digest(calibration_values["packet-manifest.json"]),
        packet_release_receipt_sha256=digest(calibration_values["packet-release-receipt.json"]),
        calibration_view_sha256=digest(calibration_values["calibration.jsonl"]),
        model_manifest_sha256=digest(training_values["training-manifest.json"]),
        checkpoint_sha256=digest(training_values["checkpoint.safetensors"]),
        golden_resource_sha256=digest(golden.read_bytes()),
        logits=logits,
        labels=labels,
        independent_state_count=len({row["family_id"] for row in rows}),
        created_at=canonical_timestamp(),
    )
    output = {
        "temperature-fit.json": canonical(fit) + b"\n",
        "calibration.jsonl": calibration_values["calibration.jsonl"],
        "packet-manifest.json": calibration_values["packet-manifest.json"],
        "packet-release-receipt.json": calibration_values["packet-release-receipt.json"],
    }
    output["capsule-descriptor.json"] = _make_descriptor(
        "fit", output, packet, calibration_values["packet-release-receipt.json"]
    )
    from benchmarks.human_training import _write_capsule

    return _write_capsule(output_parent, output_name, output)


def _blind_runtime_context(backend: Any, packet_sha256: str) -> dict[str, Any]:
    """Build the one frozen context which a blind run is allowed to evaluate."""

    from saracura.serialization import SERIALIZER_VERSION

    return {
        "model_id": backend.model.id,
        "model_revision": backend.model.revision,
        "checkpoint_sha256": backend.model.checkpoint_sha256,
        "architecture_config_sha256": backend.calibration_metadata.architecture_config_sha256,
        "serializer_version": SERIALIZER_VERSION,
        "tokenizer_revision": backend.calibration_metadata.tokenizer_revision,
        "truncation_policy_id": backend.calibration_metadata.truncation_policy_id,
        "precision": backend.calibration_metadata.precision,
        "quantization": backend.calibration_metadata.quantization,
        "output_transform": backend.calibration_metadata.output_transform,
        "workflow_id": "support-routing",
        "workflow_revision": "phase4a-local-minilm-routing.v1",
        "question_id": "department",
        "dataset_id": "support-routing-human-ptbr",
        "dataset_revision": f"human-ptbr-v1.{packet_sha256[:32]}",
        "split_manifest_sha256": packet_sha256,
        "locale": "pt-BR",
        "domain": "support",
        "head": "choice",
        "cardinality_bucket": "2-5",
        "risk_policy": None,
    }


@dataclass(frozen=True)
class _BlindStaticInputs:
    training_values: dict[str, bytes]
    fit_values: dict[str, bytes]
    fit: dict[str, Any]
    packet: dict[str, Any]
    blind_descriptor: bytes
    descriptor_digest: str
    blind_view_sha256: str


def _prepare_blind_static(
    training_capsule: Path, fit_capsule: Path, blind_capsule: Path
) -> _BlindStaticInputs:
    """Verify non-model custody inputs without replaying either view."""

    from benchmarks.human_training import (
        TRAINING_LIMITS,
        _descriptor,
        _packet_only,
        _trusted_capsule,
    )

    training_values, _ = _trusted_capsule(training_capsule, TRAINING_LIMITS, kind="training")
    fit_values = _read_capsule(fit_capsule, FIT_CAPSULE_LIMITS)
    fit_descriptor = _descriptor(fit_values, kind="fit")
    fit = _temperature_fit(fit_values["temperature-fit.json"])
    packet = _packet_only(training_values["packet-manifest.json"])
    try:
        fit_ledger = {item["path"]: item for item in fit_descriptor["files"]}
        calibration_ledger = packet["files"]["calibration"]
        blind_ledger = packet["files"]["blind_test"]
        calibration_count = packet["accepted_counts"]["by_split"]["calibration"]
        calibration_labels = packet["accepted_counts"]["by_split_label"]["calibration"]
        if (
            fit_values["packet-manifest.json"] != training_values["packet-manifest.json"]
            or fit_values["packet-release-receipt.json"]
            != training_values["packet-release-receipt.json"]
            or fit_descriptor["packet_sha256"] != packet["packet_sha256"]
            or fit_descriptor["packet_release_receipt_sha256"]
            != digest(training_values["packet-release-receipt.json"])
            or fit_ledger["calibration.jsonl"]
            != {
                "path": "calibration.jsonl",
                "bytes": calibration_ledger["bytes"],
                "sha256": calibration_ledger["sha256"],
            }
            or fit["packet_manifest_sha256"] != digest(training_values["packet-manifest.json"])
            or fit["packet_release_receipt_sha256"]
            != digest(training_values["packet-release-receipt.json"])
            or fit["training_manifest_sha256"] != digest(training_values["training-manifest.json"])
            or fit["checkpoint_sha256"] != digest(training_values["checkpoint.safetensors"])
            or fit["calibration_view_sha256"] != calibration_ledger["sha256"]
            or fit["fit_sample_count"] != calibration_count
            # The signed packet accepts one distinct family for every view
            # record, so this count is an independent-state count as well.
            or fit["fit_independent_state_count"] != calibration_count
            or fit["per_label_counts"] != calibration_labels
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise HumanResearchError("fit capsule packet binding") from exc
    blind_descriptor = read_verified_external_file(
        blind_capsule / "capsule-descriptor.json", maximum=1 * 1024 * 1024
    )
    blind_descriptor_value = _blind_descriptor_precommit(blind_descriptor)
    try:
        blind_descriptor_ledger = {item["path"]: item for item in blind_descriptor_value["files"]}
        if (
            blind_descriptor_value["packet_sha256"] != packet["packet_sha256"]
            or blind_descriptor_value["packet_release_receipt_sha256"]
            != digest(training_values["packet-release-receipt.json"])
            or blind_descriptor_ledger["blind-test.jsonl"]
            != {
                "path": "blind-test.jsonl",
                "bytes": blind_ledger["bytes"],
                "sha256": blind_ledger["sha256"],
            }
            or blind_descriptor_ledger["packet-manifest.json"]
            != {
                "path": "packet-manifest.json",
                "bytes": len(training_values["packet-manifest.json"]),
                "sha256": digest(training_values["packet-manifest.json"]),
            }
            or blind_descriptor_ledger["packet-release-receipt.json"]
            != {
                "path": "packet-release-receipt.json",
                "bytes": len(training_values["packet-release-receipt.json"]),
                "sha256": digest(training_values["packet-release-receipt.json"]),
            }
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise HumanResearchError("blind capsule packet binding") from exc
    blind_view_sha256 = blind_ledger["sha256"]
    return _BlindStaticInputs(
        training_values=training_values,
        fit_values=fit_values,
        fit=fit,
        packet=packet,
        blind_descriptor=blind_descriptor,
        descriptor_digest=digest(blind_descriptor),
        blind_view_sha256=blind_view_sha256,
    )


def _prepare_blind_runtime(
    encoder_snapshot: Path, training_capsule: Path, static: _BlindStaticInputs
) -> Any:
    """Construct the backend and deterministically replay only an incomplete fit."""

    from benchmarks.human_calibration import GOLDEN_RESOURCE, fit_artifact
    from benchmarks.human_training import _load_rows
    from saracura.backends.minilm import MiniLMRoutingBackend
    from saracura.calibration.models import CalibrationContext

    backend = MiniLMRoutingBackend(
        encoder_snapshot=encoder_snapshot, training_capsule=training_capsule, device="mps"
    )
    expected_context = _blind_runtime_context(backend, static.packet["packet_sha256"])
    fit = static.fit
    if (
        CalibrationContext.model_validate(
            {name: fit[name] for name in CalibrationContext.model_fields}
        ).model_dump(mode="json")
        != CalibrationContext.model_validate(expected_context).model_dump(mode="json")
        or fit["checkpoint_sha256"] != backend.model.checkpoint_sha256
        or fit["golden_resource_sha256"]
        != digest((Path(__file__).with_name("fixtures") / GOLDEN_RESOURCE).read_bytes())
    ):
        raise HumanResearchError("fit runtime context binding")
    fit_packet = validate_packet_view_subset(
        static.fit_values["packet-manifest.json"],
        {"calibration": static.fit_values["calibration.jsonl"]},
    )
    try:
        fit_receipt = verify_release_receipt(static.fit_values["packet-release-receipt.json"])
    except Exception as exc:
        raise HumanResearchError("fit capsule packet binding") from exc
    _bind_receipt(fit_receipt, fit_packet)
    if fit_packet["packet_sha256"] != static.packet["packet_sha256"]:
        raise HumanResearchError("fit capsule packet binding")
    rows = _load_rows(static.fit_values["calibration.jsonl"], "calibration")
    per_label = {label: sum(row["candidate_label"] == label for row in rows) for label in LABELS}
    if (
        fit["calibration_view_sha256"] != digest(static.fit_values["calibration.jsonl"])
        or fit["fit_sample_count"] != len(rows)
        or fit["fit_independent_state_count"] != len({row["family_id"] for row in rows})
        or fit["per_label_counts"] != per_label
        or fit_packet["accepted_counts"]["by_split"]["calibration"] != len(rows)
        or fit_packet["accepted_counts"]["by_split_label"]["calibration"] != per_label
    ):
        raise HumanResearchError("fit calibration binding")
    logits, labels = _scores_for_view(backend, rows)
    expected_fit = fit_artifact(
        CalibrationContext.model_validate(expected_context),
        packet_manifest_sha256=digest(static.training_values["packet-manifest.json"]),
        packet_release_receipt_sha256=digest(static.training_values["packet-release-receipt.json"]),
        calibration_view_sha256=digest(static.fit_values["calibration.jsonl"]),
        model_manifest_sha256=digest(static.training_values["training-manifest.json"]),
        checkpoint_sha256=digest(static.training_values["checkpoint.safetensors"]),
        golden_resource_sha256=digest(
            (Path(__file__).with_name("fixtures") / GOLDEN_RESOURCE).read_bytes()
        ),
        logits=logits,
        labels=labels,
        independent_state_count=len({row["family_id"] for row in rows}),
        created_at=fit["created_at"],
    )
    if expected_fit != fit:
        raise HumanResearchError("fit calibration binding")
    return backend


def _prepare_blind_inputs(
    encoder_snapshot: Path,
    training_capsule: Path,
    fit_capsule: Path,
    blind_capsule: Path,
) -> tuple[dict[str, bytes], dict[str, Any], Any, dict[str, Any], bytes, str]:
    """Prepare an incomplete/new blind run, including required fit replay."""

    static = _prepare_blind_static(training_capsule, fit_capsule, blind_capsule)
    backend = _prepare_blind_runtime(encoder_snapshot, training_capsule, static)
    return (
        static.training_values,
        static.fit,
        backend,
        static.packet,
        static.blind_descriptor,
        static.descriptor_digest,
    )


def _blind_state_matches(
    state: dict[str, Any],
    *,
    expected_state: str,
    key: str,
    packet: dict[str, Any],
    descriptor_digest: str,
    training_values: dict[str, bytes],
    fit: dict[str, Any],
    predecessor: str | None,
) -> None:
    if (
        state["state"] != expected_state
        or state["custody_key"] != key
        or state["packet_sha256"] != packet["packet_sha256"]
        or state["packet_manifest_bytes_sha256"] != digest(training_values["packet-manifest.json"])
        or state["packet_release_receipt_sha256"]
        != digest(training_values["packet-release-receipt.json"])
        or state["blind_capsule_sha256"] != descriptor_digest
        or state["training_manifest_sha256"] != digest(training_values["training-manifest.json"])
        or state["checkpoint_sha256"] != digest(training_values["checkpoint.safetensors"])
        or state["fit_sha256"] != fit["fit_sha256"]
        or state["previous_state_sha256"] != predecessor
    ):
        raise HumanResearchError("blind exposure binding")


def _active_blind_values(path: Path) -> dict[str, bytes]:
    """Read only one of the five spec-authorized active-run file sets."""

    allowed = (
        ("01-precommitted.json",),
        ("01-precommitted.json", "02-running.json"),
        ("01-precommitted.json", "02-running.json", "blind-report.json"),
        (
            "01-precommitted.json",
            "02-running.json",
            "blind-report.json",
            "calibration-candidate.json",
        ),
        tuple(BLIND_RUN_LIMITS),
    )
    for names in allowed:
        try:
            return _read_capsule(path, {name: BLIND_RUN_LIMITS[name] for name in names})
        except HumanResearchError:
            continue
    raise HumanResearchError("blind active state file set")


def _validate_complete_sealed_run(
    values: dict[str, bytes], *, static: _BlindStaticInputs, key: str
) -> None:
    """Validate a sealed active directory without model construction or scores.

    This path has no record-level blind evidence available to replay.  It
    therefore checks every relationship that is anchored by the trusted
    training/fit capsules and blind descriptor, and publishes only an exact
    descriptor-complete state.  A run missing any one immutable byte follows
    the incomplete replay path instead.
    """

    from benchmarks.human_training import _descriptor
    from saracura.calibration.models import CalibrationContext, ResearchCalibrationCandidate

    precommit = validate_blind_exposure_state(
        values["01-precommitted.json"], expected_state="precommitted"
    )
    running = validate_blind_exposure_state(values["02-running.json"], expected_state="running")
    sealed = validate_blind_exposure_state(values["03-sealed.json"], expected_state="sealed")
    _blind_state_matches(
        precommit,
        expected_state="precommitted",
        key=key,
        packet=static.packet,
        descriptor_digest=static.descriptor_digest,
        training_values=static.training_values,
        fit=static.fit,
        predecessor=None,
    )
    _blind_state_matches(
        running,
        expected_state="running",
        key=key,
        packet=static.packet,
        descriptor_digest=static.descriptor_digest,
        training_values=static.training_values,
        fit=static.fit,
        predecessor=precommit["state_sha256"],
    )
    _blind_state_matches(
        sealed,
        expected_state="sealed",
        key=key,
        packet=static.packet,
        descriptor_digest=static.descriptor_digest,
        training_values=static.training_values,
        fit=static.fit,
        predecessor=running["state_sha256"],
    )
    try:
        candidate = ResearchCalibrationCandidate.model_validate(
            _json(values["calibration-candidate.json"], maximum=1 * 1024 * 1024)
        )
    except Exception as exc:
        raise HumanResearchError("calibration candidate") from exc
    report = _blind_report(values["blind-report.json"])
    descriptor = _descriptor(values, kind="blind-run")
    fit_context = CalibrationContext.model_validate(
        {name: static.fit[name] for name in CalibrationContext.model_fields}
    )
    expected_count = static.packet["accepted_counts"]["by_split"]["blind_test"]
    expected_per_label = static.packet["accepted_counts"]["by_split_label"]["blind_test"]
    if (
        sealed["blind_report_sha256"] != report["report_sha256"]
        or sealed["calibration_candidate_sha256"] != candidate.candidate_sha256
        or descriptor["packet_sha256"] != static.packet["packet_sha256"]
        or descriptor["packet_release_receipt_sha256"]
        != digest(static.training_values["packet-release-receipt.json"])
        or CalibrationContext.model_validate(
            {name: getattr(candidate, name) for name in CalibrationContext.model_fields}
        ).model_dump(mode="json")
        != fit_context.model_dump(mode="json")
        or candidate.parameters.temperature != float(static.fit["temperature"])
        or candidate.fit_sample_count != static.fit["fit_sample_count"]
        or candidate.fit_independent_state_count != static.fit["fit_independent_state_count"]
        or candidate.evaluation_sample_count != expected_count
        or candidate.evaluation_independent_state_count != expected_count
        or candidate.fit_sha256 != static.fit["fit_sha256"]
        or candidate.blind_view_sha256 != static.blind_view_sha256
        or candidate.exposure_run_sha256 != running["state_sha256"]
        or candidate.blind_report_sha256 != report["report_sha256"]
        or candidate.metrics_before.model_dump(mode="json") != report["metrics_before"]
        or candidate.metrics_after.model_dump(mode="json") != report["metrics_after"]
        or report["model_manifest_sha256"]
        != digest(static.training_values["training-manifest.json"])
        or report["packet_manifest_sha256"]
        != digest(static.training_values["packet-manifest.json"])
        or report["fit_sha256"] != static.fit["fit_sha256"]
        or report["blind_view_sha256"] != static.blind_view_sha256
        or report["exposure_run_sha256"] != running["state_sha256"]
        or report["golden_resource_sha256"] != static.fit["golden_resource_sha256"]
        or report["metrics_after"]["count"] != expected_count
        or report["metrics_before"]["count"] != expected_count
        or any(
            report[name]["per_label"][label]["support"] != expected_per_label[label]
            for name in ("metrics_before", "metrics_after")
            for label in LABELS
        )
    ):
        raise HumanResearchError("blind sealed binding")


def _continue_blind_run(
    active_fd: int,
    registry_fd: int,
    initial: dict[str, bytes],
    *,
    training_values: dict[str, bytes],
    fit: dict[str, Any],
    backend: Any,
    packet: dict[str, Any],
    blind_capsule: Path,
    descriptor_digest: str,
    key: str,
) -> None:
    """Advance exactly one validated active state; existing bytes are immutable."""

    from benchmarks.human_calibration import (
        GOLDEN_RESOURCE,
        blind_report,
        calibration_candidate,
        canonical_timestamp,
    )
    from benchmarks.human_training import _load_rows, _make_descriptor
    from saracura.calibration.models import CalibrationContext, ResearchCalibrationCandidate

    precommit = validate_blind_exposure_state(
        initial["01-precommitted.json"], expected_state="precommitted"
    )
    _blind_state_matches(
        precommit,
        expected_state="precommitted",
        key=key,
        packet=packet,
        descriptor_digest=descriptor_digest,
        training_values=training_values,
        fit=fit,
        predecessor=None,
    )
    running: dict[str, Any]
    if "02-running.json" in initial:
        running = validate_blind_exposure_state(
            initial["02-running.json"], expected_state="running"
        )
        _blind_state_matches(
            running,
            expected_state="running",
            key=key,
            packet=packet,
            descriptor_digest=descriptor_digest,
            training_values=training_values,
            fit=fit,
            predecessor=precommit["state_sha256"],
        )
    else:
        running = blind_exposure_state(
            state="running",
            custody_key=key,
            packet_sha256=packet["packet_sha256"],
            packet_manifest_bytes_sha256=digest(training_values["packet-manifest.json"]),
            packet_release_receipt_sha256=digest(training_values["packet-release-receipt.json"]),
            blind_capsule_sha256=descriptor_digest,
            training_manifest_sha256=digest(training_values["training-manifest.json"]),
            checkpoint_sha256=digest(training_values["checkpoint.safetensors"]),
            fit_sha256=fit["fit_sha256"],
            previous_state_sha256=precommit["state_sha256"],
            created_at=canonical_timestamp(),
        )
        _write_at(active_fd, "02-running.json", canonical(running) + b"\n")
        _fsync(active_fd)
        _fsync(registry_fd)

    # Complete runs are handled by ``resume_blind`` before runtime preparation.
    # Reaching one here would otherwise replay blind rows and weaken that
    # boundary, so fail closed rather than silently choosing a second path.
    if "03-sealed.json" in initial:
        raise HumanResearchError("complete blind recovery path")

    # Only after a durable precommit may the exact released blind view be opened.
    blind_values, blind_packet = _trusted_view_capsule(
        blind_capsule, kind="blind", view_name="blind_test", limits=BLIND_CAPSULE_LIMITS
    )
    if (
        blind_values["packet-manifest.json"] != training_values["packet-manifest.json"]
        or blind_values["packet-release-receipt.json"]
        != training_values["packet-release-receipt.json"]
        or blind_packet["packet_sha256"] != packet["packet_sha256"]
        or digest(blind_values["capsule-descriptor.json"]) != descriptor_digest
    ):
        raise HumanResearchError("blind capsule packet binding")
    rows = _load_rows(blind_values["blind-test.jsonl"], "blind_test")
    if len(rows) < 100 or any(
        sum(row["candidate_label"] == label for row in rows) < 10 for label in LABELS
    ):
        raise HumanResearchError("blind minimum")
    blind_digest = digest(blind_values["blind-test.jsonl"])
    golden_digest = digest((Path(__file__).with_name("fixtures") / GOLDEN_RESOURCE).read_bytes())
    # A partial report is never authoritative merely because its own digest
    # was updated.  Recompute the frozen aggregate with its recorded timestamp
    # and require byte-identical canonical content before advancing.
    logits, labels = _scores_for_view(backend, rows)
    existing_report = (
        _blind_report(initial["blind-report.json"]) if "blind-report.json" in initial else None
    )
    report = blind_report(
        model_manifest_sha256=digest(training_values["training-manifest.json"]),
        packet_manifest_sha256=digest(training_values["packet-manifest.json"]),
        fit_sha256=fit["fit_sha256"],
        blind_view_sha256=blind_digest,
        exposure_run_sha256=running["state_sha256"],
        golden_resource_sha256=golden_digest,
        logits=logits,
        labels=labels,
        temperature=float(fit["temperature"]),
        created_at=(existing_report or {}).get("created_at", canonical_timestamp()),
    )
    if existing_report is not None:
        if canonical(existing_report) + b"\n" != canonical(report) + b"\n":
            raise HumanResearchError("blind report replay binding")
    else:
        _write_at(active_fd, "blind-report.json", canonical(report) + b"\n")
        _fsync(active_fd)
        _fsync(registry_fd)
    if (
        report["model_manifest_sha256"] != digest(training_values["training-manifest.json"])
        or report["packet_manifest_sha256"] != digest(training_values["packet-manifest.json"])
        or report["fit_sha256"] != fit["fit_sha256"]
        or report["blind_view_sha256"] != blind_digest
        or report["exposure_run_sha256"] != running["state_sha256"]
        or report["golden_resource_sha256"] != golden_digest
    ):
        raise HumanResearchError("blind report binding")
    context = CalibrationContext.model_validate(
        {name: fit[name] for name in CalibrationContext.model_fields}
    )
    existing_candidate: dict[str, Any] | None = None
    if "calibration-candidate.json" in initial:
        try:
            candidate = ResearchCalibrationCandidate.model_validate(
                _json(initial["calibration-candidate.json"], maximum=1 * 1024 * 1024)
            )
        except Exception as exc:
            raise HumanResearchError("calibration candidate") from exc
        existing_candidate = candidate.model_dump(mode="json")
    candidate_value = calibration_candidate(
        context,
        temperature=float(fit["temperature"]),
        fit_sample_count=fit["fit_sample_count"],
        evaluation_sample_count=len(rows),
        fit_independent_state_count=fit["fit_independent_state_count"],
        evaluation_independent_state_count=len({row["family_id"] for row in rows}),
        fit_sha256=fit["fit_sha256"],
        blind_view_sha256=blind_digest,
        exposure_run_sha256=running["state_sha256"],
        blind_report_sha256=report["report_sha256"],
        metrics_before=report["metrics_before"],
        metrics_after=report["metrics_after"],
        created_at=(existing_candidate or {}).get("created_at", canonical_timestamp()),
    )
    candidate = ResearchCalibrationCandidate.model_validate(candidate_value)
    if existing_candidate is not None:
        if canonical(existing_candidate) + b"\n" != canonical(candidate_value) + b"\n":
            raise HumanResearchError("calibration candidate replay binding")
    else:
        _write_at(active_fd, "calibration-candidate.json", canonical(candidate_value) + b"\n")
        _fsync(active_fd)
        _fsync(registry_fd)
    if (
        CalibrationContext.model_validate(
            {name: getattr(candidate, name) for name in CalibrationContext.model_fields}
        ).model_dump(mode="json")
        != context.model_dump(mode="json")
        or candidate.parameters.temperature != float(fit["temperature"])
        or candidate.fit_sample_count != fit["fit_sample_count"]
        or candidate.fit_independent_state_count != fit["fit_independent_state_count"]
        or candidate.evaluation_sample_count != len(rows)
        or candidate.evaluation_independent_state_count != len({row["family_id"] for row in rows})
        or candidate.fit_sha256 != fit["fit_sha256"]
        or candidate.blind_view_sha256 != blind_digest
        or candidate.exposure_run_sha256 != running["state_sha256"]
        or candidate.blind_report_sha256 != report["report_sha256"]
        or candidate.metrics_before.model_dump(mode="json") != report["metrics_before"]
        or candidate.metrics_after.model_dump(mode="json") != report["metrics_after"]
    ):
        raise HumanResearchError("blind candidate binding")

    sealed = blind_exposure_state(
        state="sealed",
        custody_key=key,
        packet_sha256=packet["packet_sha256"],
        packet_manifest_bytes_sha256=digest(training_values["packet-manifest.json"]),
        packet_release_receipt_sha256=digest(training_values["packet-release-receipt.json"]),
        blind_capsule_sha256=descriptor_digest,
        training_manifest_sha256=digest(training_values["training-manifest.json"]),
        checkpoint_sha256=digest(training_values["checkpoint.safetensors"]),
        fit_sha256=fit["fit_sha256"],
        previous_state_sha256=running["state_sha256"],
        created_at=canonical_timestamp(),
        blind_report_sha256=report["report_sha256"],
        calibration_candidate_sha256=candidate.candidate_sha256,
    )
    _write_at(active_fd, "03-sealed.json", canonical(sealed) + b"\n")
    descriptor_values = {
        "01-precommitted.json": canonical(precommit) + b"\n",
        "02-running.json": canonical(running) + b"\n",
        "03-sealed.json": canonical(sealed) + b"\n",
        "blind-report.json": canonical(report) + b"\n",
        "calibration-candidate.json": canonical(candidate_value) + b"\n",
    }
    _write_at(
        active_fd,
        "capsule-descriptor.json",
        _make_descriptor(
            "blind-run", descriptor_values, packet, training_values["packet-release-receipt.json"]
        ),
    )
    _fsync(active_fd)
    _fsync(registry_fd)


def begin_blind(
    encoder_snapshot: Path,
    training_capsule: Path,
    fit_capsule: Path,
    blind_capsule: Path,
    exposure_registry: Path,
) -> Path:
    """Precommit one validated fit, then consume the blind capsule once."""

    from benchmarks.human_calibration import canonical_timestamp

    training_values, fit, backend, packet, _descriptor, descriptor_digest = _prepare_blind_inputs(
        encoder_snapshot, training_capsule, fit_capsule, blind_capsule
    )
    key = blind_custody_key(digest(training_values["packet-manifest.json"]), descriptor_digest)
    registry_fd = open_verified_directory(exposure_registry)
    active_name = f"{key}.active"
    try:
        precommit = blind_exposure_state(
            state="precommitted",
            custody_key=key,
            packet_sha256=packet["packet_sha256"],
            packet_manifest_bytes_sha256=digest(training_values["packet-manifest.json"]),
            packet_release_receipt_sha256=digest(training_values["packet-release-receipt.json"]),
            blind_capsule_sha256=descriptor_digest,
            training_manifest_sha256=digest(training_values["training-manifest.json"]),
            checkpoint_sha256=digest(training_values["checkpoint.safetensors"]),
            fit_sha256=fit["fit_sha256"],
            previous_state_sha256=None,
            created_at=canonical_timestamp(),
        )
        # A keyed active name is published only after the durable precommit;
        # _continue_blind_run is the first code allowed to open blind records.
        active_fd = _publish_blind_precommit(registry_fd, key, canonical(precommit) + b"\n")
        try:
            _continue_blind_run(
                active_fd,
                registry_fd,
                {"01-precommitted.json": canonical(precommit) + b"\n"},
                training_values=training_values,
                fit=fit,
                backend=backend,
                packet=packet,
                blind_capsule=blind_capsule,
                descriptor_digest=descriptor_digest,
                key=key,
            )
            _fsync(registry_fd)
            _rename_no_replace(registry_fd, active_name, key)
            _fsync(registry_fd)
        finally:
            os.close(active_fd)
    finally:
        os.close(registry_fd)
    return exposure_registry / key


def finalize_calibration(blind_run: Path, receipt_path: Path, output: Path) -> Path:
    """Promote one sealed blind candidate only after a research-scope receipt."""

    if blind_run.name.endswith(".active") or HEX64.fullmatch(blind_run.name) is None:
        raise HumanResearchError("sealed blind run required")
    _safe_name(output.name)
    output_parent_fd = open_verified_directory(output.parent)
    try:
        _finalize_calibration_at(blind_run, receipt_path, output.name, output_parent_fd)
    finally:
        os.close(output_parent_fd)
    return output


def _finalize_calibration_at(
    blind_run: Path,
    receipt_path: Path,
    output_name: str,
    output_parent_fd: int,
) -> None:
    """Finalize against the caller-retained verified output directory descriptor."""

    from benchmarks.human_training import _descriptor
    from saracura.calibration import research_calibration_id
    from saracura.calibration.io import write_calibration_atomic_at
    from saracura.calibration.models import (
        CalibrationContext,
        ResearchCalibrationArtifact,
        ResearchCalibrationCandidate,
        TemperatureParameters,
    )

    values = _read_capsule(blind_run, BLIND_RUN_LIMITS)
    descriptor = _descriptor(values, kind="blind-run")
    precommit = validate_blind_exposure_state(
        values["01-precommitted.json"], expected_state="precommitted"
    )
    running = validate_blind_exposure_state(values["02-running.json"], expected_state="running")
    sealed = validate_blind_exposure_state(values["03-sealed.json"], expected_state="sealed")
    if (
        precommit["custody_key"] != blind_run.name
        or blind_custody_key(
            precommit["packet_manifest_bytes_sha256"], precommit["blind_capsule_sha256"]
        )
        != blind_run.name
        or running["previous_state_sha256"] != precommit["state_sha256"]
        or sealed["previous_state_sha256"] != running["state_sha256"]
        or any(
            precommit[name] != running[name] or running[name] != sealed[name]
            for name in (
                "custody_key",
                "packet_sha256",
                "packet_manifest_bytes_sha256",
                "packet_release_receipt_sha256",
                "blind_capsule_sha256",
                "training_manifest_sha256",
                "checkpoint_sha256",
                "fit_sha256",
            )
        )
    ):
        raise HumanResearchError("blind exposure chain")
    candidate_raw = values["calibration-candidate.json"]
    report_raw = values["blind-report.json"]
    try:
        candidate = ResearchCalibrationCandidate.model_validate(
            _json(candidate_raw, maximum=1 * 1024 * 1024)
        )
    except Exception as exc:
        raise HumanResearchError("calibration candidate") from exc
    report = _blind_report(report_raw)
    if (
        candidate.fit_sha256 != precommit["fit_sha256"]
        or candidate.checkpoint_sha256 != precommit["checkpoint_sha256"]
        or candidate.split_manifest_sha256 != precommit["packet_sha256"]
        or candidate.exposure_run_sha256 != running["state_sha256"]
        or report["model_manifest_sha256"] != precommit["training_manifest_sha256"]
        or report["packet_manifest_sha256"] != precommit["packet_manifest_bytes_sha256"]
        or report["fit_sha256"] != candidate.fit_sha256
        or report["blind_view_sha256"] != candidate.blind_view_sha256
        or report["exposure_run_sha256"] != candidate.exposure_run_sha256
        or candidate.blind_report_sha256 != report.get("report_sha256")
        or sealed["blind_report_sha256"] != report.get("report_sha256")
        or sealed["calibration_candidate_sha256"] != candidate.candidate_sha256
        or report.get("report_sha256") != candidate.blind_report_sha256
        or candidate.evaluation_sample_count != report["metrics_after"]["count"]
        or candidate.metrics_before.model_dump(mode="json") != report["metrics_before"]
        or candidate.metrics_after.model_dump(mode="json") != report["metrics_after"]
        or descriptor["packet_sha256"] != precommit["packet_sha256"]
        or descriptor["packet_release_receipt_sha256"] != precommit["packet_release_receipt_sha256"]
    ):
        raise HumanResearchError("blind candidate binding")
    try:
        receipt_raw = read_verified_external_file(receipt_path, maximum=64 * 1024)
        receipt = verify_release_receipt(receipt_raw)
    except Exception as exc:
        raise HumanResearchError("research calibration receipt") from exc
    evidence = receipt.get("evidence")
    expected_evidence_names = {
        "protocol",
        "policy_registry",
        "states",
        "split_plan",
        "contributors",
        "annotations",
        "adjudications",
        "controls",
        "takedown_ledger",
        "packet",
        "training_manifest",
        "checkpoint",
        "fit",
        "blind_view",
        "exposure_run",
        "blind_report",
        "calibration_candidate",
    }
    if (
        receipt.get("scope") != "research_calibration"
        or not isinstance(evidence, dict)
        or set(evidence) != expected_evidence_names
        or evidence.get("packet") != precommit["packet_sha256"]
        or evidence.get("training_manifest") != precommit["training_manifest_sha256"]
        or evidence.get("checkpoint") != precommit["checkpoint_sha256"]
        or evidence.get("fit") != candidate.fit_sha256
        or evidence.get("blind_view") != candidate.blind_view_sha256
        or evidence.get("exposure_run") != candidate.exposure_run_sha256
        or evidence.get("blind_report") != candidate.blind_report_sha256
        or evidence.get("calibration_candidate") != candidate.candidate_sha256
    ):
        raise HumanResearchError("research receipt evidence binding")
    context = CalibrationContext.model_validate(
        {name: getattr(candidate, name) for name in CalibrationContext.model_fields}
    )
    receipt_sha256 = digest(receipt_raw)
    calibration_id = research_calibration_id(
        context,
        candidate.method,
        candidate.parameters,
        candidate.fit_sha256,
        candidate.blind_view_sha256,
        candidate.exposure_run_sha256,
        candidate.blind_report_sha256,
        receipt_sha256,
    )
    artifact = ResearchCalibrationArtifact(
        **context.model_dump(mode="python"),
        schema_version=2,
        status="verified_for_research",
        calibration_id=calibration_id,
        method=candidate.method,
        parameters=TemperatureParameters.model_validate(candidate.parameters),
        fit_sample_count=candidate.fit_sample_count,
        evaluation_sample_count=candidate.evaluation_sample_count,
        fit_independent_state_count=candidate.fit_independent_state_count,
        evaluation_independent_state_count=candidate.evaluation_independent_state_count,
        minimum_independent_state_count=100,
        metrics_before=candidate.metrics_before,
        metrics_after=candidate.metrics_after,
        fit_sha256=candidate.fit_sha256,
        blind_view_sha256=candidate.blind_view_sha256,
        exposure_run_sha256=candidate.exposure_run_sha256,
        blind_report_sha256=candidate.blind_report_sha256,
        candidate_sha256=candidate.candidate_sha256,
        release_receipt=receipt,
        release_receipt_sha256=receipt_sha256,
        created_at=candidate.created_at,
    )
    # This final copy is deliberately checked through the candidate's own
    # canonical digest contract.  A future field added to either schema cannot
    # silently become an unsigned runtime parameter.
    from saracura.calibration.models import candidate_from_research_artifact

    if candidate_from_research_artifact(artifact).model_dump(mode="json") != candidate.model_dump(
        mode="json"
    ):
        raise HumanResearchError("final candidate reconstruction")
    write_calibration_atomic_at(output_parent_fd, output_name, artifact)


def resume_blind(
    encoder_snapshot: Path,
    training_capsule: Path,
    fit_capsule: Path,
    blind_capsule: Path,
    exposure_registry: Path,
    custody_key: str,
) -> Path:
    """Resume only the first precommitted candidate with the original inputs.

    Every supplied capsule and runtime binding is revalidated before opening the
    blind view.  A partial report/candidate is deterministically recomputed and
    must match byte-for-byte; a complete sealed set is validated without a
    rescore before its pending no-replace rename.
    """

    if not _hash(custody_key):
        raise HumanResearchError("blind custody binding")
    registry_fd = open_verified_directory(exposure_registry)
    try:
        fcntl.flock(registry_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            os.stat(custody_key, dir_fd=registry_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise HumanResearchError("blind custody already used")
        active_fd = os.open(
            f"{custody_key}.active",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=registry_fd,
        )
        try:
            # Read state only after the registry claim lock rules out a racing
            # begin/seal transition.
            initial = _active_blind_values(exposure_registry / f"{custody_key}.active")
            if "03-sealed.json" in initial:
                static = _prepare_blind_static(training_capsule, fit_capsule, blind_capsule)
                expected_key = blind_custody_key(
                    digest(static.training_values["packet-manifest.json"]),
                    static.descriptor_digest,
                )
                if custody_key != expected_key:
                    raise HumanResearchError("blind custody binding")
                _validate_complete_sealed_run(initial, static=static, key=custody_key)
            else:
                training_values, fit, backend, packet, _descriptor, descriptor_digest = (
                    _prepare_blind_inputs(
                        encoder_snapshot, training_capsule, fit_capsule, blind_capsule
                    )
                )
                expected_key = blind_custody_key(
                    digest(training_values["packet-manifest.json"]), descriptor_digest
                )
                if custody_key != expected_key:
                    raise HumanResearchError("blind custody binding")
                _continue_blind_run(
                    active_fd,
                    registry_fd,
                    initial,
                    training_values=training_values,
                    fit=fit,
                    backend=backend,
                    packet=packet,
                    blind_capsule=blind_capsule,
                    descriptor_digest=descriptor_digest,
                    key=custody_key,
                )
            _fsync(active_fd)
        finally:
            os.close(active_fd)
        _fsync(registry_fd)
        _rename_no_replace(registry_fd, f"{custody_key}.active", custody_key)
        _fsync(registry_fd)
    except OSError as exc:
        raise HumanResearchError("blind active run") from exc
    finally:
        os.close(registry_fd)
    return exposure_registry / custody_key


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
    fit = commands.add_parser("fit-temperature")
    fit.add_argument("--encoder-snapshot", type=Path, required=True)
    fit.add_argument("--training-capsule", type=Path, required=True)
    fit.add_argument("--calibration-capsule", type=Path, required=True)
    fit.add_argument("--output-parent", type=Path, required=True)
    fit.add_argument("--output-name", required=True)
    blind = commands.add_parser("begin-blind")
    blind.add_argument("--encoder-snapshot", type=Path, required=True)
    blind.add_argument("--training-capsule", type=Path, required=True)
    blind.add_argument("--fit-capsule", type=Path, required=True)
    blind.add_argument("--blind-capsule", type=Path, required=True)
    blind.add_argument("--exposure-registry", type=Path, required=True)
    resume = commands.add_parser("resume-blind")
    resume.add_argument("--encoder-snapshot", type=Path, required=True)
    resume.add_argument("--training-capsule", type=Path, required=True)
    resume.add_argument("--fit-capsule", type=Path, required=True)
    resume.add_argument("--blind-capsule", type=Path, required=True)
    resume.add_argument("--exposure-registry", type=Path, required=True)
    resume.add_argument("--custody-key", required=True)
    finalize = commands.add_parser("finalize-calibration")
    finalize.add_argument("--blind-run", type=Path, required=True)
    finalize.add_argument("--receipt", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
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
        elif args.command == "fit-temperature":
            fit_temperature_stage(
                args.encoder_snapshot,
                args.training_capsule,
                args.calibration_capsule,
                args.output_parent,
                args.output_name,
            )
        elif args.command == "begin-blind":
            begin_blind(
                args.encoder_snapshot,
                args.training_capsule,
                args.fit_capsule,
                args.blind_capsule,
                args.exposure_registry,
            )
        elif args.command == "resume-blind":
            resume_blind(
                args.encoder_snapshot,
                args.training_capsule,
                args.fit_capsule,
                args.blind_capsule,
                args.exposure_registry,
                args.custody_key,
            )
        elif args.command == "finalize-calibration":
            finalize_calibration(args.blind_run, args.receipt, args.output)
        else:
            parser.print_help()
            return 2
    except Exception:
        print("GOVERNANCE_INVALID", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
