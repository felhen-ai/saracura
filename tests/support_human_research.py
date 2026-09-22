"""Test-only simulated evidence and ephemeral Ed25519 helpers for Phase 4B.1."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import saracura.research_trust as research_trust
from benchmarks.first_party_packet import (
    GUIDE_PATH,
    MANIFEST_PATH,
    POLICY_PATH,
    build_plan,
    canonical,
)
from saracura.research_trust import TrustRegistry, _parse_trust_registry
from saracura.research_trust import canonical as trust_canonical


def rendered(value: dict[str, Any]) -> bytes:
    return canonical(value) + b"\n"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def redigest(value: dict[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    value[field] = sha(canonical(unsigned))
    return value


def write_private(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o600)


def make_test_registry(private_key: Any) -> TrustRegistry:
    value: dict[str, Any] = {
        "schema_version": "research-trust-keys.v1",
        "keys": [
            {
                "key_id": "maintainer_aaaaaaaaaaaaaaaa",
                "public_key_ed25519": private_key.public_key().public_bytes_raw().hex(),
                "active_from": "2026-01-01T00:00:00Z",
                "revoked_at": None,
                "scopes": ["packet_training"],
            }
        ],
        "registry_sha256": "",
    }
    value["registry_sha256"] = sha(
        trust_canonical({key: child for key, child in value.items() if key != "registry_sha256"})
    )
    return _parse_trust_registry(trust_canonical(value) + b"\n")


def inject_test_trust(
    monkeypatch: Any, private_key: Any, *, now: datetime = datetime(2026, 9, 22, 12, tzinfo=UTC)
) -> None:
    """Install an ephemeral registry and fixed clock only inside a test process."""

    monkeypatch.setattr(
        research_trust, "package_trust_registry", lambda: make_test_registry(private_key)
    )
    inject_test_clock(monkeypatch, now=now)


def inject_test_clock(
    monkeypatch: Any, *, now: datetime = datetime(2026, 9, 22, 12, tzinfo=UTC)
) -> None:
    """Fix the private production clock only inside a test process."""

    monkeypatch.setattr(research_trust, "_utc_now", lambda: now)


def signed_receipt(private_key: Any, evidence: dict[str, str]) -> bytes:
    value: dict[str, Any] = {
        "schema_version": "maintainer-release-receipt.v1",
        "scope": "packet_training",
        "key_id": "maintainer_aaaaaaaaaaaaaaaa",
        "evidence_mode": "real_human",
        "issued_at": "2026-09-22T00:00:00Z",
        "expires_at": "2026-09-23T00:00:00Z",
        "evidence": evidence,
        "signature_ed25519": "",
    }
    value["signature_ed25519"] = private_key.sign(
        trust_canonical({key: child for key, child in value.items() if key != "signature_ed25519"})
    ).hex()
    return trust_canonical(value) + b"\n"


def state(number: int, label: str) -> dict[str, Any]:
    author = "human_0000000000000001" if number % 2 else "human_0000000000000002"
    marker = "1" if author.endswith("1") else "2"
    value: dict[str, Any] = {
        "schema_version": "support-routing-base-state.v1",
        "state_id": f"state_{number:032x}",
        "family_id": f"fam_{number:032x}",
        "locale": "pt-BR",
        "candidate_label": label,
        "text": f"mensagem unica{number} alfa{number} beta{number} gama{number} delta{number}",
        "author_id": author,
        "author_attestation_sha256": marker * 64,
        "source_atom_ids": [f"atom_{number:032x}"],
        "authoring_mode": "human_original",
        "privacy_declaration": "no_personal_data",
        "rights_declaration": "approved_first_party",
        "created_at": "2026-09-21T12:00:00Z",
    }
    return redigest(value, "content_digest")


def simulated_intake(path: Path) -> dict[str, bytes]:
    """Create a simulated-only capsule; production must reject it before use."""

    path.mkdir()
    path.chmod(0o700)
    labels = [
        "billing",
        "technical_support",
        "account_access",
        "subscription_cancellation",
        "order_delivery",
    ]
    states = [state(index + 1, labels[index % len(labels)]) for index in range(500)]
    states_raw = b"".join(rendered(value) for value in states)
    protocol_digest = sha(MANIFEST_PATH.read_bytes())
    plan = build_plan(
        tuple(
            type(
                "State",
                (),
                {
                    "state_id": item["state_id"],
                    "family_id": item["family_id"],
                    "label": item["candidate_label"],
                },
            )()
            for item in states
        ),
        "a" * 64,
        protocol_digest,
        sha(states_raw),
    )
    plan_raw = rendered(plan)
    people = [(f"human_{number:016x}", f"{number:x}") for number in range(1, 6)]
    contributors: dict[str, Any] = {
        "schema_version": "support-routing-contributors.v1",
        "protocol_sha256": protocol_digest,
        "contributors": [
            {
                "contributor_id": person,
                "identity_attestation_sha256": marker * 64,
                "contribution_grant_sha256": marker * 64,
                "review_grant_sha256": marker * 64,
                "grant_scopes": [
                    "adjudication",
                    "human_original_authoring",
                    "independent_annotation",
                ],
                "verified_by": "maintainer_aaaaaaaaaaaaaaaa",
                "verified_at": "2026-09-21T12:00:00Z",
                "revoked": False,
            }
            for person, marker in people
        ],
        "registry_sha256": "",
    }
    contributors_raw = rendered(redigest(contributors, "registry_sha256"))
    assignments = {item["state_id"]: item for item in plan["assignments"]}
    guide = sha(GUIDE_PATH.read_bytes())
    annotations: list[dict[str, Any]] = []
    for index, item in enumerate(states):
        count = 1 if assignments[item["state_id"]]["split"] == "train" else 2
        choices = (
            ("human_0000000000000003", "human_0000000000000004")
            if index % 2
            else ("human_0000000000000004", "human_0000000000000005")
        )
        for reviewer in choices[:count]:
            marker = reviewer[-1]
            row: dict[str, Any] = {
                "schema_version": "support-routing-independent-annotation.v1",
                "state_id": item["state_id"],
                "family_id": item["family_id"],
                "content_digest": item["content_digest"],
                "reviewer_id": reviewer,
                "reviewer_attestation_sha256": marker * 64,
                "review_grant_sha256": marker * 64,
                "annotation_guide_sha256": guide,
                "reviewer_label": item["candidate_label"],
                "privacy_review": "approved_no_personal_data",
                "rights_review": "approved_first_party",
                "reviewed_at": "2026-09-21T12:00:00Z",
                "annotation_digest": "",
            }
            annotations.append(redigest(row, "annotation_digest"))
    annotations_raw = b"".join(
        rendered(value)
        for value in sorted(annotations, key=lambda row: (row["state_id"], row["reviewer_id"]))
    )
    adjudications_raw = b""
    genesis: dict[str, Any] = {
        "schema_version": "support-routing-takedown-event.v1",
        "event": "genesis",
        "previous_event_digest": None,
        "source_atom_id": None,
        "affected_state_ids": [],
        "affected_family_ids": [],
        "artifact_digests": [],
        "reason": "genesis",
        "occurred_at": "2026-09-21T12:00:00Z",
        "disposition": "active",
        "event_digest": "",
    }
    takedown_raw = rendered(redigest(genesis, "event_digest"))
    inputs = {
        "protocol": protocol_digest,
        "states": sha(states_raw),
        "split_plan": sha(plan_raw),
        "contributors": sha(contributors_raw),
        "annotations": sha(annotations_raw),
        "adjudications": sha(adjudications_raw),
        "takedown_ledger": sha(takedown_raw),
    }
    control_ids = (
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
    controls: dict[str, Any] = {
        "schema_version": "support-routing-artifact-controls.v1",
        "policy_registry_sha256": sha(POLICY_PATH.read_bytes()),
        "packet_input_digests": inputs,
        "controls": [
            {
                "control_id": control_id,
                "status": "completed",
                "scope": "packet",
                "method_id": "ptbr-only-no-cross-locale-families.v1"
                if control_id == "cross_locale_family_link"
                else f"simulated-{control_id}.v1",
                "evidence_sha256": "a" * 64,
                "reviewer_id": "human_0000000000000003",
                "reviewed_at": "2026-09-21T12:00:00Z",
            }
            for control_id in control_ids
        ],
        "controls_sha256": "",
    }
    controls_raw = rendered(redigest(controls, "controls_sha256"))
    files = {
        "base-states.jsonl": states_raw,
        "split-plan.json": plan_raw,
        "contributors.json": contributors_raw,
        "annotations.jsonl": annotations_raw,
        "adjudications.jsonl": adjudications_raw,
        "controls.json": controls_raw,
        "takedown-ledger.jsonl": takedown_raw,
    }
    intake: dict[str, Any] = {
        "schema_version": "simulated-human-intake.v1",
        "files": [
            {"path": name, "bytes": len(raw), "sha256": sha(raw)}
            for name, raw in sorted(files.items())
        ],
        "protocol_sha256": protocol_digest,
        "policy_registry_sha256": sha(POLICY_PATH.read_bytes()),
        "evidence_mode": "simulated",
        "manifest_sha256": "",
    }
    files["intake-manifest.json"] = rendered(redigest(intake, "manifest_sha256"))
    for name, raw in files.items():
        write_private(path / name, raw)
    return files
