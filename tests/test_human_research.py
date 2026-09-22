from __future__ import annotations

import hashlib
import inspect
import json
import os
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import benchmarks.human_research as human_research
import saracura.research_trust as research_trust
import saracura.verified_bytes as verified_bytes
from benchmarks.first_party_packet import LABELS, POLICY_PATH, SPLITS
from benchmarks.human_research import HumanResearchError, materialize_packet
from saracura.research_trust import ResearchTrustError, package_trust_registry
from tests.support_human_research import (
    inject_test_clock,
    inject_test_trust,
    redigest,
    rendered,
    signed_receipt,
    simulated_intake,
    write_private,
)


def test_bundled_research_trust_is_empty() -> None:
    assert package_trust_registry().keys == ()


def test_production_receipt_api_has_no_trust_or_clock_injection() -> None:
    assert tuple(inspect.signature(research_trust.verify_release_receipt).parameters) == ("raw",)
    assert not hasattr(research_trust, "parse_trust_registry")


def test_simulated_intake_schema_is_never_a_production_input(tmp_path: Path) -> None:
    intake = tmp_path / "intake"
    simulated_intake(intake)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o700)
    with pytest.raises(HumanResearchError):
        materialize_packet(intake, out, "packet")


def test_mkdir_at_cleans_private_precommit_after_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_open = os.open

    def fail_precommit(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if isinstance(path, str) and ".mkdir-" in path:
            raise OSError("injected open failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_precommit)
    try:
        with pytest.raises(HumanResearchError):
            human_research._mkdir_at(parent_fd, "capsule")
    finally:
        os.close(parent_fd)
    assert list(tmp_path.iterdir()) == []


def test_mkdir_at_cleans_private_precommit_after_fchmod_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_fchmod = os.fchmod

    def fail_fchmod(fd: int, mode: int) -> None:
        raise OSError("injected chmod failure")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)
    try:
        with pytest.raises(HumanResearchError):
            human_research._mkdir_at(parent_fd, "capsule")
    finally:
        os.close(parent_fd)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(os, "fchmod", original_fchmod)


def test_mkdir_at_cleans_private_precommit_after_publication_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "capsule").mkdir()
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_open = os.open
    original_close = os.close
    private_fds: set[int] = set()
    closed_fds: set[int] = set()

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if isinstance(path, str) and ".mkdir-" in path:
            private_fds.add(fd)
        return fd

    def record_close(fd: int) -> None:
        closed_fds.add(fd)
        original_close(fd)

    monkeypatch.setattr(os, "open", record_open)
    monkeypatch.setattr(os, "close", record_close)
    try:
        with pytest.raises(HumanResearchError, match="immutable output conflict"):
            human_research._mkdir_at(parent_fd, "capsule")
    finally:
        os.close(parent_fd)

    assert private_fds
    assert private_fds <= closed_fds
    assert [entry.name for entry in tmp_path.iterdir()] == ["capsule"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt("stop"), SystemExit(7)])
def test_mkdir_at_cleans_private_precommit_and_preserves_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))

    def interrupt_publication(parent_fd: int, stage: str, final: str) -> None:
        raise interruption

    monkeypatch.setattr(human_research, "_rename_no_replace", interrupt_publication)
    try:
        with pytest.raises(type(interruption)) as error:
            human_research._mkdir_at(parent_fd, "capsule")
    finally:
        os.close(parent_fd)

    assert error.value is interruption
    assert list(tmp_path.iterdir()) == []


def test_published_blind_precommit_closes_fd_when_final_registry_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    registry_fd = os.open(registry, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_mkdir_at = human_research._mkdir_at
    original_fsync = human_research._fsync
    stage_fds: list[int] = []
    fsync_count = 0

    def record_mkdir_at(parent_fd: int, name: str) -> int:
        fd = original_mkdir_at(parent_fd, name)
        stage_fds.append(fd)
        return fd

    def fail_final_registry_fsync(fd: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fsync_count == 3:
            raise HumanResearchError("injected final registry fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(human_research, "_mkdir_at", record_mkdir_at)
    monkeypatch.setattr(human_research, "_fsync", fail_final_registry_fsync)
    try:
        with pytest.raises(HumanResearchError, match="injected final registry fsync failure"):
            human_research._publish_blind_precommit(registry_fd, "a" * 64, b"precommit")
    finally:
        os.close(registry_fd)

    assert len(stage_fds) == 1
    with pytest.raises(OSError):
        os.fstat(stage_fds[0])
    assert (registry / f"{'a' * 64}.active" / "01-precommitted.json").read_bytes() == b"precommit"


def test_strict_json_requires_nfc_and_exactly_one_terminal_lf() -> None:
    with pytest.raises(ResearchTrustError):
        research_trust.strict_json(b'{"value":"x"}')
    with pytest.raises(ResearchTrustError):
        research_trust.strict_json(b'{"value":"e\\u0301"}\n')
    with pytest.raises(HumanResearchError):
        human_research._json(b'{"value":true}\n\n', maximum=1024)


def _contributor_document(**overrides: object) -> bytes:
    entry: dict[str, Any] = {
        "contributor_id": "human_aaaaaaaaaaaaaaaa",
        "identity_attestation_sha256": "a" * 64,
        "contribution_grant_sha256": "b" * 64,
        "review_grant_sha256": "c" * 64,
        "grant_scopes": ["human_original_authoring"],
        "verified_by": "maintainer_aaaaaaaaaaaaaaaa",
        "verified_at": "2026-09-22T00:00:00Z",
        "revoked": False,
    }
    entry.update(overrides)
    value: dict[str, Any] = {
        "schema_version": "support-routing-contributors.v1",
        "protocol_sha256": "d" * 64,
        "contributors": [entry],
        "registry_sha256": "",
    }
    return rendered(redigest(value, "registry_sha256"))


def test_author_binding_is_transitive_through_the_canonical_contributor_registry() -> None:
    contributors = human_research._validate_contributors(_contributor_document(), "d" * 64)
    state = {
        "author_id": "human_aaaaaaaaaaaaaaaa",
        "author_attestation_sha256": "a" * 64,
    }
    human_research._validate_author_bindings({"state_" + "1" * 32: state}, contributors)

    state["author_id"] = "human_bbbbbbbbbbbbbbbb"
    with pytest.raises(HumanResearchError, match="author contributor binding"):
        human_research._validate_author_bindings({"state_" + "1" * 32: state}, contributors)

    state["author_id"] = "human_aaaaaaaaaaaaaaaa"
    state["author_attestation_sha256"] = "e" * 64
    with pytest.raises(HumanResearchError, match="author contributor binding"):
        human_research._validate_author_bindings({"state_" + "1" * 32: state}, contributors)

    no_author_scope = human_research._validate_contributors(
        _contributor_document(grant_scopes=["independent_annotation"]), "d" * 64
    )
    with pytest.raises(HumanResearchError, match="author contributor binding"):
        human_research._validate_author_bindings(
            {
                "state_" + "1" * 32: {
                    "author_id": "human_aaaaaaaaaaaaaaaa",
                    "author_attestation_sha256": "a" * 64,
                }
            },
            no_author_scope,
        )

    with pytest.raises(HumanResearchError):
        human_research._validate_contributors(
            _contributor_document(contribution_grant_sha256="not-a-digest"),
            "d" * 64,
        )


def test_author_grant_is_committed_by_registry_digest_and_packet_receipt() -> None:
    contributors_raw = _contributor_document()
    changed_grant_raw = _contributor_document(contribution_grant_sha256="e" * 64)
    assert human_research.digest(contributors_raw) != human_research.digest(changed_grant_raw)

    packet = {
        "protocol_sha256": "1" * 64,
        "policy_registry_sha256": "2" * 64,
        "states_sha256": "3" * 64,
        "split_plan_sha256": "4" * 64,
        "contributors_sha256": human_research.digest(contributors_raw),
        "annotations_sha256": "6" * 64,
        "adjudications_sha256": "7" * 64,
        "controls_sha256": "8" * 64,
        "takedown_ledger_sha256": "9" * 64,
        "packet_sha256": "a" * 64,
    }
    receipt = {
        "scope": "packet_training",
        "evidence": {
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
        },
    }
    human_research._bind_receipt(receipt, packet)
    packet["contributors_sha256"] = human_research.digest(changed_grant_raw)
    with pytest.raises(HumanResearchError, match="receipt evidence binding"):
        human_research._bind_receipt(receipt, packet)


def test_unsigned_contributor_maintainer_reference_is_opaque_and_format_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def trust_lookup_is_forbidden() -> object:
        raise AssertionError("unsigned intake must not load the trust registry")

    monkeypatch.setattr(research_trust, "package_trust_registry", trust_lookup_is_forbidden)
    contributors = human_research._validate_contributors(
        _contributor_document(verified_by="maintainer_bbbbbbbbbbbbbbbb"), "d" * 64
    )
    assert set(contributors) == {"human_aaaaaaaaaaaaaaaa"}
    with pytest.raises(HumanResearchError, match="contributor values"):
        human_research._validate_contributors(
            _contributor_document(verified_by="not-a-maintainer"), "d" * 64
        )


def test_reference_sets_cover_only_referenced_human_contributors() -> None:
    contributors = human_research._validate_contributors(_contributor_document(), "d" * 64)
    state = {"author_id": "human_aaaaaaaaaaaaaaaa"}
    human_research._validate_reference_sets({"state_" + "1" * 32: state}, {}, {}, [], contributors)


def test_malformed_contributor_annotation_and_adjudication_types_fail_closed() -> None:
    with pytest.raises(HumanResearchError):
        human_research._validate_contributors(_contributor_document(grant_scopes={}), "d" * 64)
    state = {
        "state_id": "state_" + "1" * 32,
        "family_id": "fam_" + "1" * 32,
        "author_id": "human_aaaaaaaaaaaaaaaa",
        "candidate_label": "billing",
        "content_digest": "a" * 64,
    }
    annotation: dict[str, Any] = {
        "schema_version": "support-routing-independent-annotation.v1",
        "state_id": state["state_id"],
        "family_id": state["family_id"],
        "content_digest": state["content_digest"],
        "reviewer_id": "human_bbbbbbbbbbbbbbbb",
        "reviewer_attestation_sha256": "b" * 64,
        "review_grant_sha256": "c" * 64,
        "annotation_guide_sha256": "d" * 64,
        "reviewer_label": [],
        "privacy_review": "approved_no_personal_data",
        "rights_review": "approved_first_party",
        "reviewed_at": "2026-09-22T00:00:00Z",
        "annotation_digest": "",
    }
    with pytest.raises(HumanResearchError):
        human_research._validate_annotations(
            rendered(redigest(annotation, "annotation_digest")),
            {state["state_id"]: state},
            {},
            "d" * 64,
        )
    adjudication: dict[str, Any] = {
        "schema_version": "support-routing-adjudication.v1",
        "state_id": state["state_id"],
        "family_id": state["family_id"],
        "content_digest": state["content_digest"],
        "author_id": state["author_id"],
        "candidate_label": [],
        "adjudicator_id": "human_cccccccccccccccc",
        "adjudicator_attestation_sha256": "c" * 64,
        "adjudication_grant_sha256": "d" * 64,
        "annotation_guide_sha256": "d" * 64,
        "annotation_digests": ["e" * 64],
        "selected_label": "billing",
        "adjudicated_at": "2026-09-22T00:00:00Z",
        "adjudication_digest": "",
    }
    with pytest.raises(HumanResearchError):
        human_research._validate_adjudications(
            rendered(redigest(adjudication, "adjudication_digest")),
            {state["state_id"]: state},
            {},
            "d" * 64,
        )


def test_ephemeral_ed25519_receipt_only_verifies_under_test_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private = crypto.Ed25519PrivateKey.generate()
    evidence = {
        name: "a" * 64
        for name in (
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
        )
    }
    inject_test_trust(monkeypatch, private)
    assert (
        research_trust.verify_release_receipt(signed_receipt(private, evidence))["scope"]
        == "packet_training"
    )


def test_receipt_mutation_and_empty_production_trust_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private = crypto.Ed25519PrivateKey.generate()
    evidence = {
        name: "a" * 64
        for name in (
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
        )
    }
    receipt = signed_receipt(private, evidence)
    inject_test_clock(monkeypatch)
    with pytest.raises(ResearchTrustError):
        research_trust.verify_release_receipt(receipt)
    inject_test_trust(monkeypatch, private)
    with pytest.raises(ResearchTrustError):
        research_trust.verify_release_receipt(
            receipt.replace(b'"packet_training"', b'"research_calibration"')
        )


def test_reconciliation_never_adjudicates_privacy_or_rights_rejection() -> None:
    state = {
        "state_id": "state_" + "1" * 32,
        "family_id": "fam_" + "1" * 32,
        "author_id": "human_" + "1" * 16,
        "candidate_label": "billing",
        "text": "texto seguro para teste",
        "content_digest": "a" * 64,
    }
    review = {
        "reviewer_id": "human_" + "2" * 16,
        "reviewer_label": "billing",
        "privacy_review": "rejected",
        "rights_review": "approved_first_party",
        "annotation_digest": "b" * 64,
    }
    adjudication = {
        "adjudicator_id": "human_" + "3" * 16,
        "annotation_digests": ["b" * 64],
        "selected_label": "billing",
    }
    with pytest.raises(HumanResearchError):
        human_research._reconcile(
            {state["state_id"]: state},
            {state["state_id"]: {"split": "train"}},
            {state["state_id"]: [review]},
            {state["state_id"]: adjudication},
        )


def test_backend_limit_excludes_before_annotation_cardinality() -> None:
    state = {
        "state_id": "state_" + "1" * 32,
        "family_id": "fam_" + "1" * 32,
        "author_id": "human_" + "1" * 16,
        "candidate_label": "billing",
        "text": "x" * 321,
        "content_digest": "a" * 64,
    }
    views, excluded = human_research._reconcile(
        {state["state_id"]: state},
        {state["state_id"]: {"split": "train"}},
        {},
        {},
    )
    assert not any(views.values())
    assert excluded == {"backend_input_limit": 1}


@pytest.mark.parametrize(
    "field",
    (
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
    ),
)
def test_packet_training_receipt_binds_every_evidence_digest(field: str) -> None:
    packet = {
        "protocol_sha256": "1" * 64,
        "policy_registry_sha256": "2" * 64,
        "states_sha256": "3" * 64,
        "split_plan_sha256": "4" * 64,
        "contributors_sha256": "5" * 64,
        "annotations_sha256": "6" * 64,
        "adjudications_sha256": "7" * 64,
        "controls_sha256": "8" * 64,
        "takedown_ledger_sha256": "9" * 64,
        "packet_sha256": "a" * 64,
    }
    evidence = {
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
    receipt: dict[str, Any] = {
        "scope": "packet_training",
        "evidence": evidence,
    }
    human_research._bind_receipt(receipt, packet)
    evidence[field] = "b" * 64
    with pytest.raises(HumanResearchError):
        human_research._bind_receipt(receipt, packet)


def test_reconciliation_disagreement_only_accepts_matching_independent_adjudication() -> None:
    state = {
        "state_id": "state_" + "1" * 32,
        "family_id": "fam_" + "1" * 32,
        "author_id": "human_" + "1" * 16,
        "candidate_label": "billing",
        "text": "texto seguro para teste",
        "content_digest": "a" * 64,
    }
    reviews = [
        {
            "reviewer_id": "human_" + "2" * 16,
            "reviewer_label": "billing",
            "privacy_review": "approved_no_personal_data",
            "rights_review": "approved_first_party",
            "annotation_digest": "b" * 64,
        },
        {
            "reviewer_id": "human_" + "3" * 16,
            "reviewer_label": "technical_support",
            "privacy_review": "approved_no_personal_data",
            "rights_review": "approved_first_party",
            "annotation_digest": "c" * 64,
        },
    ]
    assignments = {state["state_id"]: {"split": "dev"}}
    views, excluded = human_research._reconcile(
        {state["state_id"]: state}, assignments, {state["state_id"]: reviews}, {}
    )
    assert views["dev"] == []
    assert excluded == {"review_disagreement_excluded": 1}
    adjudication = {
        "adjudicator_id": "human_" + "4" * 16,
        "annotation_digests": ["b" * 64, "c" * 64],
        "selected_label": "billing",
    }
    views, excluded = human_research._reconcile(
        {state["state_id"]: state},
        assignments,
        {state["state_id"]: reviews},
        {state["state_id"]: adjudication},
    )
    assert [row["state_id"] for row in views["dev"]] == [state["state_id"]]
    assert excluded == {}


def test_reconciliation_rejects_an_unnecessary_adjudication() -> None:
    state = {
        "state_id": "state_" + "1" * 32,
        "family_id": "fam_" + "1" * 32,
        "author_id": "human_" + "1" * 16,
        "candidate_label": "billing",
        "text": "texto seguro para teste",
        "content_digest": "a" * 64,
    }
    review = {
        "reviewer_id": "human_" + "2" * 16,
        "reviewer_label": "billing",
        "privacy_review": "approved_no_personal_data",
        "rights_review": "approved_first_party",
        "annotation_digest": "b" * 64,
    }
    adjudication = {
        "adjudicator_id": "human_" + "3" * 16,
        "annotation_digests": ["b" * 64],
        "selected_label": "billing",
    }
    with pytest.raises(HumanResearchError):
        human_research._reconcile(
            {state["state_id"]: state},
            {state["state_id"]: {"split": "train"}},
            {state["state_id"]: [review]},
            {state["state_id"]: adjudication},
        )


def test_verified_directory_rejects_symlink_and_unsafe_mode(tmp_path: Path) -> None:
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    capsule.chmod(0o700)
    target = tmp_path / "target.json"
    write_private(target, b"{}\n")
    (capsule / "payload.json").symlink_to(target)
    with pytest.raises(HumanResearchError):
        human_research._read_capsule(capsule, {"payload.json": 64})


@pytest.mark.parametrize("mode,hard_link", ((0o644, False), (0o600, True)))
def test_verified_directory_rejects_unsafe_mode_and_hard_link(
    tmp_path: Path, mode: int, hard_link: bool
) -> None:
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    capsule.chmod(0o700)
    payload = capsule / "payload.json"
    if hard_link:
        source = tmp_path / "source.json"
        write_private(source, b"{}\n")
        os.link(source, payload)
    else:
        payload.write_bytes(b"{}\n")
        payload.chmod(mode)
    with pytest.raises(HumanResearchError):
        human_research._read_capsule(capsule, {"payload.json": 64})


def test_verified_external_file_detects_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    write_private(receipt, b"{}\n")
    original_read = os.read
    replaced = False

    def racing_read(fd: int, maximum: int) -> bytes:
        nonlocal replaced
        raw = original_read(fd, maximum)
        if raw and not replaced:
            replacement = tmp_path / "replacement.json"
            write_private(replacement, b'{"changed":true}\n')
            os.replace(replacement, receipt)
            replaced = True
        return raw

    monkeypatch.setattr(os, "read", racing_read)
    with pytest.raises(verified_bytes.VerifiedBytesError):
        verified_bytes.read_verified_external_file(receipt, maximum=64)


def test_verified_external_file_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    """O_NONBLOCK plus fstat must reject a FIFO before any read can hang."""

    fifo = tmp_path / "receipt.fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(verified_bytes.VerifiedBytesError):
        verified_bytes.read_verified_external_file(fifo, maximum=64)


def _simulated_packet_values() -> tuple[dict[str, bytes], bytes, dict[str, Any]]:
    """Build temporary opaque simulated bytes for direct installer-validator tests only."""

    protocol, protocol_digest = human_research._known_protocol()
    genesis: dict[str, Any] = {
        "schema_version": "support-routing-takedown-event.v1",
        "event": "genesis",
        "previous_event_digest": None,
        "source_atom_id": None,
        "affected_state_ids": [],
        "affected_family_ids": [],
        "artifact_digests": [],
        "reason": "genesis",
        "occurred_at": "2026-09-22T00:00:00Z",
        "disposition": "active",
        "event_digest": "",
    }
    values = {
        "base-states.jsonl": b"simulated states\n",
        "split-plan.json": b"simulated plan\n",
        "contributors.json": b"simulated contributors\n",
        "annotations.jsonl": b"simulated annotations\n",
        "adjudications.jsonl": b"",
        "controls.json": b'{"simulated":true}\n',
        "takedown-ledger.jsonl": rendered(redigest(genesis, "event_digest")),
    }
    views: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    sizes = {"train": 200, "dev": 100, "calibration": 100, "blind_test": 100}
    index = 0
    for split in SPLITS:
        for _ in range(sizes[split]):
            label = LABELS[index % len(LABELS)]
            views[split].append(
                {
                    "state_id": f"state_{index:032x}",
                    "family_id": f"fam_{index:032x}",
                    "locale": "pt-BR",
                    "candidate_label": label,
                    "text": f"simulated opaque packet record {index}",
                    "content_digest": hashlib.sha256(f"simulated-{index}".encode()).hexdigest(),
                }
            )
            index += 1
    values.update(human_research._view_bytes(views))
    policy_digest = protocol["policy_registry_sha256"]
    assert isinstance(policy_digest, str)
    packet = human_research._packet_manifest(
        values,
        views,
        Counter(),
        protocol_digest,
        policy_digest,
    )
    return values, rendered(packet), packet


def test_install_packet_validator_rechecks_self_digest_views_and_governance_bytes() -> None:
    values, packet_raw, packet = _simulated_packet_values()
    assert (
        human_research._validate_packet(packet_raw, values)["packet_sha256"]
        == packet["packet_sha256"]
    )
    altered_packet = dict(packet)
    altered_packet["packet_sha256"] = "0" * 64
    with pytest.raises(HumanResearchError):
        human_research._validate_packet(rendered(altered_packet), values)
    altered_values = dict(values)
    altered_values["train.jsonl"] += b'{"simulated":true}\n'
    with pytest.raises(HumanResearchError):
        human_research._validate_packet(packet_raw, altered_values)
    altered_values = dict(values)
    altered_values["controls.json"] = b'{"changed":true}\n'
    with pytest.raises(HumanResearchError):
        human_research._validate_packet(packet_raw, altered_values)


def test_malformed_controls_packet_and_receipt_types_fail_closed() -> None:
    values, _, packet = _simulated_packet_values()
    altered_packet = dict(packet)
    altered_packet["files"] = []
    with pytest.raises(HumanResearchError):
        human_research._validate_packet(rendered(redigest(altered_packet, "packet_sha256")), values)
    receipt: dict[str, Any] = {
        "schema_version": "maintainer-release-receipt.v1",
        "scope": [],
        "key_id": "maintainer_aaaaaaaaaaaaaaaa",
        "evidence_mode": "real_human",
        "issued_at": "2026-09-22T00:00:00Z",
        "expires_at": "2026-09-23T00:00:00Z",
        "evidence": {},
        "signature_ed25519": "0" * 128,
    }
    with pytest.raises(ResearchTrustError):
        research_trust.verify_release_receipt(research_trust.canonical(receipt) + b"\n")


def test_takedown_requires_the_immutable_single_genesis_event() -> None:
    values, _, _ = _simulated_packet_values()
    human_research._validate_takedown(values["takedown-ledger.jsonl"])
    event = json.loads(values["takedown-ledger.jsonl"])
    event["disposition"] = "removed"
    with pytest.raises(HumanResearchError):
        human_research._validate_takedown(rendered(redigest(event, "event_digest")))


def test_controls_require_the_complete_ordered_13_control_ledger() -> None:
    reviewer_id = "human_aaaaaaaaaaaaaaaa"
    contributors = {
        reviewer_id: human_research.Contributor(
            reviewer_id,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            ("independent_annotation",),
        )
    }
    inputs = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "protocol",
            "states",
            "split_plan",
            "contributors",
            "annotations",
            "adjudications",
            "takedown_ledger",
        )
    }
    controls: dict[str, Any] = {
        "schema_version": "support-routing-artifact-controls.v1",
        "policy_registry_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        "packet_input_digests": inputs,
        "controls": [
            {
                "control_id": control_id,
                "status": "completed",
                "scope": "packet",
                "method_id": "ptbr-only-no-cross-locale-families.v1"
                if control_id == "cross_locale_family_link"
                else f"opaque-{control_id}.v1",
                "evidence_sha256": hashlib.sha256(control_id.encode()).hexdigest(),
                "reviewer_id": reviewer_id,
                "reviewed_at": "2026-09-22T00:00:00Z",
            }
            for control_id in human_research.CONTROL_IDS
        ],
        "controls_sha256": "",
    }
    raw = rendered(redigest(controls, "controls_sha256"))
    human_research._validate_controls(
        raw,
        policy_digest=controls["policy_registry_sha256"],
        inputs=inputs,
        contributors=contributors,
    )
    controls["controls"][0]["method_id"] = []
    with pytest.raises(HumanResearchError):
        human_research._validate_controls(
            rendered(redigest(controls, "controls_sha256")),
            policy_digest=controls["policy_registry_sha256"],
            inputs=inputs,
            contributors=contributors,
        )
    controls["controls"][0]["method_id"] = "opaque-artifact_manifest.v1"
    controls["controls"] = controls["controls"][:-1]
    with pytest.raises(HumanResearchError):
        human_research._validate_controls(
            rendered(redigest(controls, "controls_sha256")),
            policy_digest=controls["policy_registry_sha256"],
            inputs=inputs,
            contributors=contributors,
        )


def test_source_artifact_denylist_covers_every_phase_4b_1_artifact_family() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    exclude = set(project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])
    assert "/**/*.jsonl" in exclude
    assert "/**/*.safetensors" in exclude
    for basename in (
        "base-states.jsonl",
        "split-plan.json",
        "contributors.json",
        "annotations.jsonl",
        "adjudications.jsonl",
        "controls.json",
        "takedown-ledger.jsonl",
        "intake-manifest.json",
        "packet-manifest.json",
        "packet-release-request.json",
        "packet-release-receipt.json",
        "release-input-manifest.json",
        "capsule-descriptor.json",
    ):
        assert f"/**/{basename}" in exclude


def test_atomic_output_install_never_replaces_an_existing_child(tmp_path: Path) -> None:
    parent = tmp_path / "output"
    parent.mkdir()
    parent.chmod(0o700)
    parent_fd, stage_name, stage_fd = human_research._stage(parent, "packet")
    try:
        human_research._write_at(stage_fd, "evidence.json", b"{}\n")
        with pytest.raises(HumanResearchError):
            human_research._write_at(stage_fd, "evidence.json", b"changed\n")
        human_research._finish_stage(parent_fd, stage_name, stage_fd, "packet")
    finally:
        os.close(stage_fd)
        os.close(parent_fd)
    assert (parent / "packet").stat().st_mode & 0o777 == 0o700
    assert (parent / "packet" / "evidence.json").stat().st_mode & 0o777 == 0o600
    parent_fd, stage_name, stage_fd = human_research._stage(parent, "packet")
    try:
        human_research._write_at(stage_fd, "evidence.json", b"second\n")
        with pytest.raises(HumanResearchError):
            human_research._finish_stage(parent_fd, stage_name, stage_fd, "packet")
    finally:
        os.close(stage_fd)
        os.close(parent_fd)
