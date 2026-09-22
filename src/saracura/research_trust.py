"""Closed package trust registry and Ed25519 release-receipt verification."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from typing import Any

import rfc8785

MAX_TRUST_BYTES = 64 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX128 = re.compile(r"^[0-9a-f]{128}$")
MAINTAINER_ID = re.compile(r"^maintainer_[0-9a-f]{16}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SCOPES = ("packet_training", "research_calibration")


class ResearchTrustError(ValueError):
    """A package-owned trust boundary failed closed."""


def canonical(value: Any) -> bytes:
    """Encode an already NFC-validated JSON value using RFC 8785."""

    try:
        return rfc8785.dumps(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResearchTrustError("canonical JSON required") from exc


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchTrustError("duplicate JSON key")
        result[key] = value
    return result


def _nfc(value: object) -> None:
    if isinstance(value, str):
        if value != unicodedata.normalize("NFC", value):
            raise ResearchTrustError("non-NFC JSON string")
        return
    if isinstance(value, list):
        for item in value:
            _nfc(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key != unicodedata.normalize("NFC", key):
                raise ResearchTrustError("non-NFC JSON key")
            _nfc(item)


def strict_json(raw: bytes) -> Any:
    """Parse exactly one canonical NFC RFC 8785 JSON document plus one LF."""

    if (
        not raw
        or len(raw) > MAX_TRUST_BYTES
        or raw.startswith(b"\xef\xbb\xbf")
        or b"\r" in raw
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        raise ResearchTrustError("JSON framing")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ResearchTrustError("non-finite JSON")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ResearchTrustError) as exc:
        if isinstance(exc, ResearchTrustError):
            raise
        raise ResearchTrustError("invalid JSON") from exc
    _nfc(value)
    if canonical(value) != raw[:-1]:
        raise ResearchTrustError("noncanonical JSON")
    return value


def _hash(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or TIMESTAMP.fullmatch(value) is None:
        raise ResearchTrustError("invalid RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ResearchTrustError("invalid RFC3339 timestamp") from exc
    if parsed.tzinfo != UTC or parsed.microsecond:
        raise ResearchTrustError("invalid RFC3339 timestamp")
    return parsed


@dataclass(frozen=True, slots=True)
class TrustKey:
    key_id: str
    public_key: bytes
    active_from: datetime
    revoked_at: datetime | None
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrustRegistry:
    keys: tuple[TrustKey, ...]
    registry_sha256: str


def _parse_trust_registry(raw: bytes) -> TrustRegistry:
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {"schema_version", "keys", "registry_sha256"}:
        raise ResearchTrustError("trust registry shape")
    if value["schema_version"] != "research-trust-keys.v1" or not isinstance(value["keys"], list):
        raise ResearchTrustError("trust registry schema")
    if len(value["keys"]) > 16 or not _hash(value["registry_sha256"]):
        raise ResearchTrustError("trust registry bounds")
    unsigned = dict(value)
    del unsigned["registry_sha256"]
    if hashlib.sha256(canonical(unsigned)).hexdigest() != value["registry_sha256"]:
        raise ResearchTrustError("trust registry digest")
    keys: list[TrustKey] = []
    for entry in value["keys"]:
        expected = {"key_id", "public_key_ed25519", "active_from", "revoked_at", "scopes"}
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ResearchTrustError("trust key shape")
        key_id, public_hex, scopes = (
            entry["key_id"],
            entry["public_key_ed25519"],
            entry["scopes"],
        )
        if (
            not isinstance(key_id, str)
            or MAINTAINER_ID.fullmatch(key_id) is None
            or not isinstance(public_hex, str)
            or HEX64.fullmatch(public_hex) is None
            or not isinstance(scopes, list)
            or not scopes
            or any(type(scope) is not str or scope not in SCOPES for scope in scopes)
            or scopes != sorted(set(scopes))
        ):
            raise ResearchTrustError("trust key values")
        active_from = _timestamp(entry["active_from"])
        revoked_at = None if entry["revoked_at"] is None else _timestamp(entry["revoked_at"])
        if revoked_at is not None and revoked_at <= active_from:
            raise ResearchTrustError("trust key lifecycle")
        keys.append(
            TrustKey(key_id, bytes.fromhex(public_hex), active_from, revoked_at, tuple(scopes))
        )
    if [key.key_id for key in keys] != sorted(key.key_id for key in keys):
        raise ResearchTrustError("trust keys not sorted")
    if len({key.key_id for key in keys}) != len(keys):
        raise ResearchTrustError("duplicate trust key")
    return TrustRegistry(tuple(keys), value["registry_sha256"])


def package_trust_registry_bytes() -> bytes:
    """Return the only production trust root: the package resource."""

    return resources.files("saracura").joinpath("research-trust-keys.v1.json").read_bytes()


def package_trust_registry() -> TrustRegistry:
    return _parse_trust_registry(package_trust_registry_bytes())


def _utc_now() -> datetime:
    return datetime.now(UTC)


def verify_release_receipt(raw: bytes) -> dict[str, Any]:
    """Verify an externally signed receipt against the bundled trust registry."""

    value = strict_json(raw)
    required = {
        "schema_version",
        "scope",
        "key_id",
        "evidence_mode",
        "issued_at",
        "expires_at",
        "evidence",
        "signature_ed25519",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ResearchTrustError("receipt shape")
    if (
        value["schema_version"] != "maintainer-release-receipt.v1"
        or value["evidence_mode"] != "real_human"
        or type(value["scope"]) is not str
        or value["scope"] not in SCOPES
        or not isinstance(value["key_id"], str)
        or MAINTAINER_ID.fullmatch(value["key_id"]) is None
        or not isinstance(value["signature_ed25519"], str)
        or HEX128.fullmatch(value["signature_ed25519"]) is None
    ):
        raise ResearchTrustError("receipt values")
    issued_at, expires_at, now = (
        _timestamp(value["issued_at"]),
        _timestamp(value["expires_at"]),
        _utc_now(),
    )
    if (
        expires_at <= issued_at
        or expires_at > issued_at + timedelta(days=31)
        or not issued_at <= now < expires_at
    ):
        raise ResearchTrustError("receipt lifetime")
    base_evidence = {
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
    }
    expected_evidence = (
        base_evidence
        if value["scope"] == "packet_training"
        else base_evidence
        | {
            "training_manifest",
            "checkpoint",
            "fit",
            "blind_view",
            "exposure_run",
            "blind_report",
            "calibration_candidate",
        }
    )
    evidence = value["evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence) != expected_evidence
        or any(not _hash(item) for item in evidence.values())
    ):
        raise ResearchTrustError("receipt evidence")
    key = next(
        (item for item in package_trust_registry().keys if item.key_id == value["key_id"]), None
    )
    if (
        key is None
        or value["scope"] not in key.scopes
        or issued_at < key.active_from
        or now < key.active_from
        or (
            key.revoked_at is not None
            and (
                issued_at >= key.revoked_at or now >= key.revoked_at or expires_at > key.revoked_at
            )
        )
    ):
        raise ResearchTrustError("receipt key lifecycle")
    unsigned = dict(value)
    del unsigned["signature_ed25519"]
    try:
        invalid_signature = importlib.import_module("cryptography.exceptions").InvalidSignature
        public_key_type = importlib.import_module(
            "cryptography.hazmat.primitives.asymmetric.ed25519"
        ).Ed25519PublicKey

        public_key_type.from_public_bytes(key.public_key).verify(
            bytes.fromhex(value["signature_ed25519"]), canonical(unsigned)
        )
    except ImportError as exc:
        raise ResearchTrustError("Ed25519 verification requires local-minilm") from exc
    except (invalid_signature, ValueError) as exc:
        raise ResearchTrustError("receipt signature") from exc
    return value
