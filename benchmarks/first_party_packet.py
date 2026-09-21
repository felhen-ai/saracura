"""Closed, offline Phase 3A packet and deterministic split primitives."""

# The protocol code keeps literal checks adjacent to make the contract reviewable.

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import rfc8785

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks/manifests/support-routing-protocol.v1.json"
GUIDE_PATH = ROOT / "docs/annotation-guides/support-routing.v1.md"
POLICY_PATH = ROOT / "benchmarks/manifests/training-data-source-policies.v1.json"
LABELS = (
    "billing",
    "technical_support",
    "account_access",
    "subscription_cancellation",
    "order_delivery",
)
OUTCOMES = ("ambiguous", "out_of_scope", "rejected")
SPLITS = ("train", "dev", "calibration", "blind_test")
WEIGHTS = (40, 20, 20, 20)
MAX_INPUT_BYTES, MAX_PLAN_BYTES, MAX_STATES = 16 * 1024 * 1024, 8 * 1024 * 1024, 10_000
MAX_TEXT_CODEPOINTS, MAX_ATOMS, MAX_COMPARISONS = 2_000, 8, 50_000_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
STATE_ID = re.compile(r"^state_[0-9a-f]{32}$")
FAMILY_ID = re.compile(r"^fam_[0-9a-f]{32}$")
AUTHOR_ID = re.compile(r"^human_[0-9a-f]{16}$")
ATOM_ID = re.compile(r"^atom_[0-9a-f]{32}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class PacketError(ValueError):
    """Expected validation failure; callers map this to a bounded public code."""


class PrivacyBlocked(PacketError):
    pass


class DuplicateBlocked(PacketError):
    pass


class ResourceLimited(PacketError):
    pass


class WriteConflict(PacketError):
    pass


class DigestMismatch(PacketError):
    pass


def canonical(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PacketError("canonical JSON required") from exc


def strict_json(raw: bytes) -> Any:
    def duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PacketError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(PacketError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, PacketError) as exc:
        if isinstance(exc, PacketError):
            raise
        raise PacketError("invalid JSON") from exc


def _snapshot(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PacketError("unreadable input") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PacketError("regular file required")
        if before.st_size > limit:
            raise ResourceLimited("input too large")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise PacketError("unreadable input") from exc
    finally:
        os.close(fd)
    descriptor = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_descriptor = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity = (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )
    if (
        len(data) > limit
        or before.st_size != len(data)
        or descriptor != after_descriptor
        or descriptor != path_identity
    ):
        raise ResourceLimited("input changed during read")
    return data


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    chars = [
        " " if (char.isspace() or unicodedata.category(char).startswith("Z")) else char
        for char in text
    ]
    return " ".join("".join(chars).split())


def word_trigrams(text: str) -> set[tuple[str, str, str]]:
    words: list[str] = []
    current: list[str] = []
    for char in normalize_text(text):
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return set(zip(words, words[1:], words[2:], strict=False))


def trigram_jaccard_at_least(left: str, right: str) -> bool:
    a, b = word_trigrams(left), word_trigrams(right)
    if not a or not b:
        return False
    return len(a & b) * 100 >= len(a | b) * 85


EMAIL = re.compile(
    r"(?i)(?<![\w.!#$%&'*+/=?^`{|}~-])[\w.!#$%&'*+/=?^`{|}~-]{1,64}@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
)
URL = re.compile(r"(?i)(?:https?://|www\.)\S+")
IPV4_CANDIDATE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w)")
IPV6_CANDIDATE = re.compile(r"(?<![\w:])[0-9A-Fa-f:]{2,}(?![\w:])")
CPF_CNPJ = re.compile(r"(?<!\d)(?:\d[.\-/]?){11,14}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
PHONE = re.compile(r"(?<!\d)(?:(?:\+?\d)|(?:\(\d{2}\)))[\d ()-]{7,20}\d(?!\d)")
SECRET = re.compile(
    r"(?i)(?:-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bBearer\s+\S+|"
    r"\b(?:api_key|apikey|token|secret)\s*[:=]\s*\S+)"
)
HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{2,30}\b")


def _has_ip(text: str) -> bool:
    candidates = (
        *(match.group(0) for match in IPV4_CANDIDATE.finditer(text)),
        *(match.group(0) for match in IPV6_CANDIDATE.finditer(text)),
    )
    return any(_valid_ip(candidate) for candidate in candidates)


def _valid_ip(candidate: str) -> bool:
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return True


def privacy_matches(text: str) -> tuple[str, ...]:
    matches: list[str] = []
    if EMAIL.search(text):
        matches.append("email")
    if URL.search(text):
        matches.append("url")
    if _has_ip(text):
        matches.append("ip")
    for match in CPF_CNPJ.finditer(text):
        digits = re.sub(r"[./-]", "", match.group(0))
        if len(digits) in (11, 14):
            matches.append("cpf_cnpj")
            break
    for match in CARD.finditer(text):
        if 13 <= len(re.sub(r"[ -]", "", match.group(0))) <= 19:
            matches.append("payment_card")
            break
    for match in PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 10 <= len(digits) <= 13 and (
            match.group(0).startswith(("+", "(")) or any(c in match.group(0) for c in " -")
        ):
            matches.append("phone")
            break
    if SECRET.search(text):
        matches.append("secret")
    if HANDLE.search(text):
        matches.append("handle")
    return tuple(matches)


BASE_FIELDS = {
    "schema_version",
    "state_id",
    "family_id",
    "locale",
    "candidate_label",
    "text",
    "author_id",
    "author_attestation_sha256",
    "source_atom_ids",
    "authoring_mode",
    "privacy_declaration",
    "rights_declaration",
    "created_at",
    "content_digest",
}


@dataclass(frozen=True)
class State:
    value: dict[str, Any]

    @property
    def family_id(self) -> str:
        return cast(str, self.value["family_id"])

    @property
    def state_id(self) -> str:
        return cast(str, self.value["state_id"])

    @property
    def label(self) -> str:
        return cast(str, self.value["candidate_label"])

    @property
    def text(self) -> str:
        return cast(str, self.value["text"])


def validate_state(value: Any) -> State:
    if not isinstance(value, dict) or set(value) != BASE_FIELDS:
        raise PacketError("closed state object required")
    if value["schema_version"] != "support-routing-base-state.v1" or value["locale"] != "pt-BR":
        raise PacketError("invalid state literals")
    if not isinstance(value["state_id"], str) or not STATE_ID.fullmatch(value["state_id"]):
        raise PacketError("invalid state id")
    if not isinstance(value["family_id"], str) or not FAMILY_ID.fullmatch(value["family_id"]):
        raise PacketError("invalid family id")
    if not isinstance(value["candidate_label"], str) or value["candidate_label"] not in LABELS:
        raise PacketError("invalid label")
    text = value["text"]
    if (
        not isinstance(text, str)
        or text != unicodedata.normalize("NFC", text)
        or not 12 <= len(text) <= MAX_TEXT_CODEPOINTS
    ):
        raise PacketError("invalid text")
    if (
        any(
            unicodedata.category(char).startswith("C") or unicodedata.category(char) in {"Zl", "Zp"}
            for char in text
        )
        or "\r" in text
        or "\n" in text
    ):
        raise PacketError("invalid text characters")
    if not isinstance(value["author_id"], str) or not AUTHOR_ID.fullmatch(value["author_id"]):
        raise PacketError("invalid author")
    if not isinstance(value["author_attestation_sha256"], str) or not HEX64.fullmatch(
        value["author_attestation_sha256"]
    ):
        raise PacketError("invalid attestation")
    atoms = value["source_atom_ids"]
    if (
        not isinstance(atoms, list)
        or not 1 <= len(atoms) <= MAX_ATOMS
        or any(not isinstance(atom, str) for atom in atoms)
        or atoms != sorted(atoms)
        or len(set(atoms)) != len(atoms)
        or any(not ATOM_ID.fullmatch(atom) for atom in atoms)
    ):
        raise PacketError("invalid atoms")
    if (
        value["authoring_mode"] != "human_original"
        or value["privacy_declaration"] != "no_personal_data"
        or value["rights_declaration"] != "approved_first_party"
    ):
        raise PacketError("invalid declarations")
    if not isinstance(value["created_at"], str) or not TIMESTAMP.fullmatch(value["created_at"]):
        raise PacketError("invalid timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise PacketError("invalid timestamp") from exc
    if parsed_timestamp.tzinfo != UTC:
        raise PacketError("invalid timestamp")
    digest = value["content_digest"]
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise PacketError("invalid digest")
    without_digest = dict(value)
    del without_digest["content_digest"]
    if hashlib.sha256(canonical(without_digest)).hexdigest() != digest:
        raise DigestMismatch("content digest mismatch")
    return State(value)


def parse_states_bytes(raw: bytes) -> tuple[State, ...]:
    if not raw or raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise PacketError("invalid JSONL framing")
    lines = raw.splitlines(keepends=True)
    if len(lines) > MAX_STATES:
        raise ResourceLimited("too many states")
    states: list[State] = []
    for line in lines:
        if not line.endswith(b"\n") or line == b"\n":
            raise PacketError("invalid JSONL line")
        payload = line[:-1]
        value = strict_json(payload)
        if canonical(value) != payload:
            raise PacketError("noncanonical state")
        states.append(validate_state(value))
    if [state.state_id for state in states] != sorted(state.state_id for state in states):
        raise PacketError("states not sorted")
    if len({state.state_id for state in states}) != len(states) or len(
        {state.family_id for state in states}
    ) != len(states):
        raise DuplicateBlocked("duplicate state or family")
    atoms = [atom for state in states for atom in state.value["source_atom_ids"]]
    if len(set(atoms)) != len(atoms):
        raise DuplicateBlocked("atom reused")
    raw_texts = [state.text for state in states]
    normalized = [normalize_text(text) for text in raw_texts]
    if len(set(raw_texts)) != len(raw_texts) or len(set(normalized)) != len(normalized):
        raise DuplicateBlocked("duplicate text")
    comparisons = len(states) * (len(states) - 1) // 2
    if comparisons > MAX_COMPARISONS:
        raise ResourceLimited("too many comparisons")
    for index, left in enumerate(states):
        if privacy_matches(left.text):
            raise PrivacyBlocked("privacy pattern")
        for right in states[index + 1 :]:
            if trigram_jaccard_at_least(left.text, right.text):
                raise DuplicateBlocked("near duplicate")
    return tuple(states)


def read_states(path: Path) -> tuple[tuple[State, ...], bytes]:
    raw = _snapshot(path, MAX_INPUT_BYTES)
    try:
        return parse_states_bytes(raw), raw
    except UnicodeDecodeError as exc:
        raise PacketError("invalid UTF-8") from exc


def hamilton(total: int) -> tuple[int, int, int, int]:
    if total < 0:
        raise ValueError("negative total")
    quotas = [total * weight for weight in WEIGHTS]
    # Largest remainders, fixed split order on equal remainder.
    remainder_order = sorted(range(4), key=lambda i: (-(quotas[i] % 100), i))
    result = [value // 100 for value in quotas]
    for index in remainder_order[: total - sum(result)]:
        result[index] += 1
    return tuple(result)  # type: ignore[return-value]


def rank_for(seed: str, family_id: str) -> str:
    if not HEX64.fullmatch(seed):
        raise PacketError("invalid seed")
    return hashlib.sha256(
        f"support-routing-stratified-sha256.v1\0{seed}\0{family_id}".encode()
    ).hexdigest()


def build_plan(
    states: Iterable[State], seed: str, manifest_sha256: str, source_sha256: str
) -> dict[str, Any]:
    ordered = sorted(states, key=lambda state: (rank_for(seed, state.family_id), state.family_id))
    assignments: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for label in LABELS:
        subset = sorted(
            (state for state in ordered if state.label == label),
            key=lambda state: (rank_for(seed, state.family_id), state.family_id),
        )
        counts = hamilton(len(subset))
        cursor = 0
        for split, count in zip(SPLITS, counts, strict=True):
            for state in subset[cursor : cursor + count]:
                assignments.append(
                    {
                        "family_id": state.family_id,
                        "state_id": state.state_id,
                        "locale": "pt-BR",
                        "candidate_label": label,
                        "split": split,
                        "rank_sha256": rank_for(seed, state.family_id),
                    }
                )
            cursor += count
        strata.append(
            {
                "locale": "pt-BR",
                "candidate_label": label,
                "total": len(subset),
                **dict(zip(SPLITS, counts, strict=True)),
            }
        )
    assignments.sort(key=lambda item: item["family_id"])
    root = {
        "schema_version": "support-routing-split-plan.v1",
        "algorithm_id": "support-routing-stratified-sha256.v1",
        "seed": seed,
        "protocol_manifest_sha256": manifest_sha256,
        "source_file_sha256": source_sha256,
        "assignments": assignments,
        "strata": strata,
    }
    root["plan_digest"] = hashlib.sha256(canonical(root)).hexdigest()
    return root


def write_plan(plan: dict[str, Any], output: Path) -> None:
    payload = canonical(plan) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise WriteConflict("output exists") from exc
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def parse_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _snapshot(path, MAX_PLAN_BYTES)
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise PacketError("invalid plan framing")
    value = strict_json(raw[:-1])
    if canonical(value) + b"\n" != raw:
        raise PacketError("noncanonical plan")
    if not isinstance(value, dict):
        raise PacketError("plan object required")
    return value, raw


def validate_plan(
    states: tuple[State, ...],
    plan: dict[str, Any],
    seed: str,
    manifest_sha256: str,
    source_sha256: str,
) -> None:
    if not isinstance(plan, dict) or set(plan) != {
        "schema_version",
        "algorithm_id",
        "seed",
        "protocol_manifest_sha256",
        "source_file_sha256",
        "assignments",
        "strata",
        "plan_digest",
    }:
        raise PacketError("invalid split plan shape")
    if (
        plan["schema_version"] != "support-routing-split-plan.v1"
        or plan["algorithm_id"] != "support-routing-stratified-sha256.v1"
    ):
        raise PacketError("invalid split plan literals")
    if not isinstance(plan["seed"], str) or not HEX64.fullmatch(plan["seed"]):
        raise PacketError("invalid split plan seed")
    if not isinstance(plan["protocol_manifest_sha256"], str) or not HEX64.fullmatch(
        plan["protocol_manifest_sha256"]
    ):
        raise PacketError("invalid protocol digest")
    if not isinstance(plan["source_file_sha256"], str) or not HEX64.fullmatch(
        plan["source_file_sha256"]
    ):
        raise PacketError("invalid source digest")
    if not isinstance(plan["plan_digest"], str) or not HEX64.fullmatch(plan["plan_digest"]):
        raise DigestMismatch("invalid plan digest")
    unsigned = dict(plan)
    del unsigned["plan_digest"]
    if hashlib.sha256(canonical(unsigned)).hexdigest() != plan["plan_digest"]:
        raise DigestMismatch("plan digest mismatch")
    expected = build_plan(states, seed, manifest_sha256, source_sha256)
    if canonical(plan) != canonical(expected):
        raise PacketError("invalid split plan")
