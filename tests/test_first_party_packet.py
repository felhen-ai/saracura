"""Focused offline tests for the Phase 3A simulated packet gate."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

import benchmarks.first_party_gate as first_party_gate
import benchmarks.first_party_packet as packet
import benchmarks.inspect_wheel as inspect_wheel
from benchmarks.first_party_gate import validate_protocol_bytes
from benchmarks.first_party_packet import (
    DigestMismatch,
    DuplicateBlocked,
    PacketError,
    PrivacyBlocked,
    ResourceLimited,
    WriteConflict,
    build_plan,
    canonical,
    hamilton,
    normalize_text,
    parse_plan,
    parse_states_bytes,
    privacy_matches,
    read_states,
    strict_json,
    validate_plan,
    word_trigrams,
    write_plan,
)
from benchmarks.validate_manifests import validate_routed_manifest


def simulated_state_value(
    number: int = 1,
    *,
    text: str | None = None,
    label: str = "billing",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "support-routing-base-state.v1",
        "state_id": f"state_{number:032x}",
        "family_id": f"fam_{number:032x}",
        "locale": "pt-BR",
        "candidate_label": label,
        "text": text or f"Cenário simulado alfa{number} beta{number} gama{number} delta{number}.",
        "author_id": f"human_{number:016x}",
        "author_attestation_sha256": f"{number % 16:x}" * 64,
        "source_atom_ids": [f"atom_{number:032x}"],
        "authoring_mode": "human_original",
        "privacy_declaration": "no_personal_data",
        "rights_declaration": "approved_first_party",
        "created_at": "2026-09-21T12:00:00Z",
    }
    value["content_digest"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def redigest(value: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(value)
    candidate.pop("content_digest", None)
    candidate["content_digest"] = hashlib.sha256(canonical(candidate)).hexdigest()
    return candidate


def states_bytes(*values: dict[str, Any]) -> bytes:
    ordered = sorted(values, key=lambda item: item["state_id"])
    return b"".join(canonical(value) + b"\n" for value in ordered)


def write_states(path: Path, *values: dict[str, Any]) -> bytes:
    raw = states_bytes(*values)
    path.write_bytes(raw)
    return raw


def protocol_payload() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(packet.MANIFEST_PATH.read_text(encoding="utf-8")))


def test_protocol_is_closed_bound_and_non_authorizing() -> None:
    value, raw, digest = first_party_gate._protocol()
    assert value["training_authorized"] is False
    assert value["quality_claims_allowed"] is False
    assert value["publication_authorized"] is False
    assert digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("workflow_id", "other"),
        ("workflow_revision", "latest"),
        ("locale", "en"),
        ("base_state_schema", "other.v1"),
        ("split_algorithm", "random.v1"),
        ("normalization", "none"),
        ("unicode_behavior", "runtime-dependent"),
        ("near_duplicate", {}),
        ("limits", {}),
        ("training_authorized", True),
    ],
)
def test_protocol_rejects_changed_literal(field: str, replacement: object) -> None:
    value = protocol_payload()
    value[field] = replacement
    with pytest.raises(PacketError):
        validate_protocol_bytes(json.dumps(value).encode())


def test_protocol_rejects_bool_as_numeric() -> None:
    value = protocol_payload()
    value["limits"]["max_states"] = True
    with pytest.raises(PacketError):
        validate_protocol_bytes(json.dumps(value).encode())

    value = protocol_payload()
    value["split_proportions"][0]["weight"] = True
    with pytest.raises(PacketError):
        validate_protocol_bytes(json.dumps(value).encode())


def test_protocol_rejects_digest_tampering_and_duplicate_keys() -> None:
    value = protocol_payload()
    value["guide_sha256"] = "0" * 64
    with pytest.raises(PacketError):
        validate_protocol_bytes(json.dumps(value).encode())
    with pytest.raises(PacketError):
        strict_json(b'{"schema_version":"a","schema_version":"b"}')


def test_manifest_router_validates_supplied_protocol_bytes(tmp_path: Path) -> None:
    value = protocol_payload()
    value["locale"] = "en"
    path = tmp_path / "support-routing-protocol.v1.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PacketError):
        validate_routed_manifest(path)


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (0, (0, 0, 0, 0)),
        (1, (1, 0, 0, 0)),
        (2, (1, 1, 0, 0)),
        (3, (1, 1, 1, 0)),
        (4, (1, 1, 1, 1)),
        (5, (2, 1, 1, 1)),
        (6, (3, 1, 1, 1)),
        (9, (3, 2, 2, 2)),
        (10, (4, 2, 2, 2)),
    ],
)
def test_hamilton_golden_vectors(total: int, expected: tuple[int, int, int, int]) -> None:
    assert hamilton(total) == expected
    assert sum(expected) == total


def test_unicode_normalization_and_trigram_golden_vectors() -> None:
    assert normalize_text("  COBRANÇA\u00a0ÁGIL  ") == "cobrança ágil"
    assert normalize_text("A\tB\u2003C") == "a b c"
    assert word_trigrams("Árvore muito alta agora") == {
        ("árvore", "muito", "alta"),
        ("muito", "alta", "agora"),
    }


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("contato teste@example.invalid", "email"),
        ("acesse https://example.invalid/a", "url"),
        ("acesse www.example.invalid", "url"),
        ("servidor 192.0.2.1.", "ip"),
        ("rede 2001:db8::1", "ip"),
        ("rede 2001:db8::", "ip"),
        ("rede fe80::", "ip"),
        ("rede ::", "ip"),
        ("documento 123.456.789-09", "cpf_cnpj"),
        ("empresa 12.345.678/0001-90", "cpf_cnpj"),
        ("cartão 4111 1111 1111 1111", "payment_card"),
        ("ligue (11)99999999", "phone"),
        ("ligue +55 11 99999-9999", "phone"),
        ("-----BEGIN PRIVATE KEY-----", "secret"),
        ("-----BEGIN RSA PRIVATE KEY-----", "secret"),
        ("Bearer simulated-token", "secret"),
        ("api_key=simulated", "secret"),
        ("fale com @simulado", "handle"),
    ],
)
def test_privacy_positive_vectors(text: str, category: str) -> None:
    assert category in privacy_matches(text)


@pytest.mark.parametrize(
    "text",
    [
        "texto sem identificadores pessoais",
        "versão 1.2.3 e porta 8080",
        "pedido fictício número 1234",
        "arroba isolada @ não é handle",
        "palavra token sem valor atribuído",
    ],
)
def test_privacy_negative_vectors(text: str) -> None:
    assert privacy_matches(text) == ()


def test_malformed_ipv6_candidate_is_bounded() -> None:
    started = time.perf_counter()
    assert privacy_matches("texto " + ":" * 25 + "g fim") == ()
    assert time.perf_counter() - started < 0.25


def test_valid_state_and_closed_canonical_jsonl() -> None:
    raw = states_bytes(simulated_state_value())
    assert len(parse_states_bytes(raw)) == 1
    with pytest.raises(PacketError):
        parse_states_bytes(b"")
    with pytest.raises(PacketError):
        parse_states_bytes(raw.rstrip(b"\n"))
    with pytest.raises(PacketError):
        parse_states_bytes(b"\xef\xbb\xbf" + raw)
    with pytest.raises(PacketError):
        parse_states_bytes(raw.replace(b"\n", b"\r\n"))
    with pytest.raises(PacketError):
        parse_states_bytes(json.dumps(simulated_state_value(), indent=2).encode() + b"\n")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("locale", "en"),
        ("candidate_label", "other"),
        ("authoring_mode", "model_generated_or_assisted"),
        ("privacy_declaration", "unknown"),
        ("rights_declaration", "unknown"),
        ("created_at", "2026-99-99T99:99:99Z"),
        ("created_at", "2026-09-21T12:00:60Z"),
    ],
)
def test_state_rejects_invalid_fields(field: str, replacement: object) -> None:
    value = simulated_state_value()
    value[field] = replacement
    with pytest.raises(PacketError):
        parse_states_bytes(states_bytes(redigest(value)))


def test_state_rejects_unknown_field_control_and_non_nfc() -> None:
    value = simulated_state_value()
    value["unknown"] = "x"
    with pytest.raises(PacketError):
        parse_states_bytes(states_bytes(redigest(value)))

    value = simulated_state_value(text="Cenário simulado com controle\u0001 inválido.")
    with pytest.raises(PacketError):
        parse_states_bytes(states_bytes(value))

    value = simulated_state_value(text="Cena com ac\u0327a\u0303o ainda não normalizada.")
    with pytest.raises(PacketError):
        parse_states_bytes(states_bytes(value))

    for separator in ("\u2028", "\u2029"):
        value = simulated_state_value(text=f"Cena simulada{separator}com quebra proibida.")
        with pytest.raises(PacketError):
            parse_states_bytes(states_bytes(value))

    with pytest.raises(PacketError):
        parse_states_bytes(b"\xff\n")
    with pytest.raises(PacketError):
        parse_states_bytes(b'{"text":"\\ud800"}\n')


@pytest.mark.parametrize("atoms", [[{}], [1, "atom_" + "0" * 32], [], ["bad"]])
def test_state_rejects_malformed_atoms_without_type_error(atoms: list[object]) -> None:
    value = simulated_state_value()
    value["source_atom_ids"] = atoms
    with pytest.raises(PacketError):
        parse_states_bytes(states_bytes(redigest(value)))


def test_state_digest_mismatch_is_distinct() -> None:
    value = simulated_state_value()
    value["content_digest"] = "0" * 64
    with pytest.raises(DigestMismatch):
        parse_states_bytes(states_bytes(value))


def test_duplicate_ids_families_atoms_and_text_are_blocked() -> None:
    first = simulated_state_value(1)
    second = simulated_state_value(2)
    second["family_id"] = first["family_id"]
    with pytest.raises(DuplicateBlocked):
        parse_states_bytes(states_bytes(first, redigest(second)))

    second = simulated_state_value(2)
    second["source_atom_ids"] = first["source_atom_ids"]
    with pytest.raises(DuplicateBlocked):
        parse_states_bytes(states_bytes(first, redigest(second)))

    second = simulated_state_value(2, text=str(first["text"]).swapcase())
    with pytest.raises(DuplicateBlocked):
        parse_states_bytes(states_bytes(first, second))


def test_near_duplicate_and_short_text_behavior() -> None:
    prefix = (
        "produto falha durante acesso normal quando pessoa tenta concluir tarefa "
        "depois de atualizar aplicativo no dispositivo principal usando rede estável"
    )
    first = simulated_state_value(1, text=f"{prefix} cedo")
    second = simulated_state_value(2, text=f"{prefix} tarde")
    with pytest.raises(DuplicateBlocked):
        parse_states_bytes(states_bytes(first, second))

    short_one = simulated_state_value(1, text="simulação alfa")
    short_two = simulated_state_value(2, text="simulação beta")
    assert len(parse_states_bytes(states_bytes(short_one, short_two))) == 2


def test_privacy_match_blocks_state_before_split() -> None:
    value = simulated_state_value(text="Contato simulado teste@example.invalid agora.")
    with pytest.raises(PrivacyBlocked):
        parse_states_bytes(states_bytes(value))


def test_state_count_resource_bound() -> None:
    with pytest.raises(ResourceLimited):
        parse_states_bytes(b"{}\n" * 10001)


def test_safe_snapshot_rejects_symlink_fifo_device_and_oversize(tmp_path: Path) -> None:
    source = tmp_path / "states.jsonl"
    source.write_bytes(states_bytes(simulated_state_value()))
    link = tmp_path / "link.jsonl"
    link.symlink_to(source)
    with pytest.raises(PacketError):
        read_states(link)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "states.fifo"
        os.mkfifo(fifo)
        with pytest.raises(PacketError):
            read_states(fifo)

    if Path("/dev/null").exists():
        with pytest.raises(PacketError):
            read_states(Path("/dev/null"))

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"x" * 9)
    with pytest.raises(ResourceLimited):
        packet._snapshot(oversized, 8)


def test_snapshot_detects_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "states.jsonl"
    source.write_bytes(states_bytes(simulated_state_value()))
    original_read = os.read
    mutated = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        data = original_read(fd, size)
        if not mutated:
            mutated = True
            source.write_bytes(data + b" ")
        return data

    monkeypatch.setattr(os, "read", mutating_read)
    with pytest.raises(ResourceLimited):
        packet._snapshot(source, packet.MAX_INPUT_BYTES)


def test_actual_state_and_plan_size_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "states.jsonl"
    source.write_bytes(states_bytes(simulated_state_value()))
    monkeypatch.setattr(packet, "MAX_INPUT_BYTES", 8)
    with pytest.raises(ResourceLimited):
        read_states(source)

    plan = tmp_path / "plan.json"
    plan.write_bytes(b"{}\n")
    monkeypatch.setattr(packet, "MAX_PLAN_BYTES", 2)
    with pytest.raises(ResourceLimited):
        parse_plan(plan)


def test_split_plan_is_deterministic_stratified_and_complete(tmp_path: Path) -> None:
    labels = list(packet.LABELS)
    values = [simulated_state_value(i + 1, label=labels[i % len(labels)]) for i in range(25)]
    raw = write_states(tmp_path / "states.jsonl", *values)
    states = parse_states_bytes(raw)
    plan_one = build_plan(states, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())
    plan_two = build_plan(
        tuple(reversed(states)), "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest()
    )
    assert canonical(plan_one) == canonical(plan_two)
    assert len(plan_one["assignments"]) == 25
    assert [item["candidate_label"] for item in plan_one["strata"]] == labels
    assert all(item["total"] == 5 for item in plan_one["strata"])
    assert all(
        [item[split] for split in packet.SPLITS] == [2, 1, 1, 1] for item in plan_one["strata"]
    )


def test_frozen_rank_assignment_and_plan_digest_vector() -> None:
    raw = states_bytes(simulated_state_value())
    states = parse_states_bytes(raw)
    plan = build_plan(states, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())
    assignment = plan["assignments"][0]
    assert (
        assignment["rank_sha256"]
        == "c43d8349624043cce56dc64e75b3815e7a7644857de5f93b2dd093d497017b71"
    )
    assert assignment["split"] == "train"
    assert plan["plan_digest"] == "63a54fc9ffa0a84fff6aecb463d947b158201bba217ecc4f6c54c8f5e95aa38a"


def test_plan_digest_bool_tampering_and_closed_shape_fail() -> None:
    raw = states_bytes(simulated_state_value())
    states = parse_states_bytes(raw)
    plan = build_plan(states, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())
    validate_plan(states, plan, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())

    tampered = json.loads(json.dumps(plan))
    tampered["strata"][0]["total"] = True
    unsigned = dict(tampered)
    unsigned.pop("plan_digest")
    tampered["plan_digest"] = hashlib.sha256(canonical(unsigned)).hexdigest()
    with pytest.raises(PacketError):
        validate_plan(states, tampered, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())

    tampered = dict(plan)
    tampered["plan_digest"] = "0" * 64
    with pytest.raises(DigestMismatch):
        validate_plan(states, tampered, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())

    tampered = dict(plan)
    tampered["extra"] = False
    with pytest.raises(PacketError):
        validate_plan(states, tampered, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())

    for field in ("source_file_sha256", "protocol_manifest_sha256"):
        tampered = json.loads(json.dumps(plan))
        tampered[field] = "f" * 64
        unsigned = dict(tampered)
        unsigned.pop("plan_digest")
        tampered["plan_digest"] = hashlib.sha256(canonical(unsigned)).hexdigest()
        with pytest.raises(PacketError):
            validate_plan(states, tampered, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())

    tampered = json.loads(json.dumps(plan))
    tampered["assignments"][0]["rank_sha256"] = "f" * 64
    unsigned = dict(tampered)
    unsigned.pop("plan_digest")
    tampered["plan_digest"] = hashlib.sha256(canonical(unsigned)).hexdigest()
    with pytest.raises(PacketError):
        validate_plan(states, tampered, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())


def test_plan_parse_is_canonical_bounded_and_no_follow(tmp_path: Path) -> None:
    raw = states_bytes(simulated_state_value())
    states = parse_states_bytes(raw)
    plan = build_plan(states, "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())
    path = tmp_path / "plan.json"
    write_plan(plan, path)
    parsed, exact = parse_plan(path)
    assert parsed == plan
    assert exact == canonical(plan) + b"\n"

    bad = tmp_path / "bad.json"
    bad.write_bytes(json.dumps(plan, indent=2).encode() + b"\n")
    with pytest.raises(PacketError):
        parse_plan(bad)
    link = tmp_path / "plan-link.json"
    link.symlink_to(path)
    with pytest.raises(PacketError):
        parse_plan(link)


def test_atomic_no_clobber_under_concurrent_writers(tmp_path: Path) -> None:
    raw = states_bytes(simulated_state_value())
    plan = build_plan(parse_states_bytes(raw), "0" * 64, "a" * 64, hashlib.sha256(raw).hexdigest())
    output = tmp_path / "plan.json"

    def attempt() -> str:
        try:
            write_plan(plan, output)
        except WriteConflict:
            return "conflict"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _: attempt(), range(2)))
    assert outcomes == ["conflict", "written"]
    assert output.read_bytes() == canonical(plan) + b"\n"


def test_cli_success_validation_and_bounded_argument_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "states.jsonl"
    write_states(source, simulated_state_value())
    output = tmp_path / "plan.json"
    assert first_party_gate.main(["validate-protocol"]) == 0
    assert first_party_gate.main(["--help"]) == 0
    assert first_party_gate.main(["unknown", str(source)]) == 2
    captured = capsys.readouterr()
    assert "ARGUMENTS_INVALID" in captured.err
    assert str(source) not in captured.out + captured.err

    assert (
        first_party_gate.main(
            [
                "build-split-plan",
                "--states",
                str(source),
                "--seed",
                "0" * 64,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        first_party_gate.main(
            ["validate-split-plan", "--states", str(source), "--plan", str(output)]
        )
        == 0
    )


def test_cli_maps_digest_privacy_and_unexpected_without_disclosure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "states-secret-name.jsonl"
    value = simulated_state_value()
    value["content_digest"] = "0" * 64
    write_states(source, value)
    args = [
        "build-split-plan",
        "--states",
        str(source),
        "--seed",
        "0" * 64,
        "--output",
        str(tmp_path / "plan.json"),
    ]
    assert first_party_gate.main(args) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "DIGEST_MISMATCH"
    assert str(source) not in captured.err

    write_states(source, simulated_state_value(text="Contato teste@example.invalid agora."))
    assert first_party_gate.main(args) == 2
    assert capsys.readouterr().err.strip() == "PRIVACY_BLOCKED"

    monkeypatch.setattr(
        first_party_gate,
        "read_states",
        lambda _: (_ for _ in ()).throw(TypeError("private")),
    )
    assert first_party_gate.main(args) == 2
    assert capsys.readouterr().err.strip() == "UNEXPECTED_FAILURE"


def test_all_commands_make_no_socket_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    assert first_party_gate.main(["validate-protocol"]) == 0
    source = tmp_path / "states.jsonl"
    write_states(source, simulated_state_value())
    plan = tmp_path / "plan.json"
    assert (
        first_party_gate.main(
            [
                "build-split-plan",
                "--states",
                str(source),
                "--seed",
                "0" * 64,
                "--output",
                str(plan),
            ]
        )
        == 0
    )
    assert (
        first_party_gate.main(["validate-split-plan", "--states", str(source), "--plan", str(plan)])
        == 0
    )


def test_build_reuses_validated_protocol_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "states.jsonl"
    write_states(source, simulated_state_value())
    protocol_value = protocol_payload()
    monkeypatch.setattr(
        first_party_gate,
        "_protocol",
        lambda: (protocol_value, b"snapshot", "a" * 64),
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == packet.MANIFEST_PATH:
            raise AssertionError("manifest reread")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    output = tmp_path / "plan.json"
    assert (
        first_party_gate.main(
            [
                "build-split-plan",
                "--states",
                str(source),
                "--seed",
                "0" * 64,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["protocol_manifest_sha256"] == "a" * 64


def test_packaging_inspection_rejects_zero_args_and_packet_basenames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["inspect_wheel.py"])
    with pytest.raises(SystemExit):
        inspect_wheel.main()
    assert inspect_wheel._forbidden_artifact("root/SPLIT-PLAN.JSON")
    assert inspect_wheel._forbidden_artifact("root/records.JSONL")
    assert inspect_wheel._forbidden_artifact("root/MODEL.SAFETENSORS")


def _write_minimal_wheel(path: Path, *, corrupt_resource: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("saracura/__init__.py", "")
        for name, value in inspect_wheel._package_resource_bytes().items():
            archive.writestr(name, b"incorrect" if name == corrupt_resource else value)
        archive.writestr("saracura-0.dist-info/METADATA", "")
        archive.writestr("saracura-0.dist-info/WHEEL", "")
        archive.writestr("saracura-0.dist-info/RECORD", "")


def _write_minimal_sdist(path: Path, *, corrupt_resource: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, value in inspect_wheel._package_resource_bytes().items():
            payload = b"incorrect" if name == corrupt_resource else value
            info = tarfile.TarInfo(f"project/src/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        readme = b"source"
        info = tarfile.TarInfo("project/README.md")
        info.size = len(readme)
        archive.addfile(info, io.BytesIO(readme))


def test_sdist_packet_basename_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel_path = tmp_path / "simulated.whl"
    _write_minimal_wheel(wheel_path)
    archive_path = tmp_path / "simulated.tar.gz"
    payload = b"{}\n"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("project/split-plan.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(sys, "argv", ["inspect_wheel.py", str(wheel_path), str(archive_path)])
    with pytest.raises(SystemExit, match="forbidden sdist entries"):
        inspect_wheel.main()


def test_clean_minimal_artifacts_pass_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_path = tmp_path / "simulated.whl"
    _write_minimal_wheel(wheel_path)
    archive_path = tmp_path / "simulated.tar.gz"
    _write_minimal_sdist(archive_path)
    monkeypatch.setattr(sys, "argv", ["inspect_wheel.py", str(wheel_path), str(archive_path)])
    assert inspect_wheel.main() == 0


@pytest.mark.parametrize(
    ("container", "resource"),
    (
        *(("wheel", resource) for resource in sorted(inspect_wheel.PACKAGE_RESOURCES)),
        *(("sdist", resource) for resource in sorted(inspect_wheel.PACKAGE_RESOURCES)),
    ),
)
def test_package_resource_bytes_are_exact_in_wheel_and_sdist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: str, resource: str
) -> None:
    wheel_path = tmp_path / "simulated.whl"
    _write_minimal_wheel(wheel_path, corrupt_resource=resource if container == "wheel" else None)
    archive_path = tmp_path / "simulated.tar.gz"
    _write_minimal_sdist(archive_path, corrupt_resource=resource if container == "sdist" else None)
    monkeypatch.setattr(sys, "argv", ["inspect_wheel.py", str(wheel_path), str(archive_path)])
    with pytest.raises(SystemExit, match="package resource bytes"):
        inspect_wheel.main()
