from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import urllib.request
from pathlib import Path

import pytest

from benchmarks import inspect_wheel
from benchmarks.universal_local import acquisition, registry
from benchmarks.universal_local.acquisition import verification_receipt, verify_snapshot
from benchmarks.universal_local.registry import Candidate


def _fake_candidate() -> Candidate:
    content = b'{"safe":true}\n'
    item = {
        "path": "nested/config.json",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "role": "fixture",
    }
    data = {
        "id": "fixture-local",
        "model_id": "fixture/model",
        "model_revision": "a" * 40,
        "weight_format": "safetensors",
        "source_weight_dtypes": {"F16": 1},
        "loader_family": "laya_option_marker",
        "files": [item],
        "dependency_contract": "universal-local.v1",
        "acquisition_state": "eligible",
        "acquisition_contract_digest": "0" * 64,
    }
    data["acquisition_contract_digest"] = registry.acquisition_contract_digest(data)
    return Candidate(data)


def test_v2_registry_is_pending_and_routes_offline() -> None:
    candidates = registry.validate_registry()
    assert {candidate.id for candidate in candidates} == {
        "saracura-compiled",
        "laya-multilingual",
        "von-option-marker",
        "mdeberta-nli",
        "qwen-system-one",
        "typesafe-jev",
        "diffusiongemma-openjev",
    }
    assert all(
        candidate.conformance_state == "pending"
        for candidate in candidates
        if candidate.acquisition_state == "eligible"
    )


def test_registry_rejects_duplicate_keys_and_contract_mutation() -> None:
    raw = registry.REGISTRY_PATH.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        duplicate = raw.replace(b'"schema_version":', b'"schema_version":"x","schema_version":', 1)
        registry.load_registry(duplicate)
    payload = json.loads(raw)
    payload["candidates"][1]["files"][0]["path"] = "../escape.json"
    with pytest.raises(ValueError):
        registry.load_registry(json.dumps(payload).encode())
    payload = json.loads(raw)
    payload["candidates"][1]["files"][1]["sha256"] = "0" * 64
    payload["candidates"][1]["acquisition_contract_digest"] = registry.acquisition_contract_digest(
        payload["candidates"][1]
    )
    with pytest.raises(ValueError, match="truth table"):
        registry.load_registry(json.dumps(payload).encode())


def test_registry_rejects_mutated_predecessor_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(registry.V1_PATH.read_bytes())
    payload["candidates"][1]["notes"] = "unreviewed predecessor mutation"
    mutated = tmp_path / "v1.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(registry, "V1_PATH", mutated)
    with pytest.raises(ValueError, match="predecessor bytes"):
        registry.load_registry()


def test_recursive_acquisition_promotes_nested_snapshot_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    content = b'{"safe":true}\n'

    def fake_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        assert url.startswith("https://huggingface.co/fixture/model/resolve/")
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open(
                relative.rsplit("/", 1)[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent,
            )
            try:
                os.write(fd, content)
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    monkeypatch.setattr(acquisition, "_stream", fake_stream)
    path = acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    assert path == tmp_path / candidate.id / candidate.acquisition_contract_digest
    assert verify_snapshot(candidate, path)
    assert (path / "nested" / "config.json").read_bytes() == content


def test_recursive_verification_rejects_symlink_hardlink_and_extra_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    content = b'{"safe":true}\n'

    def fake_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open("config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            os.write(fd, content)
            os.close(fd)
        finally:
            os.close(parent)

    monkeypatch.setattr(acquisition, "_stream", fake_stream)
    path = acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    assert verify_snapshot(candidate, path)

    os.symlink("config.json", path / "nested" / "alias.json")
    assert not verify_snapshot(candidate, path)
    os.unlink(path / "nested" / "alias.json")

    os.link(path / "nested" / "config.json", path / "nested" / "hardlink.json")
    assert not verify_snapshot(candidate, path)
    os.unlink(path / "nested" / "hardlink.json")

    (path / "unexpected").mkdir(mode=0o700)
    assert not verify_snapshot(candidate, path)


def test_acquisition_rejects_unsafe_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="unsafe directory"):
        acquisition.acquire(candidate.id, allow_network=True, root=unsafe)


def test_atomic_publication_never_replaces_existing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    content = b'{"safe":true}\n'

    def fake_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open("config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            os.write(fd, content)
            os.close(fd)
        finally:
            os.close(parent)

    original_publish = acquisition._rename_no_replace

    def install_winner(parent_fd: int, stage: str, final: str) -> None:
        os.mkdir(final, 0o700, dir_fd=parent_fd)
        with pytest.raises(ValueError, match="no-replace"):
            original_publish(parent_fd, stage, final)
        raise ValueError("simulated concurrent winner")

    monkeypatch.setattr(acquisition, "_stream", fake_stream)
    monkeypatch.setattr(acquisition, "_rename_no_replace", install_winner)
    with pytest.raises(ValueError, match="concurrent winner"):
        acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    final = tmp_path / candidate.id / candidate.acquisition_contract_digest
    assert final.is_dir()
    assert list(final.iterdir()) == []


def test_corrupt_download_is_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)

    def corrupt_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open("config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            os.write(fd, b"x" * expected_size)
            os.close(fd)
        finally:
            os.close(parent)

    monkeypatch.setattr(acquisition, "_stream", corrupt_stream)
    with pytest.raises(ValueError, match="exact-byte"):
        acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    final = tmp_path / candidate.id / candidate.acquisition_contract_digest
    assert not final.exists()


def test_verification_rejects_same_size_mutation_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    content = b'{"safe":true}\n'

    def fake_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open("config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            os.write(fd, content)
            os.close(fd)
        finally:
            os.close(parent)

    monkeypatch.setattr(acquisition, "_stream", fake_stream)
    final = acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    original_hash = acquisition._hash_fd
    target = final / "nested" / "config.json"

    def mutate_after_hash(fd: int) -> str:
        digest = original_hash(fd)
        target.write_bytes(b"x" * len(content))
        return digest

    monkeypatch.setattr(acquisition, "_hash_fd", mutate_after_hash)
    assert not verify_snapshot(candidate, final)


def test_descriptor_receipt_rejects_nested_tree_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    content = b'{"safe":true}\n'

    def fake_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open("config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            os.write(fd, content)
            os.close(fd)
        finally:
            os.close(parent)

    monkeypatch.setattr(acquisition, "_stream", fake_stream)
    final = acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    with verification_receipt(candidate, final) as receipt:
        assert receipt.read_verified_bytes("nested/config.json") == content
        (final / "nested" / "unexpected.json").write_bytes(b"{}")
        with pytest.raises(ValueError, match="stale"):
            receipt.assert_live()


def test_snapshot_survives_unrelated_registry_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    content = b'{"safe":true}\n'

    def fake_stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
        parent = acquisition._open_relative_directory(directory_fd, "nested")
        try:
            fd = os.open("config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            os.write(fd, content)
            os.close(fd)
        finally:
            os.close(parent)

    monkeypatch.setattr(acquisition, "_stream", fake_stream)
    final = acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    monkeypatch.setattr(acquisition, "registry_digest", lambda: "f" * 64)
    assert verify_snapshot(candidate, final)


def test_invalid_final_fails_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _fake_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    parent = tmp_path / candidate.id
    parent.mkdir(mode=0o700)
    final = parent / candidate.acquisition_contract_digest
    final.mkdir(mode=0o700)
    called = False

    def forbidden_stream(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(acquisition, "_stream", forbidden_stream)
    with pytest.raises(acquisition.SnapshotConflictError, match="move it aside"):
        acquisition.acquire(candidate.id, allow_network=True, root=tmp_path)
    assert not called


@pytest.mark.parametrize(
    "url",
    (
        "http://huggingface.co/repo/file",
        "https://example.com/repo/file",
        "https://user@huggingface.co/repo/file",
        "https://huggingface.co/repo/file#fragment",
    ),
)
def test_download_url_policy_rejects_untrusted_targets(url: str) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        acquisition._validate_url(url)


def test_download_requires_exact_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def geturl(self) -> str:
            return "https://huggingface.co/fixture/model/resolve/" + "a" * 40 + "/config.json"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    class Opener:
        def open(self, request: object, timeout: int) -> Response:
            return Response()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: Opener())
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="content length"):
            acquisition._stream(
                "https://huggingface.co/fixture/model/resolve/" + "a" * 40 + "/config.json",
                directory_fd,
                "config.json",
                3,
            )
    finally:
        os.close(directory_fd)
    assert not (tmp_path / "config.json").exists()


def test_default_environment_proof_does_not_import_optional_local_stack() -> None:
    for module in ("torch", "transformers", "tokenizers", "safetensors", "platformdirs", "psutil"):
        assert module not in __import__("sys").modules
        assert importlib.util.find_spec(module) is None or module not in __import__("sys").modules


def test_universal_local_is_checkout_only_and_excluded_from_package() -> None:
    assert not (Path(__file__).parents[1] / "src" / "saracura" / "universal_local").exists()
    assert not inspect_wheel._allowed_entry("benchmarks/universal_local/acquisition.py")
