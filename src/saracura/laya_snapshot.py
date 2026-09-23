"""Closed descriptor custody for the explicit Laya local snapshot.

This module intentionally has no optional ML imports.  Consumers retain this
receipt for the lifetime of a loaded model and may only read model bytes through
its descriptors.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

_CHUNK: Final = 1024 * 1024
_TOKEN = object()


class LayaSnapshotError(ValueError):
    """The caller-provided snapshot did not meet the exact local contract."""


@dataclass(frozen=True, slots=True)
class LayaLedgerFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LayaCandidate:
    candidate: str
    model_id: str
    model_revision: str
    upstream_source_revision: str
    acquisition_contract_digest: str
    conformance_vector_digest: str
    runtime_disposition: str
    files: tuple[LayaLedgerFile, ...]

    @property
    def checkpoint_sha256(self) -> str:
        return next(item.sha256 for item in self.files if item.path == "model.safetensors")

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.files)


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LayaSnapshotError("candidate ledger has duplicate keys")
        result[key] = value
    return result


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def package_laya_ledger_bytes() -> bytes:
    return resources.files("saracura").joinpath("laya-candidate.v1.json").read_bytes()


def load_laya_candidate(raw: bytes | None = None) -> LayaCandidate:
    """Load precisely the package-owned Laya ledger, never a benchmark registry."""

    try:
        value = json.loads(
            raw if raw is not None else package_laya_ledger_bytes(), object_pairs_hook=_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, LayaSnapshotError):
            raise
        raise LayaSnapshotError("candidate ledger is invalid") from error
    expected = {
        "schema_version": "laya-candidate.v1",
        "candidate": "laya-multilingual",
        "model_id": "convaiinnovations/laya-multilingual",
        "model_revision": "052592a15d198d9ad47da779604259b10b47b7aa",
        "upstream_source_revision": "573e5b62696ba441230cd6be71d593331b5d23af",
        "acquisition_contract_digest": (
            "be1356e4d2a771830e44b899d1ddd8fc2f30b36943d6becb321afdfecc5b638d"
        ),
        "conformance_vector_digest": (
            "14f2eba88e45d9b9138c3c4ceabc5413a5bdb069f838a6480b4071a9d94fa3b7"
        ),
        "runtime_disposition": "research_only_unresolved_provenance",
    }
    if (
        not isinstance(value, dict)
        or set(value) != {*expected, "files"}
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
    ):
        raise LayaSnapshotError("candidate ledger is not the reviewed Laya ledger")
    entries = value.get("files")
    if not isinstance(entries, list):
        raise LayaSnapshotError("candidate ledger files are invalid")
    files: list[LayaLedgerFile] = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "bytes", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] <= 0
            or not _sha256(entry.get("sha256"))
        ):
            raise LayaSnapshotError("candidate ledger files are invalid")
        files.append(LayaLedgerFile(entry["path"], entry["bytes"], entry["sha256"]))
    expected_files = (
        (
            "encoder/config.json",
            1938,
            "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
        ),
        (
            "rl_agent_config.json",
            472,
            "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
        ),
        (
            "tokenizer/tokenizer_config.json",
            502,
            "424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7",
        ),
        (
            "tokenizer/tokenizer.json",
            34363188,
            "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
        ),
        (
            "model.safetensors",
            643835514,
            "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        ),
    )
    if tuple((item.path, item.bytes, item.sha256) for item in files) != expected_files:
        raise LayaSnapshotError("candidate ledger file order is invalid")
    return LayaCandidate(
        candidate=expected["candidate"],
        model_id=expected["model_id"],
        model_revision=expected["model_revision"],
        upstream_source_revision=expected["upstream_source_revision"],
        acquisition_contract_digest=expected["acquisition_contract_digest"],
        conformance_vector_digest=expected["conformance_vector_digest"],
        runtime_disposition=expected["runtime_disposition"],
        files=tuple(files),
    )


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _check_directory(value: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o700
    ):
        raise LayaSnapshotError("snapshot contains an unsafe directory")


def _check_file(value: os.stat_result, expected: LayaLedgerFile) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
        or value.st_size != expected.bytes
    ):
        raise LayaSnapshotError("snapshot contains an unsafe file")


def _components(path: Path) -> tuple[str, ...]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise LayaSnapshotError("snapshot directory path is invalid")
    return tuple(part for part in absolute.parts[1:] if part not in {"", "."})


def _open_root(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current: int | None = None
    try:
        current = os.open(Path("/").anchor, flags)
        for component in _components(path):
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        assert current is not None
        _check_directory(os.fstat(current))
        return current
    except (OSError, LayaSnapshotError) as error:
        if current is not None:
            os.close(current)
        if isinstance(error, LayaSnapshotError):
            raise
        raise LayaSnapshotError("snapshot directory cannot be opened safely") from error


def _open_dir_at(parent: int, name: str) -> int:
    fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(name, flags, dir_fd=parent)
        _check_directory(os.fstat(fd))
        return fd
    except (OSError, LayaSnapshotError) as error:
        if fd is not None:
            os.close(fd)
        if isinstance(error, LayaSnapshotError):
            raise
        raise LayaSnapshotError("snapshot directory cannot be opened safely") from error


def _open_file_at(parent: int, name: str, expected: LayaLedgerFile) -> int:
    fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(name, flags, dir_fd=parent)
        _check_file(os.fstat(fd), expected)
        return fd
    except (OSError, LayaSnapshotError) as error:
        if fd is not None:
            os.close(fd)
        if isinstance(error, LayaSnapshotError):
            raise
        raise LayaSnapshotError("snapshot file cannot be opened safely") from error


def _tree(root: int) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def walk(directory: int, prefix: str) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    _check_directory(info)
                    fd = _open_dir_at(directory, entry.name)
                    try:
                        if _identity(info) != _identity(os.fstat(fd)):
                            raise LayaSnapshotError("snapshot directory identity changed")
                        directories.add(relative)
                        walk(fd, relative)
                    finally:
                        os.close(fd)
                else:
                    files.add(relative)

    try:
        walk(root, "")
    except OSError as error:
        raise LayaSnapshotError("snapshot directory cannot be enumerated safely") from error
    return files, directories


class LayaSnapshotReceipt:
    """Descriptor-bound, lifetime-scoped readers for one verified snapshot."""

    def __init__(
        self,
        token: object,
        candidate: LayaCandidate,
        root: int,
        directories: Mapping[str, int],
        files: Mapping[str, int],
    ) -> None:
        if token is not _TOKEN:
            raise LayaSnapshotError("snapshot receipts are package-owned")
        self.candidate = candidate
        self._root = root
        self._directories = dict(directories)
        self._files = dict(files)
        self._root_stat = os.fstat(root)
        self._directory_stats = {name: os.fstat(fd) for name, fd in self._directories.items()}
        self._file_stats = {name: os.fstat(fd) for name, fd in self._files.items()}
        self._closed = False

    def __enter__(self) -> LayaSnapshotReceipt:
        self.assert_live()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in self._files.values():
            os.close(fd)
        for fd in self._directories.values():
            os.close(fd)
        os.close(self._root)

    def assert_live(self) -> None:
        if self._closed:
            raise LayaSnapshotError("snapshot receipt is closed")
        if _identity(os.fstat(self._root)) != _identity(self._root_stat):
            raise LayaSnapshotError("snapshot descriptor identity changed")
        for name, fd in self._directories.items():
            if _identity(os.fstat(fd)) != _identity(self._directory_stats[name]):
                raise LayaSnapshotError("snapshot descriptor identity changed")
        for name, fd in self._files.items():
            expected = next(item for item in self.candidate.files if item.path == name)
            current = os.fstat(fd)
            _check_file(current, expected)
            if _identity(current) != _identity(self._file_stats[name]):
                raise LayaSnapshotError("snapshot descriptor identity changed")
        actual_files, actual_directories = _tree(self._root)
        if actual_files != set(self._files) or actual_directories != set(self._directories):
            raise LayaSnapshotError("snapshot tree changed")
        for path, fd in self._files.items():
            parent, name = path.rsplit("/", 1) if "/" in path else ("", path)
            parent_fd = self._directories.get(parent, self._root)
            try:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as error:
                raise LayaSnapshotError("snapshot file identity changed") from error
            if _identity(named) != _identity(os.fstat(fd)):
                raise LayaSnapshotError("snapshot file identity changed")

    def read_verified_bytes(self, path: str) -> bytes:
        self.assert_live()
        if path not in self._files or path == "model.safetensors":
            raise LayaSnapshotError("snapshot reader is not authorized for this file")
        expected = next(item for item in self.candidate.files if item.path == path)
        fd = self._files[path]
        chunks: list[bytes] = []
        offset = 0
        while chunk := os.pread(fd, _CHUNK, offset):
            offset += len(chunk)
            if offset > expected.bytes:
                raise LayaSnapshotError("snapshot file changed while reading")
            chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) != expected.bytes or hashlib.sha256(payload).hexdigest() != expected.sha256:
            raise LayaSnapshotError("snapshot file changed while reading")
        self.assert_live()
        return payload

    def descriptor_path(self, path: str) -> str:
        self.assert_live()
        if path not in self._files:
            raise LayaSnapshotError("snapshot reader is not authorized for this file")
        return f"/dev/fd/{self._files[path]}"

    def descriptor_free_bytes(self, path: str) -> int:
        self.assert_live()
        if path not in self._files:
            raise LayaSnapshotError("snapshot reader is not authorized for this file")
        values = os.fstatvfs(self._files[path])
        return int(values.f_bavail * values.f_frsize)


def verify_laya_snapshot(path: Path) -> LayaSnapshotReceipt:
    """Verify exactly one caller-supplied Laya directory and retain its FDs."""

    candidate = load_laya_candidate()
    root: int | None = None
    directories: dict[str, int] = {}
    files: dict[str, int] = {}
    receipt: LayaSnapshotReceipt | None = None
    try:
        root = _open_root(path)
        expected_files = {item.path for item in candidate.files}
        expected_directories = {"encoder", "tokenizer"}
        actual_files, actual_directories = _tree(root)
        if actual_files != expected_files or actual_directories != expected_directories:
            raise LayaSnapshotError("snapshot tree does not match the reviewed ledger")
        for name in sorted(expected_directories):
            directories[name] = _open_dir_at(root, name)
        for item in candidate.files:
            parent, name = item.path.rsplit("/", 1) if "/" in item.path else ("", item.path)
            files[item.path] = _open_file_at(directories.get(parent, root), name, item)
        receipt = LayaSnapshotReceipt(_TOKEN, candidate, root, directories, files)
        root = None
        directories = {}
        files = {}
        receipt.assert_live()
        for item in candidate.files:
            fd = receipt._files[item.path]
            digest = hashlib.sha256()
            offset = 0
            while chunk := os.pread(fd, _CHUNK, offset):
                offset += len(chunk)
                digest.update(chunk)
            if offset != item.bytes or digest.hexdigest() != item.sha256:
                raise LayaSnapshotError("snapshot file ledger mismatch")
            receipt.assert_live()
        return receipt
    except BaseException:
        if receipt is not None:
            receipt.close()
        else:
            for fd in files.values():
                os.close(fd)
            for fd in directories.values():
                os.close(fd)
            if root is not None:
                os.close(root)
        raise
