"""Anonymous, recursive, descriptor-bound acquisition for registry-v2 snapshots."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import stat
import sys
import urllib.parse
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, cast

from benchmarks.universal_local.registry import (
    Candidate,
    RegistryFile,
    file_url,
    get_candidate,
    registry_digest,
)

MAX_REDIRECT_HOSTS = {"huggingface.co", "hf.co"}
EXTRA_FREE_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 3 * 1024 * 1024 * 1024
CHUNK = 1024 * 1024


class SnapshotConflictError(ValueError):
    """An immutable but invalid final snapshot already occupies the contract key."""


def cache_root() -> Path:
    """Resolve the private cache only when an acquisition operation needs it."""
    from platformdirs import user_cache_path  # type: ignore[import-not-found]

    return Path(str(user_cache_path("saracura"))) / "universal-local"


def snapshot_path(candidate: Candidate, root: Path | None = None) -> Path:
    return (root or cache_root()) / str(candidate.id) / str(candidate.acquisition_contract_digest)


def _safe_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.lower().rstrip(".")
    return normalized in MAX_REDIRECT_HOSTS or any(
        normalized.endswith("." + allowed) for allowed in MAX_REDIRECT_HOSTS
    )


def _validate_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not _safe_host(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("download URL is outside the HTTPS host allowlist")
    return parsed


class _ClosedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str):  # type: ignore[no-untyped-def]
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_directory_fd(fd: int) -> os.stat_result:
    result = os.fstat(fd)
    if (
        not stat.S_ISDIR(result.st_mode)
        or result.st_uid != os.getuid()
        or result.st_mode & 0o777 != 0o700
        or result.st_nlink < 2
    ):
        raise ValueError("snapshot contains an unsafe directory")
    return result


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        result = _validate_directory_fd(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd, result


def _open_directory_at(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        result = _validate_directory_fd(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd, result


def _open_regular_at(directory_fd: int, name: str) -> tuple[int, os.stat_result]:
    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    result = os.fstat(fd)
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_uid != os.getuid()
        or result.st_mode & 0o777 != 0o600
        or result.st_nlink != 1
    ):
        os.close(fd)
        raise ValueError("snapshot contains an unsafe regular file")
    return fd, result


def _open_relative(root_fd: int, relative: str) -> tuple[int, os.stat_result]:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise ValueError("unsafe relative snapshot path")
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd, _ = _open_directory_at(current, part)
            os.close(current)
            current = next_fd
        return _open_regular_at(current, parts[-1])
    finally:
        os.close(current)


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(fd, CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _stable_identity(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_nlink,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _json_marker(fd: int) -> dict[str, Any]:
    result = os.fstat(fd)
    if result.st_size > 1024 * 1024:
        raise ValueError("snapshot completion marker is too large")
    payload = json.loads(
        os.read(fd, result.st_size).decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("snapshot completion marker is invalid")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("snapshot marker contains duplicate keys")
        result[key] = value
    return result


def _expected_tree(candidate: Candidate) -> tuple[set[str], set[str]]:
    files = {item["path"] for item in candidate.data["files"]} | {"snapshot.complete.json"}
    directories: set[str] = set()
    for path in files:
        parent = Path(path).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
    return files, directories


def _walk_tree(root_fd: int, relative: str = "") -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    directory_fd = os.dup(root_fd) if not relative else _open_relative_directory(root_fd, relative)
    directory_stat = os.fstat(directory_fd)
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                name = entry.name
                child = f"{relative}/{name}" if relative else name
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(entry_stat.st_mode):
                    if entry_stat.st_uid != os.getuid() or entry_stat.st_mode & 0o777 != 0o700:
                        raise ValueError("snapshot contains an unsafe directory")
                    child_fd, child_stat = _open_directory_at(directory_fd, name)
                    os.close(child_fd)
                    if (entry_stat.st_dev, entry_stat.st_ino) != (
                        child_stat.st_dev,
                        child_stat.st_ino,
                    ):
                        raise ValueError("snapshot directory identity changed")
                    directories.add(child)
                    child_files, child_dirs = _walk_tree(root_fd, child)
                    files.update(child_files)
                    directories.update(child_dirs)
                else:
                    files.add(child)
    finally:
        current_stat = os.fstat(directory_fd)
        os.close(directory_fd)
        if _stable_identity(current_stat) != _stable_identity(directory_stat):
            raise ValueError("snapshot directory identity changed")
    return files, directories


def _open_relative_directory(root_fd: int, relative: str) -> int:
    current = os.dup(root_fd)
    try:
        for part in Path(relative).parts:
            next_fd, _ = _open_directory_at(current, part)
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


_RECEIPT_TOKEN = object()


class VerificationReceipt:
    """Live descriptor custody for one recursively verified snapshot."""

    def __init__(
        self,
        token: object,
        candidate: Candidate,
        root_fd: int,
        root_stat: os.stat_result,
        file_fds: dict[str, int],
        file_stats: dict[str, os.stat_result],
        directory_fds: dict[str, int],
        directory_stats: dict[str, os.stat_result],
        registry_sha256: str,
        file_ledger: tuple[RegistryFile, ...],
    ) -> None:
        if token is not _RECEIPT_TOKEN:
            raise ValueError("verification receipts cannot be constructed by callers")
        self._candidate_id = candidate.id
        self._model_revision = candidate.model_revision
        self._acquisition_contract_digest = candidate.acquisition_contract_digest
        self._registry_sha256 = registry_sha256
        self._expected_files = {item.path: item for item in file_ledger}
        self._root_fd = root_fd
        self._root_stat = root_stat
        self._file_fds = file_fds
        self._file_stats = file_stats
        self._directory_fds = directory_fds
        self._directory_stats = directory_stats
        self._closed = False

    def __enter__(self) -> VerificationReceipt:
        self.assert_live()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in self._file_fds.values():
            os.close(fd)
        for fd in self._directory_fds.values():
            os.close(fd)
        os.close(self._root_fd)

    def assert_live(self) -> None:
        if self._closed:
            raise ValueError("verification receipt is closed")
        if _stable_identity(os.fstat(self._root_fd)) != _stable_identity(self._root_stat):
            raise ValueError("verification receipt is stale")
        for path, fd in self._file_fds.items():
            if _stable_identity(os.fstat(fd)) != _stable_identity(self._file_stats[path]):
                raise ValueError("verification receipt is stale")
        for path, fd in self._directory_fds.items():
            if _stable_identity(os.fstat(fd)) != _stable_identity(self._directory_stats[path]):
                raise ValueError("verification receipt is stale")

    def read_verified_bytes(self, path: str) -> bytes:
        """Read one allowlisted file from its retained descriptor, never by pathname."""
        self.assert_live()
        if path == "snapshot.complete.json" or path not in self._file_fds:
            raise ValueError("file is not available to snapshot consumers")
        expected = self._expected_files[path]
        fd = self._file_fds[path]
        before = os.fstat(fd)
        chunks: list[bytes] = []
        offset = 0
        while chunk := os.pread(fd, CHUNK, offset):
            chunks.append(chunk)
            offset += len(chunk)
            if offset > expected.bytes:
                raise ValueError("verified file changed while reading")
        payload = b"".join(chunks)
        after = os.fstat(fd)
        if (
            len(payload) != expected.bytes
            or hashlib.sha256(payload).hexdigest() != expected.sha256
            or _stable_identity(before) != _stable_identity(after)
            or _stable_identity(after) != _stable_identity(self._file_stats[path])
        ):
            raise ValueError("verified file changed while reading")
        return payload

    def summary(self) -> dict[str, Any]:
        """Return a non-authoritative, path-free CLI summary while custody is live."""
        self.assert_live()
        return {
            "candidate_id": self._candidate_id,
            "model_revision": self._model_revision,
            "acquisition_contract_digest": self._acquisition_contract_digest,
            "registry_sha256": self._registry_sha256,
            "files": list(self._expected_files),
        }


def verification_receipt(candidate: Candidate, path: Path | None = None) -> VerificationReceipt:
    """Open, verify, and retain all descriptors required by a later loader."""
    final = path or snapshot_path(candidate)
    root_fd: int | None = None
    opened: dict[str, int] = {}
    stats: dict[str, os.stat_result] = {}
    opened_directories: dict[str, int] = {}
    directory_stats: dict[str, os.stat_result] = {}
    try:
        root_fd, root_stat = _open_directory(final)
        expected_files, expected_dirs = _expected_tree(candidate)
        for relative in sorted(expected_dirs):
            directory_fd = _open_relative_directory(root_fd, relative)
            opened_directories[relative] = directory_fd
            directory_stats[relative] = os.fstat(directory_fd)
        actual_files, actual_dirs = _walk_tree(root_fd)
        if actual_files != expected_files or actual_dirs != expected_dirs:
            raise ValueError("snapshot tree mismatch")
        marker_fd, marker_stat = _open_regular_at(root_fd, "snapshot.complete.json")
        opened["snapshot.complete.json"] = marker_fd
        stats["snapshot.complete.json"] = marker_stat
        marker = _json_marker(marker_fd)
        marker_current = os.fstat(marker_fd)
        registry_sha256 = registry_digest()
        file_ledger = candidate.files
        expected_ledger = [
            {"path": item.path, "bytes": item.bytes, "sha256": item.sha256, "role": item.role}
            for item in file_ledger
        ]
        if marker != {
            "candidate_id": candidate.id,
            "model_revision": candidate.model_revision,
            "acquisition_contract_digest": candidate.acquisition_contract_digest,
            "files": expected_ledger,
        }:
            raise ValueError("snapshot marker mismatch")
        if marker_stat.st_dev != root_stat.st_dev or _stable_identity(
            marker_current
        ) != _stable_identity(marker_stat):
            raise ValueError("snapshot marker changed during verification")
        for item in candidate.data["files"]:
            fd, result = _open_relative(root_fd, item["path"])
            opened[item["path"]] = fd
            stats[item["path"]] = result
            digest = _hash_fd(fd)
            current = os.fstat(fd)
            if (
                result.st_size != item["bytes"]
                or digest != item["sha256"]
                or _stable_identity(current) != _stable_identity(result)
            ):
                raise ValueError("snapshot file ledger mismatch")
        if _stable_identity(os.fstat(root_fd)) != _stable_identity(root_stat):
            raise ValueError("snapshot root changed during verification")
        for relative, directory_fd in opened_directories.items():
            if _stable_identity(os.fstat(directory_fd)) != _stable_identity(
                directory_stats[relative]
            ):
                raise ValueError("snapshot directory changed during verification")
        receipt = VerificationReceipt(
            _RECEIPT_TOKEN,
            candidate,
            root_fd,
            root_stat,
            opened,
            stats,
            opened_directories,
            directory_stats,
            registry_sha256,
            file_ledger,
        )
        root_fd = None
        opened = {}
        opened_directories = {}
        return receipt
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("snapshot verification failed") from error
    finally:
        for fd in opened.values():
            os.close(fd)
        for fd in opened_directories.values():
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


def verify_snapshot(candidate: Candidate, path: Path | None = None) -> bool:
    """Return true only for a complete snapshot with a live descriptor receipt."""
    try:
        with verification_receipt(candidate, path):
            return True
    except ValueError:
        return False


def _copy_bounded(source: BinaryIO, target_fd: int, expected_size: int) -> None:
    total = 0
    while True:
        chunk = source.read(min(CHUNK, expected_size + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise ValueError("download exceeds the registry size cap")
        view = memoryview(chunk)
        while view:
            view = view[os.write(target_fd, view) :]
    if total != expected_size:
        raise ValueError("download size does not match registry")


def _stream(url: str, directory_fd: int, relative: str, expected_size: int) -> None:
    _validate_url(url)
    parts = Path(relative).parts
    parent = "." if len(parts) == 1 else str(Path(*parts[:-1]))
    parent_fd = (
        _open_relative_directory(directory_fd, parent) if parent != "." else os.dup(directory_fd)
    )
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "saracura-universal-local/1"})
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _ClosedRedirectHandler()
        )
        with opener.open(request, timeout=60) as response:
            final_url = _validate_url(response.geturl())
            if final_url.query and not _safe_host(final_url.hostname):
                raise ValueError("download redirect is unsafe")
            length = response.headers.get("Content-Length")
            if length is None or int(length) != expected_size:
                raise ValueError("download content length does not match registry")
            fd = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            try:
                _copy_bounded(response, fd, expected_size)
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        os.close(parent_fd)


def _make_directory_at(parent_fd: int, name: str) -> int:
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    fd, _ = _open_directory_at(parent_fd, name)
    return fd


def _open_or_create_private_directory(path: Path) -> int:
    with suppress(FileExistsError):
        path.mkdir(parents=True, mode=0o700)
    fd, _ = _open_directory(path)
    return fd


def _rename_no_replace(parent_fd: int, stage: str, final: str) -> None:
    """Atomically publish a directory without replacing an existing name."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is not None:
            renameatx_np.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameatx_np.restype = ctypes.c_int
            if renameatx_np(parent_fd, stage.encode(), parent_fd, final.encode(), 0x00000004) == 0:
                return
        raise ValueError("atomic no-replace snapshot publication failed")
    if sys.platform.startswith("linux"):
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
        raise ValueError("atomic no-replace snapshot publication failed")
    raise ValueError("atomic no-replace snapshot publication is unavailable")


def _ensure_tree(root_fd: int, relative: str) -> None:
    current = os.dup(root_fd)
    try:
        for part in Path(relative).parts:
            try:
                next_fd, _ = _open_directory_at(current, part)
            except FileNotFoundError:
                next_fd = _make_directory_at(current, part)
            os.close(current)
            current = next_fd
    finally:
        os.close(current)


def _write_marker(root_fd: int, candidate: Candidate) -> None:
    marker = {
        "candidate_id": candidate.id,
        "model_revision": candidate.model_revision,
        "acquisition_contract_digest": candidate.acquisition_contract_digest,
        "files": [
            {"path": item.path, "bytes": item.bytes, "sha256": item.sha256, "role": item.role}
            for item in candidate.files
        ],
    }
    encoded = json.dumps(marker, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    fd = os.open(
        "snapshot.complete.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd
    )
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def acquire(candidate_id: str, *, allow_network: bool, root: Path | None = None) -> Path:
    candidate = get_candidate(candidate_id)
    if not allow_network:
        raise ValueError("explicit --allow-network is required")
    if candidate.acquisition_state != "eligible":
        raise ValueError("candidate is blocked by the reviewed registry")
    if candidate.total_bytes > MAX_TOTAL_BYTES or any(
        item.bytes > MAX_FILE_BYTES for item in candidate.files
    ):
        raise ValueError("candidate exceeds acquisition bounds")
    contract_digest = cast(str, candidate.acquisition_contract_digest)
    base = root or cache_root()
    base_fd = _open_or_create_private_directory(base)
    try:
        try:
            parent_fd, _ = _open_directory_at(base_fd, candidate.id)
        except FileNotFoundError:
            parent_fd = _make_directory_at(base_fd, candidate.id)
    finally:
        os.close(base_fd)
    final = base / candidate.id / contract_digest
    if verify_snapshot(candidate, final):
        os.close(parent_fd)
        return final
    try:
        os.stat(contract_digest, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        os.close(parent_fd)
        raise SnapshotConflictError(
            "invalid immutable snapshot exists; inspect and move it aside before retrying"
        )
    parent = final.parent
    claim_name = f".{contract_digest}.claim"
    try:
        os.mkdir(claim_name, 0o700, dir_fd=parent_fd)
    except FileExistsError as error:
        os.close(parent_fd)
        raise ValueError("candidate acquisition is already in progress") from error
    staging_name = f".{uuid.uuid4().hex}.staging"
    staging = parent / staging_name
    try:
        if shutil.disk_usage(parent).free < candidate.total_bytes + EXTRA_FREE_BYTES:
            raise ValueError("insufficient free space for candidate")
        staging_fd = _make_directory_at(parent_fd, staging_name)
        try:
            for item in candidate.files:
                parent_path = str(Path(item.path).parent)
                if parent_path != ".":
                    _ensure_tree(staging_fd, parent_path)
                _stream(file_url(candidate, item.path), staging_fd, item.path, item.bytes)
            _write_marker(staging_fd, candidate)
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        if not verify_snapshot(candidate, staging):
            raise ValueError("staged snapshot failed exact-byte verification")
        _rename_no_replace(parent_fd, staging_name, contract_digest)
        os.fsync(parent_fd)
        if not verify_snapshot(candidate, final):
            raise ValueError("promoted snapshot failed verification")
        return final
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            os.rmdir(claim_name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
