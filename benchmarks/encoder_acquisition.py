"""Explicit, anonymous, descriptor-verified acquisition of reviewed snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat as statmod
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import BinaryIO, cast

from benchmarks.encoder_registry import Candidate, get_candidate, registry_digest

MAX_REDIRECT_HOSTS = {"huggingface.co", "hf.co"}
EXTRA_FREE_BYTES = 128 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("marker contains duplicate keys")
        result[key] = value
    return result


def _cache_root() -> Path:
    from platformdirs import user_cache_path  # type: ignore[import-not-found]

    return cast(Path, user_cache_path("saracura")) / "encoders"


def snapshot_path(candidate: Candidate, root: Path | None = None) -> Path:
    return (root or _cache_root()) / candidate.id / candidate.revision


def _safe_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower().rstrip(".")
    return host in MAX_REDIRECT_HOSTS or any(
        host.endswith("." + allowed) for allowed in MAX_REDIRECT_HOSTS
    )


def _url(candidate: Candidate, filename: str) -> str:
    return (
        "https://huggingface.co/"
        + candidate.repository
        + "/resolve/"
        + candidate.revision
        + "/"
        + urllib.parse.quote(filename)
    )


def _validate_regular_fd(fd: int, mode: int) -> os.stat_result:
    result = os.fstat(fd)
    if (
        not statmod.S_ISREG(result.st_mode)
        or result.st_nlink != 1
        or result.st_uid != os.getuid()
        or result.st_mode & 0o777 != mode
    ):
        raise ValueError("snapshot contains an unsafe regular file")
    return result


def _open_regular(path: Path, mode: int) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        result = _validate_regular_fd(fd, mode)
    except BaseException:
        os.close(fd)
        raise
    return fd, result


def _open_regular_at(directory_fd: int, name: str, mode: int) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        result = _validate_regular_fd(fd, mode)
    except BaseException:
        os.close(fd)
        raise
    return fd, result


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    stat = os.fstat(fd)
    if (
        not statmod.S_ISDIR(stat.st_mode)
        or stat.st_uid != os.getuid()
        or stat.st_mode & 0o777 != 0o700
    ):
        os.close(fd)
        raise ValueError("snapshot contains an unsafe directory")
    return fd, stat


def _verify_file(path: Path, expected_bytes: int, expected_sha: str) -> None:
    fd, stat = _open_regular(path, 0o600)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(fd)
    if stat.st_size != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError("snapshot file ledger mismatch")


def _verify_file_at(directory_fd: int, name: str, expected_bytes: int, expected_sha: str) -> None:
    fd, result = _open_regular_at(directory_fd, name, 0o600)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(fd)
    if result.st_size != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError("snapshot file ledger mismatch")


def verify_snapshot(candidate: Candidate, path: Path | None = None) -> bool:
    if candidate.disposition != "eligible":
        raise ValueError("blocked candidate cannot be acquired")
    final = path or snapshot_path(candidate)
    final_fd: int | None = None
    try:
        final_fd, final_stat = _open_directory(final)
        marker_fd, marker_stat = _open_regular_at(final_fd, "snapshot.complete.json", 0o600)
        try:
            if marker_stat.st_size > 1024 * 1024:
                return False
            marker_payload = json.loads(
                os.read(marker_fd, marker_stat.st_size), object_pairs_hook=_reject_duplicate_keys
            )
        finally:
            os.close(marker_fd)
        if marker_stat.st_dev != final_stat.st_dev or marker_stat.st_nlink != 1:
            return False
        if not isinstance(marker_payload, dict):
            return False
        if set(marker_payload) != {"candidate_id", "revision", "registry_sha256", "files"}:
            return False
        if marker_payload.get("candidate_id") != candidate.id:
            return False
        if marker_payload.get("revision") != candidate.revision:
            return False
        if marker_payload.get("registry_sha256") != registry_digest():
            return False
        expected = {item.path: item for item in candidate.files}
        expected_ledger = [
            {"path": item.path, "bytes": item.bytes, "sha256": item.sha256}
            for item in candidate.files
        ]
        if marker_payload.get("files") != expected_ledger:
            return False
        with os.scandir(final_fd) as entries:
            names = {item.name for item in entries if item.name != "snapshot.complete.json"}
        if names != set(expected):
            return False
        for name, item in expected.items():
            _verify_file_at(final_fd, name, item.bytes, item.sha256)
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    finally:
        if final_fd is not None:
            os.close(final_fd)


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _ClosedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or not _safe_host(parsed.hostname):
            raise ValueError("redirect is outside the HTTPS host allowlist")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _stream(url: str, destination: Path, expected_size: int) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not _safe_host(parsed.hostname):
        raise ValueError("download URL is outside the HTTPS host allowlist")
    request = urllib.request.Request(url, headers={"User-Agent": "saracura-encoder-gate/1"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _ClosedRedirectHandler())
    with opener.open(request, timeout=60) as response:
        final_url = urllib.parse.urlparse(response.geturl())
        if final_url.scheme != "https" or not _safe_host(final_url.hostname):
            raise ValueError("download redirect host is not allowlisted")
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            _copy_bounded(response, handle, expected_size)
            handle.flush()
            os.fsync(handle.fileno())


def _copy_bounded(source: BinaryIO, target: BinaryIO, expected_size: int) -> None:
    total = 0
    while True:
        chunk = source.read(min(1024 * 1024, expected_size + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise ValueError("download exceeds the registry size cap")
        target.write(chunk)
    if total != expected_size:
        raise ValueError("download size does not match registry")


def _create_final(path: Path) -> None:
    path.mkdir(mode=0o700)


def acquire(candidate_id: str, *, allow_network: bool, root: Path | None = None) -> Path:
    candidate = get_candidate(candidate_id)
    if not allow_network:
        raise ValueError("explicit --allow-network is required")
    if candidate.disposition != "eligible":
        raise ValueError("candidate is blocked by the reviewed registry")
    final = snapshot_path(candidate, root)
    if verify_snapshot(candidate, final):
        return final
    parent = final.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    claim = parent / f".{candidate.revision}.claim"
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("candidate acquisition is already in progress") from error
    claim_stat = os.stat(claim, follow_symlinks=False)
    staging = parent / f".{candidate.revision}.{uuid.uuid4().hex}.staging"
    owned_final = False
    final_stat: os.stat_result | None = None
    final_fd: int | None = None
    try:
        usage = shutil.disk_usage(parent)
        if usage.free < candidate.total_bytes + EXTRA_FREE_BYTES:
            raise ValueError("insufficient free space for candidate")
        staging.mkdir(mode=0o700)
        for item in candidate.files:
            target = staging / item.path
            _stream(_url(candidate, item.path), target, item.bytes)
            _verify_file(target, item.bytes, item.sha256)
        _fsync(staging)
        _create_final(final)
        owned_final = True
        final_fd, final_stat = _open_directory(final)
        for item in candidate.files:
            os.rename(staging / item.path, item.path, dst_dir_fd=final_fd)
            _verify_file_at(final_fd, item.path, item.bytes, item.sha256)
            fd, _ = _open_regular_at(final_fd, item.path, 0o600)
            os.fsync(fd)
            os.close(fd)
        marker = {
            "candidate_id": candidate.id,
            "revision": candidate.revision,
            "registry_sha256": registry_digest(),
            "files": [
                {"path": item.path, "bytes": item.bytes, "sha256": item.sha256}
                for item in candidate.files
            ],
        }
        descriptor = os.open(
            "snapshot.complete.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=final_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(marker, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(final_fd)
        _fsync(parent)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        final_owned = False
        if owned_final and final_stat is not None:
            try:
                current = os.stat(final, follow_symlinks=False)
                final_owned = (current.st_dev, current.st_ino) == (
                    final_stat.st_dev,
                    final_stat.st_ino,
                )
            except OSError:
                pass
        if final_owned and not verify_snapshot(candidate, final):
            shutil.rmtree(final, ignore_errors=True)
        raise
    finally:
        if final_fd is not None:
            os.close(final_fd)
        try:
            current = os.stat(claim, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (claim_stat.st_dev, claim_stat.st_ino):
                shutil.rmtree(claim, ignore_errors=True)
        except OSError:
            pass
