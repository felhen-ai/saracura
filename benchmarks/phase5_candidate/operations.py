"""Phase 5B explicit acquisition and evaluation operations.

These functions are the sole live entrypoints for ``describe``, ``prepare``,
``serve`` and ``evaluate``.  They are fail-closed: any missing optional
dependency, mutable revision, unsafe cache permission, non-loopback endpoint,
arbitrary path, digest mismatch or extra artifact raises instead of degrading.

Optional orchestration imports (``platformdirs``, ``psutil``,
``huggingface_hub``) happen lazily inside each command.  Nothing here registers
a backend, mutates the installed Saracura runtime, or downloads on ordinary
requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from benchmarks.phase5_candidate.descriptor import (
    CANDIDATE_ID,
    SCHEMA_VERSION,
    AcquisitionDescriptor,
    ThresholdsBlock,
    derive_disposition,
)
from benchmarks.phase5_candidate.report import SystemOneResponse

ROOT = Path(__file__).parents[2]

_ALLOWED_HOSTS = (
    "github.com",
    "raw.githubusercontent.com",
    "pypi.org",
    "files.pythonhosted.org",
    "huggingface.co",
    "hf.co",
    "xethub.hf.co",
)

_FULL_GIT_REV = re.compile(r"^[0-9a-f]{40}$")
_IMMUTABLE_REV = re.compile(r"^[0-9a-f]{64}$")

_KEV_SOURCE_REVISION = "9c41005b2180347c3c646dfc9e50c4428483ec6b"
_KEV_MODEL_REVISION = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
_QWEN_BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
_QWEN_BASE_ID = "Qwen/Qwen3.5-4B-Base"
_KEV_SOURCE_URL = "https://github.com/jaredpalmer/kev"
_KEV_MODEL_ID = "jaredpalmer/kev-4b"
_KEV_UV_LOCK_SHA256 = "a9922dbb89acdef78299fd2b4a8c3f7f0fa1b2bc08b55595b6926fa785a9c466"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _private_root() -> Path:
    configured = os.environ.get("SARACURA_PHASE5B_CACHE_ROOT")
    if configured:
        root = Path(configured)
        if not root.is_absolute():
            raise ValueError("SARACURA_PHASE5B_CACHE_ROOT must be an absolute path")
        return root
    from platformdirs import user_cache_path  # type: ignore[import-not-found,unused-ignore]

    return Path(user_cache_path("saracura")) / "phase5b"


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    if path.stat().st_mode & 0o077:
        raise RuntimeError("private cache directory permissions are unsafe")


def _force_hf_safety_flags() -> None:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def _validate_acquisition_capacity(
    descriptor: AcquisitionDescriptor,
    *,
    free_disk_bytes: int,
    physical_memory_bytes: int,
) -> None:
    source_bytes = sum(entry.bytes for entry in descriptor.source.files)
    source_count = len(descriptor.source.files)
    total_bytes = source_bytes + sum(
        entry.bytes
        for snapshot in (descriptor.checkpoint, descriptor.base_model)
        for entry in snapshot.files
    )
    limits = descriptor.limits
    violations = []
    if source_count > limits.max_source_file_count:
        violations.append("source file count")
    if source_bytes > limits.max_source_bytes:
        violations.append("source bytes")
    if total_bytes > limits.max_total_bytes:
        violations.append("total bytes")
    if free_disk_bytes < limits.min_free_disk_bytes:
        violations.append("free disk")
    if physical_memory_bytes < limits.min_physical_memory_bytes:
        violations.append("physical memory")
    if violations:
        raise ValueError("capacity limits exceeded: " + ", ".join(violations))


def _preflight_capacity(descriptor: AcquisitionDescriptor, root: Path) -> None:
    import psutil  # type: ignore[import-untyped]

    _ensure_private_dir(root)
    _validate_acquisition_capacity(
        descriptor,
        free_disk_bytes=shutil.disk_usage(root).free,
        physical_memory_bytes=int(psutil.virtual_memory().total),
    )


def _remove_git_metadata(checkout: Path) -> None:
    metadata = checkout / ".git"
    if metadata.is_symlink():
        raise RuntimeError("source checkout .git metadata may not be a symlink")
    if metadata.is_dir():
        shutil.rmtree(metadata)
    elif metadata.exists():
        metadata.unlink()


def _verify_hf_snapshot_identity(model_id: str, revision: str, local_dir: Path) -> None:
    """Bind a downloaded snapshot to its immutable Hub revision and LFS bytes."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found,unused-ignore]

    info = HfApi(token=False).model_info(
        model_id,
        revision=revision,
        files_metadata=True,
    )
    if info.sha != revision:
        raise RuntimeError(f"Hugging Face resolved an unexpected revision for {model_id}")
    if info.siblings is None:
        raise RuntimeError(f"Hugging Face returned no file metadata for {model_id}")
    remote_paths = {sibling.rfilename for sibling in info.siblings}
    local_paths = {
        path.relative_to(local_dir).as_posix()
        for path in local_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if local_paths != remote_paths:
        raise RuntimeError(f"Hugging Face snapshot file set mismatch for {model_id}")
    for sibling in info.siblings:
        lfs = sibling.lfs
        if lfs is None:
            continue
        path = local_dir / sibling.rfilename
        if path.stat().st_size != lfs.size or _sha256_file(path) != lfs.sha256:
            raise RuntimeError(f"Hugging Face LFS identity mismatch: {sibling.rfilename}")


def _remove_hf_local_metadata(local_dir: Path) -> None:
    metadata = local_dir / ".cache"
    if metadata.is_symlink():
        raise RuntimeError("Hugging Face local metadata may not be a symlink")
    if metadata.exists():
        shutil.rmtree(metadata)


def _build_hf_runtime_view(root: Path, descriptor: AcquisitionDescriptor) -> None:
    """Expose the verified base snapshot through an isolated HF cache view."""
    hub = root / "hf-home" / "hub"
    if hub.exists():
        if hub.is_symlink():
            raise RuntimeError("Hugging Face runtime cache root may not be a symlink")
        shutil.rmtree(hub)
    snapshot = (
        hub
        / f"models--{descriptor.base_model.model_id.replace('/', '--')}"
        / "snapshots"
        / descriptor.base_model.revision
    )
    _ensure_private_dir(snapshot)
    expected: set[str] = set()
    payload_base = (root / "payload" / "base").resolve()
    for entry in descriptor.base_model.files:
        source = (payload_base / entry.path).resolve()
        if not source.is_relative_to(payload_base) or not source.is_file():
            raise RuntimeError(f"invalid base payload path for runtime view: {entry.path}")
        target = snapshot / entry.path
        _ensure_private_dir(target.parent)
        target.symlink_to(source)
        if target.resolve() != source:
            raise RuntimeError(f"Hugging Face runtime link escaped its payload: {entry.path}")
        expected.add(entry.path)
    observed = {
        path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError("Hugging Face runtime view does not match the base ledger")


def _validate_models_identity(
    models: object,
    *,
    expected_checkpoint: Path,
    expected_base_id: str,
) -> None:
    run = getattr(models, "run", None)
    if not isinstance(run, str) or Path(run).resolve() != expected_checkpoint.resolve():
        raise ValueError("models.run exact checkpoint identity mismatch")
    if getattr(models, "base_model_id", None) != expected_base_id:
        raise ValueError("models base_model_id mismatch")
    if getattr(models, "backend", None) != "mlx" or getattr(models, "dtype", None) != "bfloat16":
        raise ValueError("models backend/dtype mismatch")


def _validate_systemone_result(
    payload: dict[str, object],
    *,
    question_options: dict[str, set[str]],
    expected_run: Path,
) -> SystemOneResponse:
    try:
        response = SystemOneResponse.model_validate(payload)
    except ValueError as error:
        raise ValueError("response does not match the request criteria") from error
    if set(response.answers) != set(question_options):
        raise ValueError("response question ids do not match the request")
    if Path(response.run).resolve() != expected_run.resolve():
        raise ValueError("response run does not match the exact local checkpoint")
    for question_id, answer in response.answers.items():
        if set(answer.probabilities) != question_options[question_id]:
            raise ValueError("response choices do not match the request criteria")
    return response


def _probability_delta(left: dict[str, float], right: dict[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("probability labels differ between evaluations")
    return max((abs(left[key] - right[key]) for key in left), default=0.0)


def _compare_choice_answers(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    near_tie_margin: float,
) -> tuple[float, bool, tuple[tuple[str, float], ...]]:
    from benchmarks.phase5_candidate.report import ChoiceAnswer

    first = {key: ChoiceAnswer.model_validate(value) for key, value in reference.items()}
    second = {key: ChoiceAnswer.model_validate(value) for key, value in candidate.items()}
    if set(first) != set(second):
        raise ValueError("answer ids differ")
    maximum_delta = 0.0
    stable = True
    accepted_near_ties: list[tuple[str, float]] = []
    for question_id, baseline in first.items():
        comparison = second[question_id]
        delta = _probability_delta(baseline.probabilities, comparison.probabilities)
        maximum_delta = max(maximum_delta, delta)
        if baseline.choice != comparison.choice:
            baseline_ordered = sorted(baseline.probabilities.values(), reverse=True)
            comparison_ordered = sorted(comparison.probabilities.values(), reverse=True)
            baseline_margin = (
                baseline_ordered[0] - baseline_ordered[1] if len(baseline_ordered) > 1 else 1.0
            )
            comparison_margin = (
                comparison_ordered[0] - comparison_ordered[1]
                if len(comparison_ordered) > 1
                else 1.0
            )
            if max(baseline_margin, comparison_margin) >= near_tie_margin:
                stable = False
            else:
                accepted_near_ties.append((question_id, baseline_margin))
    return maximum_delta, stable, tuple(accepted_near_ties)


def _evaluation_matrix_plan() -> tuple[tuple[str, int, int, int, int], ...]:
    """(locale, Q, new-state repeats, cached-state repeats, exact repeats)."""
    from benchmarks.phase5_candidate.fixture import matrix as fixture_matrix

    return tuple(
        (locale, len(request.questions), 20, 20, 3)
        for locale, workloads in fixture_matrix().items()
        for request in workloads.values()
    )


def _descriptor_root(descriptor_path: Path) -> Path:
    """The descriptor-digest-scoped private root for a specific acquisition."""
    raw = descriptor_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return _private_root() / digest


def _validate_descriptor_bytes(descriptor_path: Path) -> AcquisitionDescriptor:
    raw = descriptor_path.read_bytes()
    payload = json.loads(raw)
    descriptor = AcquisitionDescriptor.model_validate(payload)
    if descriptor.schema_version != SCHEMA_VERSION or descriptor.candidate_id != CANDIDATE_ID:
        raise ValueError("descriptor is not the kev-4b acquisition authority")
    return descriptor


def _require_optional_modules() -> None:
    """Lazily import the opt-in orchestration modules or fail closed."""
    import huggingface_hub
    import platformdirs
    import psutil

    _ = (platformdirs, psutil, huggingface_hub)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _strip_payload_cache_artifacts(path: Path) -> None:
    # Any symlink, __pycache__ or bytecode inside the immutable payload is a
    # contract violation and fails closed.
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(f"payload symlink is forbidden: {candidate}")
        if "__pycache__" in candidate.parts:
            raise RuntimeError(f"payload bytecode cache is forbidden: {candidate}")


def _loops_only(pid: int) -> bool:
    """Reject any non-loopback listener for the served process (lazy psutil).

    Fails closed: if ``psutil`` is unavailable, or the process cannot be
    inspected, the process is treated as non-conforming rather than silently
    passing.  Returning ``True`` on a missing optional dependency would mask a
    broken egress control.
    """
    import psutil

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    try:
        connections = process.net_connections(kind="inet")
    except psutil.NoSuchProcess:
        return False
    for connection in connections:
        addresses = [getattr(connection, "laddr", None), getattr(connection, "raddr", None)]
        for address in addresses:
            ip = getattr(address, "ip", None)
            if ip and ip not in ("127.0.0.1", "::1"):
                return False
        if getattr(connection, "laddr", None) and connection.laddr.ip not in ("127.0.0.1", "::1"):
            return False
    return True


def _pid_owns_port(pid: int, port: int) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        connections = process.net_connections(kind="inet")
    except psutil.NoSuchProcess:
        return False
    for connection in connections:
        local = getattr(connection, "laddr", None)
        if (
            local is not None
            and getattr(local, "port", None) == port
            and getattr(connection, "status", "LISTEN") == "LISTEN"
            and getattr(local, "ip", None) in ("127.0.0.1", "::1")
        ):
            return True
    return False


def _verify_ledger(root: Path, entries: list[dict[str, object]], role: str) -> None:
    """Verify a complete, ordered SHA-256/byte ledger against ``root``.

    Every declared entry must exist exactly, and no undeclared regular file may
    exist below ``root``.  Symlinks and ``__pycache__`` are forbidden.
    """
    declared = {(str(e["path"]), int(cast(int, e["bytes"])), str(e["sha256"])) for e in entries}
    seen: set[str] = set()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"payload symlink is forbidden: {path}")
        if "__pycache__" in path.parts:
            raise RuntimeError(f"payload bytecode cache is forbidden: {path}")
        if not path.is_file():
            continue
        seen.add(rel)
        digest = _sha256_file(path)
        triplet = (rel, path.stat().st_size, digest)
        if triplet not in declared:
            raise RuntimeError(f"ledger mismatch for {role} file: {rel}")
    declared_paths = {str(e["path"]) for e in entries}
    if seen != declared_paths:
        missing = declared_paths - seen
        extra = seen - declared_paths
        raise RuntimeError(
            f"ledger tree mismatch ({role}): missing={sorted(missing)} extra={sorted(extra)}"
        )


def _allowlisted_url(value: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(value)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"non-http(s) URL rejected: {value}")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS and not host.endswith(tuple(f".{h}" for h in _ALLOWED_HOSTS)):
        raise ValueError(f"host not on the network allowlist: {host}")


def describe(output: Path) -> int:
    """Download pinned source/model/base into private staging and emit a descriptor.

    This is an explicit operator action outside CI.  It never overwrites the
    canonical manifest and refuses arbitrary or mutable inputs.  It emits only
    a caller-selected candidate descriptor file.
    """
    if output.exists():
        raise RuntimeError("descriptor output already exists; choose a new file")
    _ensure_private_dir(_private_root())
    _force_hf_safety_flags()
    _require_optional_modules()

    from huggingface_hub import snapshot_download

    staging = _private_root() / "staging-describe"
    _ensure_private_dir(staging)
    hf_cache = staging / "hf-cache"
    _ensure_private_dir(hf_cache)
    os.environ["HF_HOME"] = str(hf_cache)

    _force_hf_safety_flags()

    checkpoint_dir = staging / "checkpoint"
    base_dir = staging / "base"
    source_dir = staging / "source"

    _ensure_private_dir(checkpoint_dir)
    _ensure_private_dir(base_dir)
    if source_dir.exists():
        if source_dir.is_symlink():
            raise RuntimeError("descriptor source staging may not be a symlink")
        shutil.rmtree(source_dir)
    _ensure_private_dir(source_dir)

    # Kev serving source via an immutable Git revision (GitHub archive bytes are
    # identity-adjacent but the descriptor ledger is computed over checkout
    # bytes; describe is the authorship step that records them).
    _clone_kev_source(source_dir)

    def download_or_resume(model_id: str, revision: str, target: Path) -> Path:
        try:
            _verify_hf_snapshot_identity(model_id, revision, target)
            return target
        except RuntimeError:
            snapshot_download(
                model_id,
                revision=revision,
                token=False,
                local_dir=str(target),
                cache_dir=str(hf_cache),
                max_workers=1,
            )
            _verify_hf_snapshot_identity(model_id, revision, target)
            return target

    checkpoint = download_or_resume(_KEV_MODEL_ID, _KEV_MODEL_REVISION, checkpoint_dir)
    base = download_or_resume(_QWEN_BASE_ID, _QWEN_BASE_REVISION, base_dir)
    _remove_hf_local_metadata(checkpoint_dir)
    _remove_hf_local_metadata(base_dir)

    lock_bytes = (source_dir / "kev" / "uv.lock").read_bytes()
    if hashlib.sha256(lock_bytes).hexdigest() != _KEV_UV_LOCK_SHA256:
        raise RuntimeError("upstream uv.lock digest does not match the reviewed binding")

    descriptor = {
        "schema_version": SCHEMA_VERSION,
        "reviewed_at": time.strftime("%Y-%m-%d"),
        "candidate_id": CANDIDATE_ID,
        "licenses": {
            "source": "Apache-2.0",
            "adapter": "Apache-2.0",
            "base": "Apache-2.0",
            "source_url": "https://github.com/jaredpalmer/kev/blob/main/LICENSE",
            "adapter_url": f"https://huggingface.co/{_KEV_MODEL_ID}/blob/{_KEV_MODEL_REVISION}/LICENSE",
            "base_url": f"https://huggingface.co/{_QWEN_BASE_ID}/blob/{_QWEN_BASE_REVISION}/LICENSE",
        },
        "source": {
            "repository": _KEV_SOURCE_URL,
            "revision": _KEV_SOURCE_REVISION,
            "python_constraint": ">=3.12,<3.14",
            "lock_sha256": _KEV_UV_LOCK_SHA256,
            "files": _ledger_for(source_dir, "source"),
        },
        "checkpoint": {
            "model_id": _KEV_MODEL_ID,
            "revision": _KEV_MODEL_REVISION,
            "files": _ledger_for(Path(checkpoint), "adapter"),
        },
        "base_model": {
            "model_id": _QWEN_BASE_ID,
            "revision": _QWEN_BASE_REVISION,
            "files": _ledger_for(Path(base), "base"),
        },
        "runtime": {
            "python": "3.12",
            "loopback_host": "127.0.0.1",
            "port_range": [49152, 65535],
            "backend": "mlx",
            "dtype": "bf16",
            "offline_environment_flags": ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"],
            "lock_digest": _KEV_UV_LOCK_SHA256,
        },
        "limits": {
            "max_total_bytes": 13_000_000_000,
            "max_source_file_count": 4096,
            "max_source_bytes": 500_000_000,
            "min_free_disk_bytes": 20_000_000_000,
            "min_physical_memory_bytes": 16 * 1024**3,
            "startup_timeout_seconds": 600,
            "request_timeout_seconds": 60,
        },
        "thresholds": {
            "max_latency_seconds": 60.0,
            "p95_latency_seconds_q1": 5.0,
            "p95_latency_seconds_q10": 15.0,
            "p95_latency_seconds_q50": 45.0,
            "max_repeat_probability_delta": 1e-6,
            "max_together_separate_probability_delta": 0.04,
            "near_tie_margin": 0.04,
            "max_peak_rss_gib": 22.0,
            "max_peak_physical_footprint_gib": 22.0,
            "preferred_swap_delta_gib": 2.0,
            "degraded_swap_delta_gib": 8.0,
            "max_swap_delta_gib": 8.0,
            "cold_readiness_timeout_seconds": 600,
        },
        "allowed_readiness_evidence": [
            "pinned_license_reviewed_acquisition",
            "local_systems_report",
        ],
    }
    # Validate before emitting so the caller never receives a malformed authority.
    AcquisitionDescriptor.model_validate(descriptor)
    _atomic_json_write(output, descriptor)
    return 0


def _clone_kev_source(dest: Path) -> None:
    """Acquire the pinned Kev source tree at the exact immutable revision."""
    subprocess.run(
        [
            "git",
            "-c",
            "credential.helper=",
            "clone",
            "--no-checkout",
            _KEV_SOURCE_URL,
            str(dest / "kev"),
        ],
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    subprocess.run(
        ["git", "-C", str(dest / "kev"), "checkout", _KEV_SOURCE_REVISION],
        check=True,
    )
    resolved = subprocess.run(
        ["git", "-C", str(dest / "kev"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != _KEV_SOURCE_REVISION:
        raise RuntimeError("Git resolved an unexpected Kev source revision")
    _remove_git_metadata(dest / "kev")


def _ledger_for(root: Path, role: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "role": role,
            }
        )
    entries.sort(key=lambda entry: str(entry["path"]))
    return entries


def prepare(descriptor_path: Path) -> int:
    """Acquire and verify the exact ledger bytes, then build the locked
    Python 3.12 environment from the verified source.

    Any nonzero subprocess result, missing/extra artifact, mutable revision,
    unsafe permission or digest mismatch fails closed before service launch.
    """
    descriptor = _validate_descriptor_bytes(descriptor_path)
    if descriptor.checkpoint.model_id != _KEV_MODEL_ID:
        raise ValueError("checkpoint model id does not match the reviewed binding")
    if descriptor.checkpoint.revision != _KEV_MODEL_REVISION:
        raise ValueError("checkpoint revision does not match the reviewed binding")
    if descriptor.base_model.model_id != _QWEN_BASE_ID:
        raise ValueError("base model id does not match the reviewed binding")
    if descriptor.base_model.revision != _QWEN_BASE_REVISION:
        raise ValueError("base model revision does not match the reviewed binding")
    root = _descriptor_root(descriptor_path)
    _preflight_capacity(descriptor, root)
    _force_hf_safety_flags()
    _require_optional_modules()
    if not (descriptor.source.revision and _FULL_GIT_REV.fullmatch(descriptor.source.revision)):
        raise ValueError("source revision is not immutable")
    if descriptor.source.revision != _KEV_SOURCE_REVISION:
        raise ValueError("source revision does not match the reviewed binding")
    if descriptor.runtime.python != "3.12":
        raise ValueError("runtime python must be 3.12")
    if descriptor.runtime.backend != "mlx" or descriptor.runtime.dtype != "bf16":
        raise ValueError("runtime backend/dtype must be mlx bf16")

    from huggingface_hub import snapshot_download

    _allowlisted_url(descriptor.source.repository)
    for license_url in (
        descriptor.licenses.source_url,
        descriptor.licenses.adapter_url,
        descriptor.licenses.base_url,
    ):
        _allowlisted_url(license_url)

    payload = root / "payload"
    hf_home = root / "hf-home"
    _ensure_private_dir(hf_home)

    _force_hf_safety_flags()

    def verify_payload() -> None:
        _verify_ledger(
            payload / "source", [e.model_dump() for e in descriptor.source.files], "source"
        )
        _verify_ledger(
            payload / "checkpoint",
            [e.model_dump() for e in descriptor.checkpoint.files],
            "adapter",
        )
        _verify_ledger(
            payload / "base", [e.model_dump() for e in descriptor.base_model.files], "base"
        )

    try:
        verify_payload()
    except (FileNotFoundError, RuntimeError):
        if payload.exists():
            if payload.is_symlink():
                raise RuntimeError("payload root may not be a symlink") from None
            shutil.rmtree(payload)
        _ensure_private_dir(payload)
        _ensure_private_dir(payload / "source")
        _ensure_private_dir(payload / "checkpoint")
        _ensure_private_dir(payload / "base")

        # Download source into the payload tree from the reviewed Git revision.
        _clone_kev_source(payload / "source")

        # Download snapshots only into the descriptor-scoped private cache.
        snapshot_download(
            descriptor.checkpoint.model_id,
            revision=descriptor.checkpoint.revision,
            token=False,
            local_dir=str(payload / "checkpoint"),
            cache_dir=str(hf_home),
            max_workers=1,
        )
        snapshot_download(
            descriptor.base_model.model_id,
            revision=descriptor.base_model.revision,
            token=False,
            local_dir=str(payload / "base"),
            cache_dir=str(hf_home),
            max_workers=1,
        )
        _verify_hf_snapshot_identity(
            descriptor.checkpoint.model_id,
            descriptor.checkpoint.revision,
            payload / "checkpoint",
        )
        _verify_hf_snapshot_identity(
            descriptor.base_model.model_id,
            descriptor.base_model.revision,
            payload / "base",
        )
        _remove_hf_local_metadata(payload / "checkpoint")
        _remove_hf_local_metadata(payload / "base")
        verify_payload()

    # Locked Python 3.12 environment from the verified source only.
    environment = root / "environment"
    _ensure_private_dir(environment)
    uv_cache = root / "uv-cache"
    _ensure_private_dir(uv_cache)
    prepare_env = dict(os.environ)
    prepare_env.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(environment),
            "UV_CACHE_DIR": str(uv_cache),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv executable is required for the locked environment")
    result = subprocess.run(
        [
            uv_executable,
            "sync",
            "--locked",
            "--extra",
            "serve",
            "--python",
            "3.12",
            "--no-install-project",
            "--no-dev",
        ],
        cwd=payload / "source" / "kev",
        env=prepare_env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv sync failed with exit code {result.returncode}")
    _build_hf_runtime_view(root, descriptor)
    # Post-step verification: the payload must remain byte-for-byte ledger-clean.
    _verify_ledger(payload / "source", [e.model_dump() for e in descriptor.source.files], "source")
    _verify_ledger(
        payload / "checkpoint", [e.model_dump() for e in descriptor.checkpoint.files], "adapter"
    )
    _verify_ledger(payload / "base", [e.model_dump() for e in descriptor.base_model.files], "base")
    _strip_payload_cache_artifacts(payload)
    return 0


def _readonly_payload(payload: Path) -> None:
    """Make the payload read-only (dirs read/execute, files read) before import."""
    for path in payload.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"payload symlink is forbidden: {path}")
    for directory in sorted((p for p in payload.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(directory, 0o500)
    for file in payload.rglob("*"):
        if file.is_file():
            os.chmod(file, 0o400)
    os.chmod(payload, 0o500)


def _restore_payload_writable(payload: Path) -> None:
    """Restore owner write permission before removal, without following symlinks."""
    for path in payload.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            os.chmod(path, 0o600)
    for directory in sorted((p for p in payload.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(directory, 0o700)
    os.chmod(payload, 0o700)


def _select_port(descriptor: AcquisitionDescriptor) -> int:
    lower, upper = descriptor.runtime.port_range
    for port in range(lower, upper + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.05)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no unused port in the fixed research range")


def _child_environment(
    descriptor: AcquisitionDescriptor, root: Path, run_id: str, bearer: str
) -> dict[str, str]:
    """Build the child service environment from an allowlist, not inheritance."""
    derived = root / "derived" / run_id
    paths = {
        "derived": derived,
        "home": derived / "home",
        "tmp": derived / "tmp",
        "cache": derived / "cache",
        "pycache": derived / "pycache",
        "uv-cache": derived / "uv-cache",
        "hf-home": root / "hf-home",
    }
    for path in paths.values():
        _ensure_private_dir(path)
    shim = paths["derived"] / "runtime-shim"
    _ensure_private_dir(shim)
    base_path = (root / "payload" / "base").resolve()
    shim_source = f'''"""Exact offline snapshot resolver generated by Saracura Phase 5B."""
import huggingface_hub

_MODEL_ID = {descriptor.base_model.model_id!r}
_REVISION = {descriptor.base_model.revision!r}
_LOCAL_PATH = {str(base_path)!r}

def _offline_snapshot_download(repo_id=None, *args, revision=None, **kwargs):
    resolved_repo = repo_id if repo_id is not None else kwargs.get("repo_id")
    resolved_revision = revision if revision is not None else kwargs.get("revision")
    if resolved_repo == _MODEL_ID and resolved_revision == _REVISION:
        return _LOCAL_PATH
    raise RuntimeError("offline snapshot request does not match the verified base model")

huggingface_hub.snapshot_download = _offline_snapshot_download
'''
    shim_path = shim / "sitecustomize.py"
    shim_path.write_text(shim_source, encoding="utf-8")
    os.chmod(shim_path, 0o600)
    return {
        "KEV_BACKEND": "mlx",
        "KEV_DTYPE": "bf16",
        "KEV_LORA_SCALE": "1",
        "KEV_MERGE": "1",
        "KEV_DATE_FACTS": "0",
        "KEV_CUDA_GRAPHS": "0",
        "KEV_FUSED": "0",
        "KEV_PREFIX_CACHE": "4",
        "KEV_PREFIX_MIN_TOKENS": "0",
        "KEV_PREFIX_MAX_TOKENS": "65536",
        "KEV_API_KEY": bearer,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HOME": str(paths["hf-home"]),
        "UV_CACHE_DIR": str(paths["uv-cache"]),
        "PYTHONPATH": os.pathsep.join((str(shim), str(root / "payload" / "source" / "kev"))),
        "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HTTP_PROXY": "http://127.0.0.1:0",
        "HTTPS_PROXY": "http://127.0.0.1:0",
        "http_proxy": "http://127.0.0.1:0",
        "https_proxy": "http://127.0.0.1:0",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "HOME": str(paths["home"]),
        "TMPDIR": str(paths["tmp"]),
        "XDG_CACHE_HOME": str(paths["cache"]),
    }


def _reject_unrecognized_kev_env(inherited: dict[str, str]) -> None:
    known = {
        "KEV_BACKEND",
        "KEV_DTYPE",
        "KEV_LORA_SCALE",
        "KEV_MERGE",
        "KEV_DATE_FACTS",
        "KEV_CUDA_GRAPHS",
        "KEV_FUSED",
        "KEV_PREFIX_CACHE",
        "KEV_PREFIX_MIN_TOKENS",
        "KEV_PREFIX_MAX_TOKENS",
        "KEV_API_KEY",
    }
    for key in inherited:
        if key.startswith("KEV_") and key not in known:
            raise RuntimeError(f"unrecognized inherited KEV_* variable: {key}")


def _request_bearer() -> str:
    return secrets.token_urlsafe(32)


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    _ensure_private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _process_identity_matches(pid: int, expected_create_time: float) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        return bool(process.is_running() and process.create_time() == expected_create_time)
    except psutil.NoSuchProcess:
        return False


def _parse_footprint_bytes(output: str) -> int | None:
    match = re.search(r"Physical footprint:\s*([0-9.]+)\s*([KMG])", output, re.I)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return int(float(match.group(1)) * scale)


def _parse_pageouts(output: str) -> int | None:
    match = re.search(r"^Pages? out:\s*(\d+)\.", output, re.I | re.M)
    return int(match.group(1)) if match else None


def _parse_swap_used_bytes(output: str) -> int | None:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output, re.I)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return int(float(match.group(1)) * scale)


def _host_measurements() -> dict[str, int | str | None]:
    if os.uname().sysname != "Darwin":
        return {"pageouts": None, "swap_used_bytes": None, "memory_pressure": None}
    vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)
    pressure = subprocess.run(
        ["memory_pressure", "-Q"], capture_output=True, text=True, check=False
    )
    pressure_text = f"{pressure.stdout}\n{pressure.stderr}".lower()
    state = next((word for word in ("critical", "warn", "normal") if word in pressure_text), None)
    if state is None:
        free_match = re.search(r"memory free percentage:\s*(\d+)%", pressure_text)
        if free_match:
            free_percent = int(free_match.group(1))
            state = "critical" if free_percent <= 5 else "warn" if free_percent <= 10 else "normal"
    return {
        "pageouts": _parse_pageouts(vm_stat.stdout) if vm_stat.returncode == 0 else None,
        "swap_used_bytes": _parse_swap_used_bytes(swap.stdout) if swap.returncode == 0 else None,
        "memory_pressure": state,
    }


def _service_measurements(pid: int) -> dict[str, int | None]:
    import psutil

    try:
        process = psutil.Process(pid)
        rss = int(process.memory_info().rss)
    except psutil.NoSuchProcess:
        return {"rss_bytes": None, "physical_footprint_bytes": None}
    if os.uname().sysname != "Darwin":
        return {"rss_bytes": rss, "physical_footprint_bytes": None}
    result = subprocess.run(
        ["vmmap", "-summary", str(pid)], capture_output=True, text=True, check=False
    )
    footprint = _parse_footprint_bytes(result.stdout) if result.returncode == 0 else None
    return {"rss_bytes": rss, "physical_footprint_bytes": footprint}


def _write_cold_capacity_failure(
    descriptor_path: Path,
    descriptor: AcquisitionDescriptor,
    root: Path,
    run_id: str,
    started_at: float,
    baseline: Mapping[str, object],
    pid: int,
    reason: str,
) -> Path:
    from benchmarks.phase5_candidate.fixture import matrix as fixture_matrix
    from benchmarks.phase5_candidate.report import CapacityFailure, SanitizedReport

    host_after = _host_measurements()
    service = _service_measurements(pid)
    swap_before = baseline.get("swap_used_bytes")
    swap_after = host_after.get("swap_used_bytes")
    pageouts_before = baseline.get("pageouts")
    pageouts_after = host_after.get("pageouts")
    swap_delta = (
        max(0, int(cast(int, swap_after)) - int(cast(int, swap_before))) / 1024**3
        if swap_before is not None and swap_after is not None
        else None
    )
    pageout_delta = (
        max(0, int(cast(int, pageouts_after)) - int(cast(int, pageouts_before)))
        if pageouts_before is not None and pageouts_after is not None
        else None
    )
    fixture = fixture_matrix()
    report = SanitizedReport.model_validate(
        {
            "schema_version": "phase5b-report.v1",
            "saracura_commit": _repo_commit(),
            "acquisition_descriptor_sha256": hashlib.sha256(
                descriptor_path.read_bytes()
            ).hexdigest(),
            "threshold_digest": hashlib.sha256(
                json.dumps(descriptor.thresholds.model_dump(), sort_keys=True).encode()
            ).hexdigest(),
            "source_revision": descriptor.source.revision,
            "model_revision": descriptor.checkpoint.revision,
            "base_model_revision": descriptor.base_model.revision,
            "fixture_digest": hashlib.sha256(
                json.dumps(fixture, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "os": os.uname().sysname,
            "architecture": os.uname().machine,
            "chip_family": _chip_family(),
            "physical_memory_bucket": _memory_bucket(),
            "backend": "mlx",
            "dtype": "bfloat16",
            "model_bytes": sum(entry.bytes for entry in descriptor.checkpoint.files)
            + sum(entry.bytes for entry in descriptor.base_model.files),
            "cold_load_seconds": time.monotonic() - started_at,
            "request_p50_ms": 0.0,
            "request_p95_ms": 0.0,
            "per_decision_p50_ms": 0.0,
            "per_decision_p95_ms": 0.0,
            "decisions_per_second": 0.0,
            "peak_service_rss_gib": (
                service["rss_bytes"] / 1024**3 if service["rss_bytes"] is not None else None
            ),
            "peak_physical_footprint_gib": (
                service["physical_footprint_bytes"] / 1024**3
                if service["physical_footprint_bytes"] is not None
                else None
            ),
            "swap_used_delta_gib": swap_delta,
            "pageout_delta": pageout_delta,
            "memory_pressure_state": str(host_after.get("memory_pressure") or "unavailable"),
            "invalid_output_count": 0,
            "repeat_stability_max_delta": 0.0,
            "isolation_delta_max": 0.0,
            "option_order_behavior": "not_run_cold_capacity_failure",
            "capacity_failures": [
                CapacityFailure(category="cold_load_memory", detail=reason).model_dump()
            ],
            "disposition": "reject_local",
            "per_q_p95_ms": {},
            "max_request_ms": 0.0,
            "selected_choice_stability": False,
            "repeat_probability_delta": 0.0,
            "together_separate_probability_delta": 0.0,
            "unauthorized_accepted_count": 0,
        }
    )
    artifact = root / "derived" / run_id / "cold-capacity-failure.json"
    _atomic_json_write(artifact, report.model_dump(mode="json"))
    return artifact


class _EvaluationMonitor:
    def __init__(self, pid: int, port: int, create_time: float) -> None:
        self.pid = pid
        self.port = port
        self.create_time = create_time
        self.stop_event = threading.Event()
        self.failure: str | None = None
        self.peak_rss_bytes: int | None = None
        self.peak_footprint_bytes: int | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        last_sample = 0.0
        while not self.stop_event.wait(0.1):
            if not _process_identity_matches(self.pid, self.create_time):
                self.failure = "recorded service process identity changed during evaluation"
                return
            if not _loops_only(self.pid) or not _pid_owns_port(self.pid, self.port):
                self.failure = "service socket ownership or loopback security check failed"
                return
            now = time.monotonic()
            if now - last_sample >= 0.5:
                sample = _service_measurements(self.pid)
                rss = sample["rss_bytes"]
                footprint = sample["physical_footprint_bytes"]
                if rss is not None:
                    self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
                if footprint is not None:
                    self.peak_footprint_bytes = max(self.peak_footprint_bytes or 0, footprint)
                last_sample = now

    def start(self) -> None:
        self.thread.start()

    def check(self) -> None:
        if self.failure is not None:
            raise RuntimeError(self.failure)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.check()


def serve(descriptor_path: Path) -> int:
    """Launch the pinned Kev service out of process under a locked Python 3.12
    environment, PID-owned loopback, offline flags, per-run bearer and no
    fallback.  Confirms ``/v1/models`` identity before returning.
    """
    descriptor = _validate_descriptor_bytes(descriptor_path)
    _require_optional_modules()
    _reject_unrecognized_kev_env(dict(os.environ))

    root = _descriptor_root(descriptor_path)
    payload = root / "payload"
    environment = root / "environment"
    if not (payload / "source" / "kev").exists():
        raise RuntimeError("verified payload missing; run prepare first")
    if not environment.exists():
        raise RuntimeError("locked environment missing; run prepare first")

    _restore_payload_writable(payload)
    _readonly_payload(payload)
    _strip_payload_cache_artifacts(payload)

    port = _select_port(descriptor)
    run_id = secrets.token_hex(8)
    bearer = _request_bearer()
    host_baseline = _host_measurements()
    env = _child_environment(descriptor, root, run_id, bearer)
    run_dir = root / "derived" / run_id
    stdout_path = run_dir / "service.stdout.log"
    stderr_path = run_dir / "service.stderr.log"
    run_arg = str(payload / "checkpoint")

    python = environment / "bin" / "python"
    if not python.exists():
        python = environment / "bin" / "python3"
    if not python.exists():
        raise RuntimeError("locked python interpreter missing from the environment")

    started_at = time.monotonic()
    with stdout_path.open("ab") as stdout_log, stderr_path.open("ab") as stderr_log:
        os.chmod(stdout_path, 0o600)
        os.chmod(stderr_path, 0o600)
        process = subprocess.Popen(
            [
                str(python),
                "-m",
                "kev.serve",
                "--run",
                run_arg,
                "--fallback",
                run_arg,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=payload / "source" / "kev",
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
        )

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + descriptor.limits.startup_timeout_seconds
    from benchmarks.phase5_candidate.client import fetch_models, post_systemone_without_bearer

    models = None
    last_capacity_check = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _restore_payload_writable(payload)
            if (
                process.returncode is not None
                and process.returncode < 0
                and stderr_path.stat().st_size == 0
            ):
                artifact = _write_cold_capacity_failure(
                    descriptor_path,
                    descriptor,
                    root,
                    run_id,
                    started_at,
                    host_baseline,
                    process.pid,
                    "service process was terminated during memory-intensive cold load",
                )
                raise RuntimeError(f"cold load exceeded local capacity; report: {artifact}")
            raise RuntimeError(
                f"service exited early with code {process.returncode}; inspect {stderr_path}"
            )
        now = time.monotonic()
        if now - last_capacity_check >= 1.0:
            current_host = _host_measurements()
            swap_before = host_baseline.get("swap_used_bytes")
            swap_after = current_host.get("swap_used_bytes")
            swap_delta_gib = (
                max(0, int(cast(int, swap_after)) - int(cast(int, swap_before))) / 1024**3
                if swap_before is not None and swap_after is not None
                else 0.0
            )
            reason = None
            if current_host.get("memory_pressure") == "critical":
                reason = "critical memory pressure during cold load"
            elif swap_delta_gib > descriptor.thresholds.max_swap_delta_gib:
                reason = "swap growth exceeded the frozen cold-load limit"
            if reason is not None:
                try:
                    artifact = _write_cold_capacity_failure(
                        descriptor_path,
                        descriptor,
                        root,
                        run_id,
                        started_at,
                        host_baseline,
                        process.pid,
                        reason,
                    )
                finally:
                    process.terminate()
                    process.wait(timeout=10)
                    _restore_payload_writable(payload)
                raise RuntimeError(f"cold load exceeded local capacity; report: {artifact}")
            last_capacity_check = now
        if not _loops_only(process.pid):
            if process.poll() is not None:
                continue
            process.terminate()
            _restore_payload_writable(payload)
            raise RuntimeError("service opened a non-loopback listener")
        if not _pid_owns_port(process.pid, port):
            time.sleep(0.1)
            continue
        try:
            if post_systemone_without_bearer(base, payload={"questions": []}, timeout=5.0) != 401:
                process.terminate()
                _restore_payload_writable(payload)
                raise RuntimeError("service accepted a request without a bearer token")
            models = fetch_models(base, bearer=bearer, timeout=5.0)
            break
        except Exception:
            time.sleep(0.1)
    if models is None:
        process.terminate()
        _restore_payload_writable(payload)
        raise RuntimeError("service did not become ready within the startup timeout")

    try:
        _validate_models_identity(
            models,
            expected_checkpoint=payload / "checkpoint",
            expected_base_id=_QWEN_BASE_ID,
        )
    except ValueError:
        process.terminate()
        _restore_payload_writable(payload)
        raise

    state = {
        "pid": process.pid,
        "port": port,
        "bearer": bearer,
        "run_id": run_id,
        "ready": True,
        "create_time": __import__("psutil").Process(process.pid).create_time(),
        "cold_readiness_seconds": time.monotonic() - started_at,
        "host_baseline": host_baseline,
    }
    try:
        _atomic_json_write(root / "derived" / run_id / "serve-state.json", state)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        _restore_payload_writable(payload)
        raise
    return 0


def evaluate(descriptor_path: Path) -> int:
    """Evaluate only public fixtures, then stop the recorded service on all paths."""
    _validate_descriptor_bytes(descriptor_path)
    _require_optional_modules()
    root = _descriptor_root(descriptor_path)
    state_path = next((root / "derived").glob("*/serve-state.json"), None)
    if state_path is None:
        raise RuntimeError("no served process found; run serve first")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pid = int(state["pid"])
    port = int(state["port"])
    create_time = float(state["create_time"])
    if not _process_identity_matches(pid, create_time):
        raise RuntimeError("recorded service PID no longer identifies the launched process")
    if not _loops_only(pid) or not _pid_owns_port(pid, port):
        raise RuntimeError("recorded service PID does not own the loopback port")
    payload = root / "payload"
    monitor = _EvaluationMonitor(pid, port, create_time)
    monitor.start()
    try:
        return _evaluate_live(descriptor_path, state, monitor)
    finally:
        monitor.stop_event.set()
        monitor.thread.join(timeout=2)
        try:
            _stop_recorded_service(pid, create_time)
        finally:
            if payload.exists():
                _restore_payload_writable(payload)


def _stop_recorded_service(pid: int, create_time: float) -> None:
    import psutil

    if not _process_identity_matches(pid, create_time):
        return
    process = psutil.Process(pid)
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _evaluate_live(
    descriptor_path: Path, state: dict[str, object], monitor: _EvaluationMonitor
) -> int:
    descriptor = _validate_descriptor_bytes(descriptor_path)
    root = _descriptor_root(descriptor_path)
    base = f"http://127.0.0.1:{state['port']}"
    bearer = str(state["bearer"])
    payload_root = root / "payload"
    checkpoint = payload_root / "checkpoint"
    from benchmarks.phase5_candidate.client import (
        fetch_models,
        oversize_probe,
        post_systemone,
        post_systemone_permute,
        post_systemone_separate,
        post_systemone_without_bearer,
    )
    from benchmarks.phase5_candidate.fixture import matrix as fixture_matrix

    unauthorized_status = post_systemone_without_bearer(
        base, payload={"questions": []}, timeout=5.0
    )
    unauthorized_accepted = int(unauthorized_status != 401)
    identities = fetch_models(base, bearer=bearer, timeout=5.0)
    _validate_models_identity(
        identities,
        expected_checkpoint=checkpoint,
        expected_base_id=_QWEN_BASE_ID,
    )

    fixture = fixture_matrix()
    request_latencies: list[float] = []
    per_q_latencies: dict[int, list[float]] = {1: [], 10: [], 50: []}
    invalid_outputs = 0
    completed = True
    repeat_delta = 0.0
    isolation_delta = 0.0
    selected_stable = True
    option_behavior: list[str] = []
    request_over_limit = False

    def send(
        method: Callable[..., tuple[int, dict[str, Any]]],
        request_payload: dict[str, object],
        question_options: dict[str, set[str]],
    ) -> tuple[SystemOneResponse | None, float]:
        nonlocal invalid_outputs, request_over_limit
        monitor.check()
        started = time.monotonic()
        try:
            status, response_payload = method(
                base,
                bearer=bearer,
                payload=request_payload,
                timeout=descriptor.limits.request_timeout_seconds,
            )
            elapsed = time.monotonic() - started
            if status != 200:
                invalid_outputs += 1
                return None, elapsed * 1000.0
            validated = _validate_systemone_result(
                response_payload,
                question_options=question_options,
                expected_run=checkpoint,
            )
            return validated, elapsed * 1000.0
        except Exception:
            elapsed = time.monotonic() - started
            invalid_outputs += 1
            return None, elapsed * 1000.0
        finally:
            if time.monotonic() - started > descriptor.thresholds.max_latency_seconds:
                request_over_limit = True
            monitor.check()

    for locale, workloads in fixture.items():
        for workload, request in workloads.items():
            q_count = len(request.questions)
            options = {q.id: {criterion.id for criterion in q.criteria} for q in request.questions}
            cached_payload: dict[str, object] = {
                "state": request.state,
                "questions": [question.model_dump() for question in request.questions],
            }
            for _ in range(3):
                _warm, _elapsed = send(post_systemone, cached_payload, options)
            new_state_results: list[SystemOneResponse] = []
            cached_results: list[SystemOneResponse] = []
            for repeat in range(20):
                new_payload = {
                    **cached_payload,
                    "state": {**request.state, "repeat_nonce": repeat},
                }
                result, elapsed_ms = send(post_systemone, new_payload, options)
                request_latencies.append(elapsed_ms)
                per_q_latencies[q_count].append(elapsed_ms)
                if result is not None:
                    new_state_results.append(result)
                result, elapsed_ms = send(post_systemone, cached_payload, options)
                request_latencies.append(elapsed_ms)
                per_q_latencies[q_count].append(elapsed_ms)
                if result is not None:
                    cached_results.append(result)

            if len(new_state_results) != 20 or len(cached_results) != 20:
                completed = False
            if cached_results:
                reference = cached_results[0].answers
                for result in cached_results[1:4]:
                    delta, stable, _flips = _compare_choice_answers(
                        reference,
                        result.answers,
                        near_tie_margin=descriptor.thresholds.near_tie_margin,
                    )
                    repeat_delta = max(repeat_delta, delta)
                    selected_stable = selected_stable and stable
            else:
                completed = False

            together, _ = send(post_systemone, cached_payload, options)
            separate, _ = send(post_systemone_separate, cached_payload, options)
            if together is not None and separate is not None:
                delta, stable, flips = _compare_choice_answers(
                    together.answers,
                    separate.answers,
                    near_tie_margin=descriptor.thresholds.near_tie_margin,
                )
                isolation_delta = max(isolation_delta, delta)
                selected_stable = selected_stable and stable
                option_behavior.append(f"{locale}-{workload}:near_tie_flips={len(flips)}")
            else:
                completed = False
                option_behavior.append(f"{locale}-{workload}:unavailable")

            permuted, _ = send(post_systemone_permute, cached_payload, options)
            option_behavior.append(
                f"{locale}-{workload}:permute={'observed' if permuted is not None else 'invalid'}"
            )

    invalid_probe_status, _ = post_systemone(
        base,
        bearer=bearer,
        payload={"questions": [{"id": "invalid", "type": "unknown"}]},
        timeout=descriptor.limits.request_timeout_seconds,
    )
    invalid_probe_ok = invalid_probe_status in (400, 422)
    branch_status = oversize_probe(
        base, bearer=bearer, timeout=descriptor.limits.request_timeout_seconds, kind="branch"
    )
    state_status = oversize_probe(
        base, bearer=bearer, timeout=descriptor.limits.request_timeout_seconds, kind="state"
    )
    branch_ok = branch_status == 422
    state_truncation = state_status == 200
    _verify_ledger(
        payload_root / "source", [entry.model_dump() for entry in descriptor.source.files], "source"
    )
    _verify_ledger(
        checkpoint, [entry.model_dump() for entry in descriptor.checkpoint.files], "adapter"
    )
    _verify_ledger(
        payload_root / "base", [entry.model_dump() for entry in descriptor.base_model.files], "base"
    )

    host_after = _host_measurements()
    baseline = state.get("host_baseline", {})
    current = _service_measurements(int(cast(int | str, state["pid"])))
    peak_rss = monitor.peak_rss_bytes or current["rss_bytes"]
    peak_footprint = monitor.peak_footprint_bytes or current["physical_footprint_bytes"]
    swap_before = baseline.get("swap_used_bytes") if isinstance(baseline, dict) else None
    swap_after = host_after.get("swap_used_bytes")
    pageouts_before = baseline.get("pageouts") if isinstance(baseline, dict) else None
    pageouts_after = host_after.get("pageouts")
    pressure = host_after.get("memory_pressure")
    metrics_available = all(
        value is not None
        for value in (
            peak_rss,
            peak_footprint,
            swap_before,
            swap_after,
            pageouts_before,
            pageouts_after,
            pressure,
        )
    )
    sorted_latencies = sorted(request_latencies)
    q95_ms = {
        str(question_count): _percentile(sorted(per_q_latencies[question_count]), 0.95)
        for question_count in (1, 10, 50)
    }
    request_p50 = _percentile(sorted_latencies, 0.50)
    request_p95 = _percentile(sorted_latencies, 0.95)
    per_decision = sorted(
        elapsed / q_count for q_count, samples in per_q_latencies.items() for elapsed in samples
    )
    peak_rss_gib = int(peak_rss) / 1024**3 if peak_rss is not None else None
    peak_footprint_gib = int(peak_footprint) / 1024**3 if peak_footprint is not None else None
    swap_delta_gib = (
        (int(swap_after) - int(swap_before)) / 1024**3
        if swap_before is not None and swap_after is not None
        else None
    )
    pageout_delta = (
        max(0, int(pageouts_after) - int(pageouts_before))
        if pageouts_before is not None and pageouts_after is not None
        else None
    )
    unauthorized_accepted = unauthorized_accepted
    if not metrics_available:
        disposition = "blocked_upstream"
    elif state_truncation:
        disposition = "reject_local"
    else:
        assert peak_rss_gib is not None
        assert peak_footprint_gib is not None
        assert swap_delta_gib is not None
        disposition = derive_disposition(
            functional_pass=invalid_outputs == 0
            and repeat_delta <= descriptor.thresholds.max_repeat_probability_delta,
            identity_pass=True,
            security_pass=unauthorized_accepted == 0 and branch_ok and invalid_probe_ok,
            cold_readiness_seconds=float(cast(float | int | str, state["cold_readiness_seconds"])),
            latency_p95_pass={
                q_count: _percentile(sorted(per_q_latencies[q_count]), 0.95) / 1000.0
                <= getattr(descriptor.thresholds, f"p95_latency_seconds_q{q_count}")
                for q_count in (1, 10, 50)
            },
            any_request_over_60s=request_over_limit,
            peak_rss_gib=float(peak_rss_gib),
            peak_footprint_gib=float(peak_footprint_gib),
            swap_delta_gib=float(swap_delta_gib),
            memory_pressure_critical=pressure == "critical",
            invalid_outputs=invalid_outputs,
            unauthorized_accepted=unauthorized_accepted,
            oversize_rejected_correctly=branch_ok,
            invalid_probe_rejected_correctly=invalid_probe_ok,
            matrix_completed=completed,
            thresholds=descriptor.thresholds,
        )

    from benchmarks.phase5_candidate.report import CapacityFailure, SanitizedReport

    capacity_failures = (
        ()
        if metrics_available
        else (CapacityFailure(category="measurement", detail="required host metric unavailable"),)
    )
    report = SanitizedReport.model_validate(
        {
            "schema_version": "phase5b-report.v1",
            "saracura_commit": _repo_commit(),
            "acquisition_descriptor_sha256": hashlib.sha256(
                descriptor_path.read_bytes()
            ).hexdigest(),
            "threshold_digest": hashlib.sha256(
                json.dumps(descriptor.thresholds.model_dump(), sort_keys=True).encode()
            ).hexdigest(),
            "source_revision": descriptor.source.revision,
            "model_revision": descriptor.checkpoint.revision,
            "base_model_revision": descriptor.base_model.revision,
            "fixture_digest": hashlib.sha256(
                json.dumps(fixture, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "os": os.uname().sysname,
            "architecture": os.uname().machine,
            "chip_family": _chip_family(),
            "physical_memory_bucket": _memory_bucket(),
            "backend": "mlx",
            "dtype": "bfloat16",
            "model_bytes": sum(entry.bytes for entry in descriptor.checkpoint.files),
            "cold_load_seconds": state.get("cold_readiness_seconds"),
            "request_p50_ms": request_p50,
            "request_p95_ms": request_p95,
            "per_decision_p50_ms": _percentile(per_decision, 0.50),
            "per_decision_p95_ms": _percentile(per_decision, 0.95),
            "decisions_per_second": 1000.0 / request_p50 if request_p50 > 0 else 0.0,
            "peak_service_rss_gib": peak_rss_gib,
            "peak_physical_footprint_gib": peak_footprint_gib,
            "swap_used_delta_gib": swap_delta_gib,
            "pageout_delta": pageout_delta,
            "memory_pressure_state": str(pressure or "unavailable"),
            "invalid_output_count": invalid_outputs,
            "repeat_stability_max_delta": repeat_delta,
            "isolation_delta_max": isolation_delta,
            "option_order_behavior": ";".join(option_behavior),
            "capacity_failures": [failure.model_dump() for failure in capacity_failures],
            "disposition": disposition,
            "raw_confidence_concentration": None,
            "state_truncation_contract_failure": state_truncation,
            "per_q_p95_ms": q95_ms,
            "max_request_ms": max(request_latencies, default=0.0),
            "selected_choice_stability": selected_stable,
            "repeat_probability_delta": repeat_delta,
            "together_separate_probability_delta": isolation_delta,
            "unauthorized_accepted_count": unauthorized_accepted,
        }
    )
    artifact = root / "derived" / str(state["run_id"]) / "evaluate-report.json"
    _atomic_json_write(artifact, report.model_dump(mode="json"))
    return 0


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _repo_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _chip_family() -> str:
    uname = os.uname()
    if uname.machine == "arm64" and uname.sysname == "Darwin":
        return "apple_silicon"
    return uname.machine


def _memory_bucket() -> str:
    from benchmarks.phase5_candidate.report import memory_bucket

    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return memory_bucket(total)


def sanitized_disposition(
    *,
    functional_pass: bool,
    identity_pass: bool,
    security_pass: bool,
    cold_readiness_seconds: float,
    latency_p95_pass: dict[int, bool],
    any_request_over_60s: bool,
    peak_rss_gib: float,
    peak_footprint_gib: float,
    swap_delta_gib: float,
    memory_pressure_critical: bool | None,
    invalid_outputs: int,
    unauthorized_accepted: int,
    oversize_rejected_correctly: bool,
    invalid_probe_rejected_correctly: bool,
    matrix_completed: bool,
    thresholds: ThresholdsBlock,
) -> str:
    return derive_disposition(
        functional_pass=functional_pass,
        identity_pass=identity_pass,
        security_pass=security_pass,
        cold_readiness_seconds=cold_readiness_seconds,
        latency_p95_pass=latency_p95_pass,
        any_request_over_60s=any_request_over_60s,
        peak_rss_gib=peak_rss_gib,
        peak_footprint_gib=peak_footprint_gib,
        swap_delta_gib=swap_delta_gib,
        memory_pressure_critical=memory_pressure_critical,
        invalid_outputs=invalid_outputs,
        unauthorized_accepted=unauthorized_accepted,
        oversize_rejected_correctly=oversize_rejected_correctly,
        invalid_probe_rejected_correctly=invalid_probe_rejected_correctly,
        matrix_completed=matrix_completed,
        thresholds=thresholds,
    )
