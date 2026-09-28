"""Offline-testable managed CUDA protocol and Linux evidence collectors.

This module owns the client side only. All host readers, clocks, transport and
subprocess calls are injectable so the complete protocol is testable offline.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.phase5_candidate.fixture import matrix
from benchmarks.phase5_managed_cuda.wire import convert_fixture_to_wire

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8182
GIB = 1024**3


class EvidenceBlocked(RuntimeError):
    """Mandatory evidence is absent, ambiguous, or preemption-like."""


class CandidateRejected(RuntimeError):
    """Candidate returned positive contract-invalid evidence."""


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: Mapping[str, str]
    body: bytes
    elapsed_ms: float


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def validate_loopback_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username
        or parsed.password
    ):
        raise ValueError("endpoint must use literal http://127.0.0.1")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("endpoint URL must not include path, query or fragment")
    if parsed.port is None or not (1 <= parsed.port <= 65535):
        raise ValueError("endpoint must include a valid port")
    return url.rstrip("/")


def request_json(
    url: str,
    payload: Mapping[str, Any] | None,
    *,
    timeout: float,
    opener: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> HttpResult:
    """Send a no-auth JSON request with redirects/proxies disabled."""
    parsed_url = urllib.parse.urlsplit(url)
    if parsed_url.scheme != "http" or parsed_url.hostname != "127.0.0.1":
        raise ValueError("HTTP transport accepts only literal 127.0.0.1")
    if parsed_url.username or parsed_url.password or parsed_url.fragment:
        raise ValueError("endpoint credentials and fragments are forbidden")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    safe_opener = opener or urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    start = clock()
    try:
        with safe_opener.open(request, timeout=timeout) as response:
            return HttpResult(
                response.status, dict(response.headers), response.read(), (clock() - start) * 1000
            )
    except urllib.error.HTTPError as error:
        return HttpResult(error.code, dict(error.headers), error.read(), (clock() - start) * 1000)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise EvidenceBlocked("loopback transport failed") from error


def validate_response(
    body: bytes,
    *,
    question_ids: set[str],
    question_options: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Validate exact pinned successful raw Kev shape and probability simplex."""
    try:
        return _validate_response(
            body, question_ids=question_ids, question_options=question_options
        )
    except CandidateRejected:
        raise
    except (TypeError, KeyError, IndexError, AttributeError, OverflowError, ValueError) as error:
        raise CandidateRejected("response schema is malformed") from error


def _validate_response(
    body: bytes,
    *,
    question_ids: set[str],
    question_options: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Implementation behind the controlled malformed-response boundary."""

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError("duplicate JSON object key")
            output[key] = item
        return output

    try:
        value = json.loads(body, object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CandidateRejected("response is not JSON") from error
    if not isinstance(value, dict) or set(value) != {"model", "answers", "usage", "latency_ms"}:
        raise CandidateRejected("response keys do not match the pinned contract")
    if value["model"] != "kev-latest" or not isinstance(value["answers"], dict):
        raise CandidateRejected("response model or answers are invalid")
    if set(value["answers"]) != question_ids:
        raise CandidateRejected("response question ids do not match request")
    for question_id, answer in value["answers"].items():
        if not isinstance(answer, dict) or set(answer) != {
            "type",
            "choice",
            "confidence",
            "probabilities",
        }:
            raise CandidateRejected("answer schema is invalid")
        confidence = answer["confidence"]
        if (
            type(confidence) not in (float, int)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise CandidateRejected("confidence must be finite and in [0,1]")
        if answer["type"] != "choice":
            raise CandidateRejected("answer type must be choice")
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, dict) or not probabilities:
            raise CandidateRejected("probabilities must be a non-empty object")
        if not isinstance(answer["choice"], str):
            raise CandidateRejected("selected option must be a string")
        if question_options is not None and set(probabilities) != question_options[question_id]:
            raise CandidateRejected("response option ids do not match request criteria")
        values = list(probabilities.values())
        if any(
            type(n) not in (float, int) or not math.isfinite(n) or n < 0 or n > 1 for n in values
        ):
            raise CandidateRejected("probabilities must be finite values in [0,1]")
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
            raise CandidateRejected("probabilities do not sum to one")
        if answer["choice"] not in probabilities:
            raise CandidateRejected("selected option is absent from probabilities")
    usage = value["usage"]
    if (
        not isinstance(usage, dict)
        or set(usage) != {"input_tokens", "output_tokens"}
        or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in usage.items())
    ):
        raise CandidateRejected("usage counters must be non-negative integers")
    latency = value["latency_ms"]
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
        raise CandidateRejected("latency_ms must be finite and non-negative")
    return value


def _question_options(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    questions = payload.get("questions")
    if not isinstance(questions, Mapping):
        raise EvidenceBlocked("request question mapping is invalid")
    options: dict[str, set[str]] = {}
    for question_id, question in questions.items():
        if not isinstance(question, Mapping) or not isinstance(question.get("criteria"), Mapping):
            raise EvidenceBlocked("request criteria mapping is invalid")
        options[question_id] = set(question["criteria"])
    return options


def _percentile95(values: list[float]) -> float:
    if not values:
        raise EvidenceBlocked("empty timing cell")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def execute_matrix(
    gateway_url: str,
    *,
    timeout: float = 60.0,
    post: Callable[..., HttpResult] = request_json,
    on_request: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run 3 warmups and exactly 240 measured requests in twelve n=20 cells."""
    base = validate_loopback_url(gateway_url)
    cells: dict[str, dict[str, Any]] = {}
    for locale, workloads in matrix().items():
        for qkey, decision in workloads.items():
            payload = convert_fixture_to_wire(decision)
            for kind in ("new_state", "cached_state"):
                times: list[float] = []
                last_response_valid = False
                for index in range(23):
                    request_payload = dict(payload)
                    if kind == "new_state":
                        request_payload["state"] = {
                            "fixture": locale,
                            "cell": qkey,
                            "sample": index,
                        }
                    elif index >= 3:
                        request_payload["state"] = {
                            "fixture": locale,
                            "cell": qkey,
                            "sample": "cached",
                        }
                    if on_request:
                        on_request()
                    result = post(f"{base}/v1/systemone", request_payload, timeout=timeout)
                    if result.status >= 500:
                        raise EvidenceBlocked(f"gateway returned HTTP {result.status}")
                    if result.status != 200:
                        raise CandidateRejected(f"measured request returned HTTP {result.status}")
                    validate_response(
                        result.body,
                        question_ids=set(request_payload["questions"]),
                        question_options=_question_options(request_payload),
                    )
                    last_response_valid = True
                    if index >= 3:
                        times.append(result.elapsed_ms)
                if len(times) != 20:
                    raise EvidenceBlocked("measured cell did not contain n=20")
                cells[f"{locale}/{qkey}/{kind}"] = {
                    "n": len(times),
                    "p95_ms": _percentile95(times),
                    "max_ms": max(times),
                    "latencies_ms": times,
                    "last_response_valid": last_response_valid,
                }
    if sum(cell["n"] for cell in cells.values()) != 240:
        raise EvidenceBlocked("matrix did not produce exactly 240 measured requests")
    return {"measured_requests": 240, "warmups": 36, "cells": cells}


def run_supplemental_probes(
    gateway_url: str,
    *,
    timeout: float = 60.0,
    post: Callable[..., HttpResult] = request_json,
) -> dict[str, Any]:
    """Measure repeat stability, together/separate isolation, and option order."""
    base = validate_loopback_url(gateway_url)
    fixture = matrix()["en"]
    q1 = convert_fixture_to_wire(fixture["q1"])
    repeat_a = post(f"{base}/v1/systemone", q1, timeout=timeout)
    repeat_b = post(f"{base}/v1/systemone", q1, timeout=timeout)
    repeat_first = _valid_probe_response(repeat_a, q1)
    repeat_second = _valid_probe_response(repeat_b, q1)
    repeat_delta = _probability_delta(repeat_first, repeat_second)

    together_decision = fixture["q10"]
    together_payload = convert_fixture_to_wire(together_decision)
    together = _valid_probe_response(
        post(f"{base}/v1/systemone", together_payload, timeout=timeout), together_payload
    )
    separate_answers: dict[str, Any] = {}
    for question_id, question in together_payload["questions"].items():
        separate_payload = dict(together_payload)
        separate_payload["questions"] = {question_id: question}
        separate_result = _valid_probe_response(
            post(f"{base}/v1/systemone", separate_payload, timeout=timeout),
            separate_payload,
        )
        separate_answers.update(separate_result["answers"])
    separate = {"answers": separate_answers}
    isolation_delta = _probability_delta(together, separate)

    permuted = json.loads(json.dumps(q1))
    question_id = next(iter(permuted["questions"]))
    criteria = permuted["questions"][question_id]["criteria"]
    permuted["questions"][question_id]["criteria"] = dict(reversed(list(criteria.items())))
    permutation_response = _valid_probe_response(
        post(f"{base}/v1/systemone", permuted, timeout=timeout), permuted
    )
    order_delta = _probability_delta(repeat_first, permutation_response)
    choice_stable = all(
        together["answers"][qid]["choice"] == separate["answers"][qid]["choice"]
        for qid in together["answers"]
    )
    return {
        "repeat_probability_delta": repeat_delta,
        "together_separate_probability_delta": isolation_delta,
        "option_permutation_probability_delta": order_delta,
        "together_separate_choice_stable": choice_stable,
    }


def _valid_probe_response(result: HttpResult, payload: Mapping[str, Any]) -> dict[str, Any]:
    if result.status >= 500:
        raise EvidenceBlocked(f"supplemental probe server failure HTTP {result.status}")
    if result.status != 200:
        raise CandidateRejected(f"supplemental probe returned HTTP {result.status}")
    return validate_response(
        result.body,
        question_ids=set(payload["questions"]),
        question_options=_question_options(payload),
    )


def _probability_delta(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    if set(first["answers"]) != set(second["answers"]):
        raise CandidateRejected("supplemental response question ids changed")
    maximum = 0.0
    for question_id in first["answers"]:
        left = first["answers"][question_id]["probabilities"]
        right = second["answers"][question_id]["probabilities"]
        if set(left) != set(right):
            raise CandidateRejected("supplemental response option ids changed")
        maximum = max(maximum, *(abs(left[key] - right[key]) for key in left))
    return maximum


def _read_proc_cmdline(pid: int, proc_root: Path = Path("/proc")) -> str:
    return (proc_root / str(pid) / "cmdline").read_text(encoding="utf-8")


def _read_proc_environ(pid: int, proc_root: Path = Path("/proc")) -> tuple[dict[str, str], bool]:
    raw = (proc_root / str(pid) / "environ").read_bytes()
    if not raw or not raw.endswith(b"\0") or raw.endswith(b"\0\0"):
        raise EvidenceBlocked("proc environ framing is malformed")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise EvidenceBlocked("proc environ contains invalid UTF-8") from error
    body = text[:-1]
    items = body.split("\0")
    if any(item == "" for item in items):
        raise EvidenceBlocked("proc environ has an empty interior entry")
    result: dict[str, str] = {}
    secret_present = False
    for item in items:
        if "=" not in item:
            raise EvidenceBlocked("proc environ entry is malformed")
        key, value = item.split("=", 1)
        if key == "":
            raise EvidenceBlocked("proc environ has an empty key")
        if key in result:
            raise EvidenceBlocked("proc environ has duplicate keys")
        if key == "KEV_API_KEY":
            secret_present = bool(value)
            result[key] = "<nonempty>" if secret_present else ""
        else:
            result[key] = value
    return result, secret_present


def _read_proc_stat(pid: int, proc_root: Path = Path("/proc")) -> dict[str, int]:
    raw = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    end = raw.rfind(")")
    if end < 0:
        raise EvidenceBlocked("malformed proc stat")
    fields = raw[end + 2 :].split()
    if len(fields) < 20:
        raise EvidenceBlocked("truncated proc stat")
    return {"utime": int(fields[11]), "stime": int(fields[12]), "starttime": int(fields[19])}


def _get_listen_socket_inode(
    pid: int, expected_port: int, proc_root: Path = Path("/proc"), host: str = DEFAULT_HOST
) -> int:
    lines = (proc_root / "net/tcp").read_text(encoding="ascii").splitlines()[1:]
    port_hex = f"{expected_port:04X}"
    host_hex = bytes(reversed(bytes.fromhex("7f000001"))).hex().upper()
    matches = [
        int(row.split()[9])
        for row in lines
        if len(row.split()) >= 10
        and row.split()[3] == "0A"
        and row.split()[1].rsplit(":", 1)[-1] == port_hex
        and row.split()[1].split(":", 1)[0] == host_hex
        and host == DEFAULT_HOST
    ]
    owned: list[int] = []
    for inode in matches:
        for fd in (proc_root / str(pid) / "fd").iterdir():
            try:
                if os.readlink(fd) == f"socket:[{inode}]":
                    owned.append(inode)
            except OSError:
                continue
    if len(set(owned)) != 1:
        raise EvidenceBlocked("direct LISTEN socket ownership is missing or ambiguous")
    inode = owned[0]
    owners: set[int] = set()
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        fd_directory = process / "fd"
        try:
            entries = tuple(fd_directory.iterdir())
        except OSError:
            continue
        for fd in entries:
            try:
                if os.readlink(fd) == f"socket:[{inode}]":
                    owners.add(int(process.name))
            except OSError:
                continue
    if owners != {pid}:
        raise EvidenceBlocked("direct LISTEN socket is shared or owned by another PID")
    return inode


def _validate_python_entry(
    argv0: str,
    *,
    pid: int,
    proc_root: Path,
    path_stat_reader: Callable[[Path, bool], Any] = lambda path, follow: (
        path.stat() if follow else path.lstat()
    ),
    symlink_reader: Callable[[Path], str] = os.readlink,
    proc_exe_reader: Callable[[int, Path], str] = lambda process_id, root: os.readlink(
        root / str(process_id) / "exe"
    ),
    approved_system_roots: tuple[Path, ...] = (Path("/usr/bin"), Path("/usr/local/bin")),
) -> tuple[Path, Path]:
    """Validate lexical capsule entry and its bounded kernel executable identity."""
    entry = Path(argv0)
    if (
        not entry.is_absolute()
        or str(entry) != argv0
        or argv0.startswith("//")
        or any(part in (".", "..") for part in argv0.split("/"))
    ):
        raise EvidenceBlocked("Python argv0 must be normalized absolute path")
    roots = [parent for parent in entry.parents if re.fullmatch(r"[0-9a-f]{64}", parent.name)]
    if len(roots) != 1:
        raise EvidenceBlocked("Python argv0 requires exactly one content-addressed capsule root")
    capsule = roots[0]
    if entry != capsule / "environment/bin/python":
        raise EvidenceBlocked("Python argv0 is not the exact capsule environment/bin/python entry")

    def metadata(path: Path, *, follow: bool, directory: bool = False) -> Any:
        try:
            info = path_stat_reader(path, follow)
        except OSError as error:
            raise EvidenceBlocked("Python executable path metadata is unavailable") from error
        mode = info.st_mode
        if info.st_uid != 0 or (not stat.S_ISLNK(mode) and mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise EvidenceBlocked("Python executable path is not root-owned and non-writable")
        if directory and (not stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            raise EvidenceBlocked("capsule ancestor is not a real directory")
        return info

    try:
        resolved_capsule = capsule.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise EvidenceBlocked("capsule root is not a real normalized directory") from error
    if resolved_capsule != capsule:
        raise EvidenceBlocked("capsule root is not a real normalized directory")
    for current in (capsule, capsule / "environment", capsule / "environment/bin"):
        metadata(current, follow=False, directory=True)

    try:
        entry_info = metadata(entry, follow=False)
    except EvidenceBlocked:
        raise
    target = entry
    was_symlink = stat.S_ISLNK(entry_info.st_mode)
    if was_symlink:
        visited: set[Path] = set()
        for hop in range(9):
            if target in visited:
                raise EvidenceBlocked("Python symlink chain loops")
            visited.add(target)
            try:
                link_stat = metadata(target, follow=False)
            except EvidenceBlocked:
                raise
            if not stat.S_ISLNK(link_stat.st_mode):
                break
            if hop == 8:
                raise EvidenceBlocked("Python symlink chain exceeds eight hops")
            try:
                raw_target = symlink_reader(target)
            except OSError as error:
                raise EvidenceBlocked("Python symlink target is unavailable") from error
            next_target = Path(raw_target)
            unresolved = next_target if next_target.is_absolute() else target.parent / next_target
            if (
                str(next_target) != raw_target
                or raw_target.startswith("//")
                or any(part in (".", "..") for part in str(unresolved).split("/"))
            ):
                raise EvidenceBlocked("Python symlink target is not normalized")
            target = unresolved
            allowed = (capsule / "environment/bin", *approved_system_roots)
            if not any(target == root or root in target.parents for root in allowed):
                if target.is_absolute() and any(
                    prefix in target.parents for prefix in (Path("/usr"), Path("/usr/local"))
                ):
                    raise EvidenceBlocked("Python symlink target uses an unapproved system prefix")
                raise EvidenceBlocked("Python symlink target escapes approved executable roots")
    else:
        target = entry

    allowed_system = tuple(Path(root) for root in approved_system_roots)
    if (
        was_symlink
        and not any(target == root or root in target.parents for root in allowed_system)
        and (target != entry or capsule not in target.parents)
    ):
        raise EvidenceBlocked("Python final target is outside approved system prefixes")
    final_info = metadata(target, follow=True)
    if not stat.S_ISREG(final_info.st_mode) or not final_info.st_mode & 0o111:
        raise EvidenceBlocked("Python final target is not a regular executable")
    if was_symlink and not any(target == root or root in target.parents for root in allowed_system):
        raise EvidenceBlocked("Python symlink final target is outside approved system prefixes")
    try:
        proc_exe_raw = proc_exe_reader(pid, proc_root)
        proc_exe = Path(proc_exe_raw)
        proc_info = path_stat_reader(proc_root / str(pid) / "exe", True)
    except OSError as error:
        raise EvidenceBlocked("proc executable identity is unavailable") from error
    if (
        not proc_exe.is_absolute()
        or str(proc_exe) != proc_exe_raw
        or proc_exe_raw.startswith("//")
        or any(part in (".", "..") for part in proc_exe_raw.split("/"))
    ):
        raise EvidenceBlocked("proc executable path is not normalized absolute path")
    if proc_exe != target or (final_info.st_dev, final_info.st_ino) != (
        proc_info.st_dev,
        proc_info.st_ino,
    ):
        raise EvidenceBlocked("Python final target does not match proc executable path and inode")
    return capsule, target


def validate_runtime_identity(
    pid: int,
    expected_checkpoint: str,
    host: str = DEFAULT_HOST,
    direct_upstream_port: int = DEFAULT_PORT,
    *,
    gateway_url: str | None = None,
    getuid: Callable[[], int] = os.getuid,
    geteuid: Callable[[], int] = os.geteuid,
    proc_root: Path = Path("/proc"),
    transport: Callable[..., HttpResult] = request_json,
    stat_reader: Callable[[int, Path], dict[str, int]] | None = None,
    owner_reader: Callable[[Path], int] | None = None,
    cmdline_reader: Callable[[int, Path], str] | None = None,
    environ_reader: Callable[[int, Path], tuple[dict[str, str], bool]] | None = None,
    cwd_reader: Callable[[Path], Path] | None = None,
    socket_reader: Callable[[int, int, Path, str], int] | None = None,
    path_stat_reader: Callable[[Path, bool], Any] = lambda path, follow: (
        path.stat() if follow else path.lstat()
    ),
    symlink_reader: Callable[[Path], str] = os.readlink,
    proc_exe_reader: Callable[[int, Path], str] = lambda process_id, root: os.readlink(
        root / str(process_id) / "exe"
    ),
    approved_system_roots: tuple[Path, ...] = (Path("/usr/bin"), Path("/usr/local/bin")),
) -> dict[str, Any]:
    """Fail closed on command/environment/capsule/socket/CPU identity mismatch."""
    if not Path(expected_checkpoint).is_absolute():
        raise EvidenceBlocked("expected checkpoint must be an absolute path")
    if geteuid() == 0:
        raise EvidenceBlocked("runner refuses root")
    if direct_upstream_port != 8182:
        raise EvidenceBlocked("direct upstream port must remain 8182")
    proc = proc_root / str(pid)
    read_owner = owner_reader or (lambda path: path.stat().st_uid)

    def identity_read(reader: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return reader(*args, **kwargs)
        except OSError as error:
            raise EvidenceBlocked("candidate runtime identity became unavailable") from error

    try:
        initial_owner = read_owner(proc)
    except OSError as error:
        raise EvidenceBlocked("candidate PID is missing or inaccessible") from error
    if initial_owner != getuid():
        raise EvidenceBlocked("candidate PID is not owned by invoking user")
    read_stat = stat_reader or _read_proc_stat
    read_cmdline = cmdline_reader or _read_proc_cmdline
    read_environ = environ_reader or _read_proc_environ
    read_cwd = cwd_reader or (lambda path: path.resolve(strict=True))
    read_socket = socket_reader or _get_listen_socket_inode
    before = identity_read(read_stat, pid, proc_root)
    raw = identity_read(read_cmdline, pid, proc_root)
    argv = [item for item in raw.split("\0") if item]
    if len(argv) != 11 or argv[1:3] != ["-m", "kev.serve"]:
        raise EvidenceBlocked("candidate command is not capsule Python -m kev.serve")
    if Path(argv[0]).name != "python":
        raise EvidenceBlocked("candidate executable is not Python")
    flags = argv[3:]
    if len(flags) != 8 or any(not flags[index].startswith("--") for index in (0, 2, 4, 6)):
        raise EvidenceBlocked("candidate argv shape is invalid")
    pairs = list(zip(flags[::2], flags[1::2], strict=True))
    expected_flags = {"--run", "--fallback", "--host", "--port"}
    if len({key for key, _ in pairs}) != len(pairs) or {key for key, _ in pairs} != expected_flags:
        raise EvidenceBlocked("candidate argv has duplicate or unknown flags")
    args = dict(pairs)
    expected = {
        "--run": expected_checkpoint,
        "--fallback": expected_checkpoint,
        "--host": host,
        "--port": str(direct_upstream_port),
    }
    if args != expected:
        raise EvidenceBlocked("candidate command identity mismatch")
    if flags != [item for pair in expected.items() for item in pair]:
        raise EvidenceBlocked("candidate command ordering differs from the pinned argv")
    capsule_root, _executable = _validate_python_entry(
        argv[0],
        pid=pid,
        proc_root=proc_root,
        path_stat_reader=path_stat_reader,
        symlink_reader=symlink_reader,
        proc_exe_reader=proc_exe_reader,
        approved_system_roots=approved_system_roots,
    )
    checkpoint = identity_read(Path(expected_checkpoint).resolve, strict=True)
    process_cwd = identity_read(read_cwd, proc / "cwd")
    if process_cwd != capsule_root:
        raise EvidenceBlocked("candidate working directory is not the capsule root")
    module_path = identity_read(
        (capsule_root / "payload" / "source" / "kev" / "kev" / "serve.py").resolve, strict=True
    )
    fallback = identity_read(Path(args["--fallback"]).resolve, strict=True)
    if any(
        path != capsule_root and capsule_root not in path.parents
        for path in (checkpoint, fallback, module_path)
    ):
        raise EvidenceBlocked("candidate module/checkpoint/fallback escapes capsule root")
    if not module_path.is_file() or checkpoint != fallback:
        raise EvidenceBlocked("candidate module or checkpoint/fallback identity mismatch")
    for entry in identity_read(lambda: tuple(capsule_root.iterdir())):
        if entry.name == "kev" or entry.name.startswith("kev."):
            raise EvidenceBlocked("capsule root has a top-level kev entry that shadows the module")
    environ, secret_present = identity_read(read_environ, pid, proc_root)
    required: dict[str, str] = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost",
        "KEV_BACKEND": "torch",
        "KEV_DTYPE": "bf16",
        "KEV_MERGE": "0",
        "KEV_FUSED": "0",
        "KEV_CUDA_GRAPHS": "0",
        "KEV_DATE_FACTS": "0",
        "KEV_LORA_SCALE": "1",
        "KEV_PREFIX_CACHE": "4",
        "KEV_PREFIX_MIN_TOKENS": "0",
        "KEV_PREFIX_MAX_TOKENS": "65536",
        "HOME": f"{capsule_root}/derived",
        "XDG_CACHE_HOME": f"{capsule_root}/cache",
        "HF_HOME": f"{capsule_root}/hf-home",
        "HUGGINGFACE_HUB_CACHE": f"{capsule_root}/hf-home/hub",
        "TRANSFORMERS_CACHE": f"{capsule_root}/cache",
        "TORCH_HOME": f"{capsule_root}/cache",
        "TRITON_CACHE_DIR": f"{capsule_root}/cache",
        "TMPDIR": f"{capsule_root}/tmp",
        "PYTHONPYCACHEPREFIX": f"{capsule_root}/cache",
        "KEV_BENCHMARK_OUTPUT": f"{capsule_root}/reports",
        "PYTHONPATH": f"{capsule_root}/payload/source/kev",
        "KEV_API_KEY": "<nonempty>",
    }
    if (
        set(environ.keys()) != set(required.keys())
        or any(environ.get(key) != value for key, value in required.items())
        or not secret_present
    ):
        raise EvidenceBlocked("candidate environment identity mismatch")
    inode = identity_read(read_socket, pid, direct_upstream_port, proc_root, host)
    if gateway_url is None:
        raise EvidenceBlocked("gateway URL required for CPU attribution")
    control = matrix()["en"]["q1"]
    result = transport(
        f"{validate_loopback_url(gateway_url)}/v1/systemone",
        convert_fixture_to_wire(control),
        timeout=10,
    )
    if result.status != 200:
        raise EvidenceBlocked("gateway attribution control did not return HTTP 200")
    candidate_rejection: CandidateRejected | None = None
    try:
        validate_response(
            result.body,
            question_ids={q.id for q in control.questions},
            question_options=_question_options(convert_fixture_to_wire(control)),
        )
    except CandidateRejected as error:
        candidate_rejection = error
    after = identity_read(read_stat, pid, proc_root)
    if (
        before["starttime"] != after["starttime"]
        or after["utime"] + after["stime"] <= before["utime"] + before["stime"]
    ):
        raise EvidenceBlocked("candidate PID changed or received no gateway-attributed CPU work")
    if (
        identity_read(read_owner, proc) != initial_owner
        or initial_owner != getuid()
        or identity_read(read_socket, pid, direct_upstream_port, proc_root, host) != inode
    ):
        raise EvidenceBlocked("candidate ownership or LISTEN socket changed")
    if candidate_rejection is not None:
        raise candidate_rejection
    return {
        "cmdline_valid": True,
        "environment_valid": True,
        "socket_valid": True,
        "cpu_attribution_valid": True,
        "pid": pid,
        "uid": initial_owner,
        "starttime": before["starttime"],
        "socket_inode": inode,
    }


def _parse_status(raw: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in raw.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
            parts = line.split()
            if len(parts) != 3 or parts[2] != "kB":
                raise EvidenceBlocked("malformed proc status metric")
            values[parts[0][:-1]] = int(parts[1]) * 1024
    if set(values) != {"VmRSS", "VmHWM", "VmSwap"} or any(value < 0 for value in values.values()):
        raise EvidenceBlocked("mandatory proc status metrics missing or invalid")
    return values


def _read_host_swap() -> int:
    matches = re.findall(
        r"^(?:SwapTotal|SwapFree):\s+(\d+)\s+kB$",
        Path("/proc/meminfo").read_text(encoding="ascii"),
        re.M,
    )
    if len(matches) != 2:
        raise EvidenceBlocked("host swap metrics missing")
    return (int(matches[0]) - int(matches[1])) * 1024


def _validate_proc_sample(sample: Mapping[str, int]) -> None:
    if set(sample) != {"VmRSS", "VmHWM", "VmSwap"} or any(
        type(value) is not int or value < 0 for value in sample.values()
    ):
        raise EvidenceBlocked("procfs sample is missing or malformed")
    if sample["VmRSS"] == 0 or sample["VmHWM"] == 0:
        raise EvidenceBlocked("procfs RSS metrics are invalid")


def _validate_gpu_sample(sample: Mapping[str, Any], pid: int) -> None:
    required = {
        "started",
        "ended",
        "duration_ms",
        "compute_pids",
        "candidate_gpu_mib",
        "device_total_mib",
    }
    if set(sample) != required:
        raise EvidenceBlocked("nvidia-smi sample is missing or malformed")
    numeric = (sample["started"], sample["ended"], sample["duration_ms"])
    memory = (sample["candidate_gpu_mib"], sample["device_total_mib"])
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in numeric + memory
    ):
        raise EvidenceBlocked("nvidia-smi sample contains invalid numeric evidence")
    if sample["started"] < 0 or sample["ended"] < sample["started"]:
        raise EvidenceBlocked("nvidia-smi sample timestamps are invalid")
    if sample["duration_ms"] > 3000 or sample["candidate_gpu_mib"] < 0:
        raise EvidenceBlocked("nvidia-smi sample is delayed or has invalid candidate memory")
    if sample["device_total_mib"] < 24576:
        raise EvidenceBlocked("candidate GPU has less than 24 GiB total memory")
    if sample["compute_pids"] != [pid]:
        raise EvidenceBlocked("candidate GPU attribution is missing, shared, or foreign")


def _nvidia_sample(
    pid: int,
    run: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    env = {"PATH": "/usr/bin:/bin"}
    start = clock()
    try:
        device = run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
            env=env,
            check=False,
        )
        apps = run(
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceBlocked("nvidia-smi unavailable or delayed") from error
    end = clock()
    if device.returncode != 0 or apps.returncode != 0 or end - start > 3:
        raise EvidenceBlocked("nvidia-smi failed or exceeded sampling bound")
    capacities: dict[str, float] = {}
    for line in device.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise EvidenceBlocked("malformed nvidia-smi device row")
        try:
            capacity = float(parts[1])
        except ValueError as error:
            raise EvidenceBlocked("malformed nvidia-smi device capacity") from error
        if not parts[0] or not math.isfinite(capacity) or capacity <= 0:
            raise EvidenceBlocked("invalid nvidia-smi device capacity")
        capacities[parts[0]] = capacity
    if not capacities:
        raise EvidenceBlocked("no GPU devices reported")
    compute: dict[str, dict[int, float]] = {}
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise EvidenceBlocked("malformed nvidia-smi compute row")
        try:
            gpu_pid, memory = int(parts[1]), float(parts[2])
        except ValueError as error:
            raise EvidenceBlocked("malformed nvidia-smi compute metric") from error
        if parts[0] not in capacities or gpu_pid <= 0 or not math.isfinite(memory) or memory < 0:
            raise EvidenceBlocked("invalid nvidia-smi compute identity or memory")
        if gpu_pid in compute.setdefault(parts[0], {}):
            raise EvidenceBlocked("duplicate nvidia-smi compute PID")
        compute[parts[0]][gpu_pid] = memory
    candidate_devices = [uuid for uuid, pids in compute.items() if pid in pids]
    if len(candidate_devices) != 1:
        raise EvidenceBlocked("candidate GPU memory attribution missing or ambiguous")
    gpu = candidate_devices[0]
    if any(other != pid for other in compute[gpu]):
        raise EvidenceBlocked("foreign GPU compute PID observed")
    if capacities[gpu] < 24576:
        raise EvidenceBlocked("candidate GPU has less than 24 GiB total memory")
    return {
        "started": start,
        "ended": end,
        "duration_ms": (end - start) * 1000,
        "compute_pids": sorted(compute[gpu]),
        "candidate_gpu_mib": compute[gpu][pid],
        "device_total_mib": capacities[gpu],
    }


def sample_resources(
    pid: int,
    duration_seconds: float = 1.0,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    proc_reader: Callable[[int], dict[str, int]] | None = None,
    gpu_reader: Callable[[int], dict[str, Any]] | None = None,
    swap_reader: Callable[[], int] = _read_host_swap,
    identity_reader: Callable[[int], dict[str, int]] | None = None,
    owner_reader: Callable[[int], int] | None = None,
    expected_identity: Mapping[str, int] | None = None,
    socket_reader: Callable[[int, int, Path, str], int] | None = None,
    host: str = DEFAULT_HOST,
    direct_upstream_port: int = DEFAULT_PORT,
    readiness_event: threading.Event | None = None,
    proc_root: Path = Path("/proc"),
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Collect procfs and GPU samples concurrently; fail on any cadence gap."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    read_proc = proc_reader or (
        lambda p: _parse_status((proc_root / str(p) / "status").read_text(encoding="ascii"))
    )
    read_gpu = gpu_reader or _nvidia_sample
    read_identity = identity_reader or _read_proc_stat
    read_owner = owner_reader or (lambda p: (proc_root / str(p)).stat().st_uid)
    read_socket = socket_reader or _get_listen_socket_inode
    owner = expected_identity["uid"] if expected_identity else read_owner(pid)
    starttime = (
        expected_identity["starttime"] if expected_identity else read_identity(pid).get("starttime")
    )
    expected_socket_inode = expected_identity.get("socket_inode") if expected_identity else None
    if starttime is None or (expected_identity and expected_identity.get("pid") != pid):
        raise EvidenceBlocked("validated candidate identity is incomplete or mismatched")
    if read_owner(pid) != owner or read_identity(pid).get("starttime") != starttime:
        raise EvidenceBlocked("candidate PID changed after identity validation")
    if (
        expected_socket_inode is not None
        and read_socket(pid, direct_upstream_port, proc_root, host) != expected_socket_inode
    ):
        raise EvidenceBlocked("candidate LISTEN socket changed after identity validation")
    host_swap_before = swap_reader()
    start = clock()
    internal_stop = threading.Event()
    proc_samples: list[tuple[float, dict[str, int]]] = []
    gpu_samples: list[dict[str, Any]] = []
    closing_proc: list[tuple[float, dict[str, int]]] = []
    closing_gpu: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    readiness_lock = threading.Lock()

    def signal_ready() -> None:
        if readiness_event is not None:
            with readiness_lock:
                if proc_samples and gpu_samples:
                    readiness_event.set()

    def proc_loop() -> None:
        previous = start
        while not internal_stop.is_set():
            is_closing = (
                bool(stop_event and stop_event.is_set()) or clock() - start >= duration_seconds
            )
            now = clock()
            try:
                current_identity = read_identity(pid)
                if current_identity.get("starttime") != starttime or read_owner(pid) != owner:
                    raise EvidenceBlocked("PID owner/starttime changed")
                if (
                    expected_socket_inode is not None
                    and read_socket(pid, direct_upstream_port, proc_root, host)
                    != expected_socket_inode
                ):
                    raise EvidenceBlocked("candidate LISTEN socket changed during sampling")
                sample = read_proc(pid)
                if clock() - now > 1.0:
                    raise EvidenceBlocked("procfs observation was delayed beyond one second")
                _validate_proc_sample(sample)
                if now - previous > 1.0:
                    raise EvidenceBlocked("procfs cadence gap exceeded one second")
                if is_closing:
                    closing_proc.append((now, sample))
                else:
                    proc_samples.append((now, sample))
                signal_ready()
                previous = now
                if is_closing:
                    return
                sleep(min(0.25, max(0.0, duration_seconds - (now - start))))
            except BaseException as error:
                errors.append(error)
                internal_stop.set()
                return

    def gpu_loop() -> None:
        previous = start
        while not internal_stop.is_set():
            is_closing = (
                bool(stop_event and stop_event.is_set()) or clock() - start >= duration_seconds
            )
            now = clock()
            try:
                current_identity = read_identity(pid)
                if current_identity.get("starttime") != starttime or read_owner(pid) != owner:
                    raise EvidenceBlocked("PID owner/starttime changed during GPU sampling")
                if (
                    expected_socket_inode is not None
                    and read_socket(pid, direct_upstream_port, proc_root, host)
                    != expected_socket_inode
                ):
                    raise EvidenceBlocked("candidate LISTEN socket changed during GPU sampling")
                sample = read_gpu(pid)
                _validate_gpu_sample(sample, pid)
                if now - previous > 3.0 or float(sample["started"]) - previous > 3.0:
                    raise EvidenceBlocked("nvidia-smi cadence gap exceeded three seconds")
                if is_closing:
                    closing_gpu.append(sample)
                else:
                    gpu_samples.append(sample)
                signal_ready()
                previous = now
                if is_closing:
                    return
                sleep(min(1.0, max(0.0, duration_seconds - (clock() - start))))
            except BaseException as error:
                errors.append(error)
                internal_stop.set()
                return

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(proc_loop), pool.submit(gpu_loop)]
        for future in futures:
            future.result()
    if errors:
        raise EvidenceBlocked("mandatory concurrent resource sampling failed") from errors[0]
    if not proc_samples or not gpu_samples or not closing_proc or not closing_gpu:
        raise EvidenceBlocked("resource window lacks periodic or closing observations")
    if closing_proc[0][0] - proc_samples[-1][0] > 1.0:
        raise EvidenceBlocked("closing procfs observation exceeded one-second cadence")
    if float(closing_gpu[0]["started"]) - float(gpu_samples[-1]["started"]) > 3.0:
        raise EvidenceBlocked("closing nvidia-smi observation exceeded three-second cadence")
    host_swap_after = swap_reader()
    all_proc_samples = proc_samples + closing_proc
    all_gpu_samples = gpu_samples + closing_gpu
    baseline_swap = all_proc_samples[0][1]["VmSwap"]
    sampled_swaps = [sample[1]["VmSwap"] for sample in all_proc_samples]
    if any(value < baseline_swap for value in sampled_swaps):
        raise EvidenceBlocked("negative swap delta or candidate VmSwap counter reset")
    candidate_swap_delta = max(sampled_swaps) - baseline_swap
    host_swap_delta = host_swap_after - host_swap_before
    if candidate_swap_delta < 0 or host_swap_delta < 0:
        raise EvidenceBlocked("negative swap delta is ambiguous")
    if host_swap_delta > 0 and candidate_swap_delta == 0:
        raise EvidenceBlocked("host swap increased without candidate attribution")
    device_total = min(sample["device_total_mib"] for sample in all_gpu_samples)
    if device_total < 24576:
        raise EvidenceBlocked("candidate GPU has less than 24 GiB total memory")
    identity_survived = (
        read_identity(pid).get("starttime") == starttime
        and read_owner(pid) == owner
        and (
            expected_socket_inode is None
            or read_socket(pid, direct_upstream_port, proc_root, host) == expected_socket_inode
        )
    )
    if not identity_survived:
        raise EvidenceBlocked("candidate PID owner/starttime/socket changed during sampling")
    return {
        "peak_rss_bytes": max(x[1]["VmRSS"] for x in all_proc_samples),
        "peak_hwm_bytes": max(x[1]["VmHWM"] for x in all_proc_samples),
        "peak_vmswap_bytes": max(x[1]["VmSwap"] for x in all_proc_samples),
        "candidate_swap_delta_bytes": candidate_swap_delta,
        "host_swap_delta_bytes": host_swap_delta,
        "gpu_peak_mib": max(x["candidate_gpu_mib"] for x in all_gpu_samples),
        "device_total_mib": device_total,
        "proc_timestamps": [x[0] for x in proc_samples],
        "nvidia_samples": gpu_samples,
        "closing_proc_timestamp": closing_proc[0][0],
        "closing_nvidia_sample": closing_gpu[0],
        "identity_survived": True,
    }


class ResourceMonitor:
    """Run a resource window concurrently with requests until explicitly stopped."""

    def __init__(
        self,
        pid: int,
        *,
        sampler: Callable[..., dict[str, Any]] = sample_resources,
        expected_identity: Mapping[str, int] | None = None,
        readiness_timeout: float = 5.0,
    ) -> None:
        self._pid = pid
        self._sampler = sampler
        self._expected_identity = expected_identity
        self._readiness_timeout = readiness_timeout
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None

    def start(self) -> ResourceMonitor:
        if self._thread is not None:
            raise ValueError("resource monitor already started")

        def collect() -> None:
            try:
                self.result = self._sampler(
                    self._pid,
                    duration_seconds=86_400.0,
                    stop_event=self._stop_event,
                    expected_identity=self._expected_identity,
                    readiness_event=self._ready_event,
                )
            except BaseException as error:
                self.error = error
                self._ready_event.set()

        self._thread = threading.Thread(
            target=collect, name="phase5b-resource-monitor", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + self._readiness_timeout
        while not self._ready_event.wait(min(0.05, max(0.0, deadline - time.monotonic()))):
            if self.error is not None or not self._thread.is_alive():
                raise EvidenceBlocked(
                    "resource monitor failed before initial samples"
                ) from self.error
            if time.monotonic() >= deadline:
                self._stop_event.set()
                self._thread.join(timeout=1)
                raise EvidenceBlocked("resource monitor initial-sample readiness timed out")
        if self.error is not None:
            raise EvidenceBlocked("resource monitor failed before initial samples") from self.error
        if self.result is not None:
            raise EvidenceBlocked("resource monitor ended before readiness")
        return self

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            raise ValueError("resource monitor was not started")
        self._stop_event.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise EvidenceBlocked("resource monitor did not stop promptly")
        if self.error is not None:
            raise EvidenceBlocked("resource monitor failed") from self.error
        if self.result is None:
            raise EvidenceBlocked("resource monitor produced no evidence")
        if (
            self.result.get("closing_proc_timestamp") is None
            or self.result.get("closing_nvidia_sample") is None
        ):
            raise EvidenceBlocked("resource monitor ended without closing observations")
        return self.result


def run_oversize_probes(
    gateway_url: str,
    timeout: float,
    *,
    post: Callable[..., HttpResult] = request_json,
    diagnostic_request: Callable[[Callable[[], HttpResult]], HttpResult] | None = None,
    on_primary_complete: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Probe valid Q=1 controls then mutate exactly one contracted dimension."""
    base = validate_loopback_url(gateway_url)
    control = convert_fixture_to_wire(matrix()["en"]["q1"])
    control_result = post(f"{base}/v1/systemone", control, timeout=timeout)
    if control_result.status >= 500:
        raise EvidenceBlocked("oversize valid control transport/server failure")
    if control_result.status != 200:
        raise CandidateRejected("oversize valid control was not HTTP 200")
    validate_response(
        control_result.body,
        question_ids=set(control["questions"]),
        question_options=_question_options(control),
    )
    invalid = post(f"{base}/v1/systemone", {**control, "unknown": True}, timeout=timeout)
    if invalid.status >= 500:
        raise EvidenceBlocked("invalid-contract probe server failure")
    if invalid.status != 422:
        raise CandidateRejected("invalid contract did not return 422")
    branch_payload = json.loads(json.dumps(control))
    branch_payload["questions"][next(iter(branch_payload["questions"]))]["instructions"] = [
        json.dumps({"item": i}) for i in range(100_000)
    ]
    branch = post(f"{base}/v1/systemone", branch_payload, timeout=timeout)
    state_payload = {**control, "state": [json.dumps({"item": i}) for i in range(100_000)]}
    # The diagnostic state workload has its own resource window and a fresh
    # valid control immediately before it.
    state_control = post(f"{base}/v1/systemone", control, timeout=timeout)
    if state_control.status >= 500:
        raise EvidenceBlocked("state valid control server failure")
    if state_control.status != 200:
        raise CandidateRejected("state valid control was not HTTP 200")
    validate_response(
        state_control.body,
        question_ids=set(control["questions"]),
        question_options=_question_options(control),
    )

    def state_call() -> HttpResult:
        return post(f"{base}/v1/systemone", state_payload, timeout=timeout)

    if on_primary_complete:
        on_primary_complete()
    state = state_call() if diagnostic_request is None else diagnostic_request(state_call)
    for label, result in (("branch", branch), ("state", state)):
        if result.status >= 500:
            raise EvidenceBlocked(f"{label} oversize probe server failure")
    if branch.status != 422:
        if branch.status == 200:
            validate_response(
                branch.body,
                question_ids=set(branch_payload["questions"]),
                question_options=_question_options(branch_payload),
            )
            raise CandidateRejected("oversize branch was accepted")
        raise EvidenceBlocked(f"oversize branch protocol drift HTTP {branch.status}")
    if state.status == 422:
        raise EvidenceBlocked("pinned source state 422 is protocol drift")
    if state.status != 200:
        raise EvidenceBlocked(f"oversize state protocol drift HTTP {state.status}")
    parsed = validate_response(
        state.body,
        question_ids=set(state_payload["questions"]),
        question_options=_question_options(state_payload),
    )
    tokens = parsed["usage"].get("input_tokens")
    if type(tokens) is not int or not 65536 <= tokens < 73728:
        raise CandidateRejected("state truncation response lacks bounded saturated input_tokens")
    return {
        "control_status": 200,
        "invalid_status": 422,
        "branch_status": 422,
        "state_status": 200,
        "state_outcome": "accepted_with_pinned_source_truncation",
        "reported_input_tokens": tokens,
    }
