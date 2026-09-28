from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.phase5_managed_cuda.runner import (
    CandidateRejected,
    EvidenceBlocked,
    HttpResult,
    ResourceMonitor,
    _nvidia_sample,
    _read_proc_environ,
    _validate_python_entry,
    execute_matrix,
    request_json,
    run_oversize_probes,
    run_supplemental_probes,
    sample_resources,
    validate_loopback_url,
    validate_response,
    validate_runtime_identity,
)


def _valid_response(payload: dict[str, Any]) -> bytes:
    questions = payload["questions"]
    answers = {
        key: {
            "choice": sorted(item["criteria"])[0],
            "probabilities": {
                criterion: 1.0 if index == 0 else 0.0
                for index, criterion in enumerate(sorted(item["criteria"]))
            },
        }
        for key, item in questions.items()
    }
    return json.dumps(
        {
            "model": "kev-latest",
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "latency_ms": 1.0,
        }
    ).encode()


def test_loopback_rejects_dns_and_non_loopback() -> None:
    for value in (
        "http://localhost:8182",
        "http://127.0.0.2:8182",
        "https://127.0.0.1:8182",
        "http://user@127.0.0.1:8182",
    ):
        with pytest.raises(ValueError):
            validate_loopback_url(value)


def test_success_response_is_exact_and_validated() -> None:
    payload = {"questions": {"q": {"criteria": {"a": "A"}}}}
    assert validate_response(_valid_response(payload), question_ids={"q"})["model"] == "kev-latest"
    malformed = json.loads(_valid_response(payload))
    malformed["surprise"] = True
    with pytest.raises(CandidateRejected):
        validate_response(json.dumps(malformed).encode(), question_ids={"q"})
    malformed = json.loads(_valid_response(payload))
    malformed["answers"]["q"]["probabilities"] = {"unknown": 1.0}
    with pytest.raises(CandidateRejected):
        validate_response(
            json.dumps(malformed).encode(),
            question_ids={"q"},
            question_options={"q": {"a"}},
        )
    with pytest.raises(CandidateRejected):
        validate_response(b'{"model":"kev-latest","model":"kev-latest"}', question_ids={"q"})


def test_http_transport_sends_no_auth_and_does_not_follow_redirects() -> None:
    observed: dict[str, Any] = {}

    class Response:
        def __init__(self) -> None:
            self.status = 200
            self.headers: dict[str, str] = {}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    class Opener:
        def open(self, request: Any, *, timeout: float) -> Response:
            observed["headers"] = dict(request.header_items())
            observed["timeout"] = timeout
            return Response()

    request_json(
        "http://127.0.0.1:8181/v1/systemone", {"model": "kev-latest"}, timeout=2, opener=Opener()
    )
    assert "Authorization" not in observed["headers"]
    assert observed["timeout"] == 2
    from benchmarks.phase5_managed_cuda.runner import _NoRedirect

    _NoRedirect().redirect_request(None, None, 302, "redirect", {}, "http://127.0.0.1:8182")
    payload = {"questions": {"q": {"criteria": {"a": "A"}}}}
    malformed = json.loads(_valid_response(payload))
    malformed["answers"]["q"]["probabilities"]["a"] = 0.2
    with pytest.raises(CandidateRejected):
        validate_response(json.dumps(malformed).encode(), question_ids={"q"})


def test_matrix_executes_exactly_240_measurements_and_36_warmups() -> None:
    calls = 0

    def fake_post(_url: str, payload: dict[str, Any], *, timeout: float) -> HttpResult:
        nonlocal calls
        calls += 1
        return HttpResult(200, {}, _valid_response(payload), 12.0)

    result = execute_matrix("http://127.0.0.1:8181", post=fake_post)
    assert result["measured_requests"] == 240
    assert result["warmups"] == 36
    assert calls == 276
    assert len(result["cells"]) == 12
    assert {cell["n"] for cell in result["cells"].values()} == {20}
    assert all(cell["last_response_valid"] for cell in result["cells"].values())
    assert all(len(cell["latencies_ms"]) == 20 for cell in result["cells"].values())


def test_matrix_blocks_transport_and_protocol_failures() -> None:
    with pytest.raises(EvidenceBlocked):
        execute_matrix(
            "http://127.0.0.1:8181",
            post=lambda *_args, **_kwargs: (_ for _ in ()).throw(EvidenceBlocked("transport")),
        )
    with pytest.raises(CandidateRejected):
        execute_matrix(
            "http://127.0.0.1:8181",
            post=lambda *_args, **_kwargs: HttpResult(422, {}, b"{}", 1.0),
        )


@pytest.mark.parametrize("probe", ["matrix", "oversize"])
def test_list_valued_choice_is_controlled_candidate_rejection(probe: str) -> None:
    def fake_post(_url: str, payload: dict[str, Any], *, timeout: float) -> HttpResult:
        response = json.loads(_valid_response(payload))
        first = next(iter(response["answers"].values()))
        first["choice"] = []
        return HttpResult(200, {}, json.dumps(response).encode(), 1.0)

    with pytest.raises(CandidateRejected, match="selected option must be a string"):
        if probe == "matrix":
            execute_matrix("http://127.0.0.1:8181", post=fake_post)
        else:
            run_oversize_probes("http://127.0.0.1:8181", 60, post=fake_post)


def test_supplemental_probes_measure_all_pinned_comparisons() -> None:
    paths: list[str] = []

    def fake_post(url: str, payload: dict[str, Any], *, timeout: float) -> HttpResult:
        paths.append(url)
        return HttpResult(200, {}, _valid_response(payload), 1.0)

    result = run_supplemental_probes("http://127.0.0.1:8181", post=fake_post)
    assert len(paths) == 14
    assert set(paths) == {"http://127.0.0.1:8181/v1/systemone"}
    assert result["repeat_probability_delta"] == 0.0
    assert result["together_separate_probability_delta"] == 0.0
    assert result["option_permutation_probability_delta"] == 0.0
    assert result["together_separate_choice_stable"] is True


def test_oversize_probes_use_valid_controls_and_extract_tokens() -> None:
    payloads: list[dict[str, Any]] = []

    def fake_post(_url: str, payload: dict[str, Any], *, timeout: float) -> HttpResult:
        payloads.append(json.loads(json.dumps(payload)))
        if "unknown" in payload:
            return HttpResult(422, {}, b"{}", 1.0)
        if isinstance(payload.get("state"), list):
            response = json.loads(_valid_response(payload))
            response["usage"]["input_tokens"] = 65_607
            return HttpResult(200, {}, json.dumps(response).encode(), 1.0)
        first = next(iter(payload["questions"].values()))
        if len(first["instructions"]) == 100_000:
            return HttpResult(422, {}, b"{}", 1.0)
        return HttpResult(200, {}, _valid_response(payload), 1.0)

    result = run_oversize_probes("http://127.0.0.1:8181", 60, post=fake_post)
    assert len(payloads) == 5
    assert payloads[1] == payloads[0] | {"unknown": True}
    assert len(payloads[2]["questions"]) == 1
    assert len(next(iter(payloads[2]["questions"].values()))["instructions"]) == 100_000
    assert {
        key: value
        for key, value in next(iter(payloads[2]["questions"].values())).items()
        if key != "instructions"
    } == {
        key: value
        for key, value in next(iter(payloads[0]["questions"].values())).items()
        if key != "instructions"
    }
    assert len(payloads[4]["state"]) == 100_000
    assert {key: value for key, value in payloads[4].items() if key != "state"} == {
        key: value for key, value in payloads[3].items() if key != "state"
    }
    assert result["branch_status"] == 422
    assert result["state_outcome"] == "accepted_with_pinned_source_truncation"
    assert result["reported_input_tokens"] == 65_607


@pytest.mark.parametrize("status", [401, 403, 422, 429])
def test_state_protocol_drift_is_blocked(status: int) -> None:
    def fake_post(_url: str, payload: dict[str, Any], *, timeout: float) -> HttpResult:
        if isinstance(payload.get("state"), list):
            return HttpResult(status, {}, b"{}", 1.0)
        if (
            "unknown" in payload
            or len(next(iter(payload["questions"].values()))["instructions"]) == 100_000
        ):
            return HttpResult(422, {}, b"{}", 1.0)
        return HttpResult(200, {}, _valid_response(payload), 1.0)

    with pytest.raises(EvidenceBlocked):
        run_oversize_probes("http://127.0.0.1:8181", 60, post=fake_post)


def test_oversize_transport_failure_blocks() -> None:
    with pytest.raises(EvidenceBlocked):
        run_oversize_probes(
            "http://127.0.0.1:8181",
            60,
            post=lambda *_args, **_kwargs: HttpResult(503, {}, b"", 1.0),
        )


def test_runtime_identity_rejects_missing_pid() -> None:
    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            pid=999999,
            expected_checkpoint="/fake",
            gateway_url="http://127.0.0.1:8181",
            geteuid=lambda: 1000,
        )


def test_runtime_identity_exact_command_environment_socket_and_gateway_cpu(tmp_path: Any) -> None:
    capsule = tmp_path / ("a" * 64)
    executable = capsule / "environment/bin/python"
    module = capsule / "payload" / "source" / "kev" / "kev" / "serve.py"
    checkpoint = capsule / "checkpoint"
    system_root = tmp_path / "approved-python"
    system_root.mkdir()
    final_python = system_root / "python3.12"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(capsule, target_is_directory=True)
    final_python.write_text("python", encoding="utf-8")
    final_python.chmod(0o755)
    (executable.parent / "python3").symlink_to(final_python)
    executable.symlink_to("python3")
    (process / "exe").symlink_to(final_python)
    module.write_text("", encoding="utf-8")
    checkpoint.write_text("", encoding="utf-8")
    command = [
        str(executable),
        "-m",
        "kev.serve",
        "--run",
        str(checkpoint),
        "--fallback",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "8182",
    ]
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    environment = {
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
        "HOME": f"{capsule}/derived",
        "XDG_CACHE_HOME": f"{capsule}/cache",
        "HF_HOME": f"{capsule}/hf-home",
        "HUGGINGFACE_HUB_CACHE": f"{capsule}/hf-home/hub",
        "TRANSFORMERS_CACHE": f"{capsule}/cache",
        "TORCH_HOME": f"{capsule}/cache",
        "TRITON_CACHE_DIR": f"{capsule}/cache",
        "TMPDIR": f"{capsule}/tmp",
        "PYTHONPYCACHEPREFIX": f"{capsule}/cache",
        "KEV_BENCHMARK_OUTPUT": f"{capsule}/reports",
        "PYTHONPATH": f"{capsule}/payload/source/kev",
        "KEV_API_KEY": "private-secret",
    }
    (process / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items()) + b"\0"
    )
    ticks = iter((1, 2))
    socket_calls: list[int] = []
    transport_calls: list[tuple[str, dict[str, Any]]] = []

    def transport(url: str, payload: dict[str, Any], *, timeout: float) -> HttpResult:
        transport_calls.append((url, payload))
        return HttpResult(200, {}, _valid_response(payload), 1.0)

    def socket_reader(pid: int, port: int, root: Path, host: str) -> int:
        socket_calls.append(port)
        assert pid == 321 and root == proc_root and host == "127.0.0.1"
        return 17

    observed = validate_runtime_identity(
        pid=321,
        expected_checkpoint=str(checkpoint),
        gateway_url="http://127.0.0.1:8181",
        getuid=lambda: process.stat().st_uid,
        geteuid=lambda: 1000,
        proc_root=proc_root,
        **_identity_test_seams(executable, approved_system_roots=(system_root,)),
        transport=transport,
        stat_reader=lambda _pid, _root: {"starttime": 99, "utime": next(ticks), "stime": 1},
        socket_reader=socket_reader,
    )
    assert observed == {
        "cmdline_valid": True,
        "environment_valid": True,
        "socket_valid": True,
        "cpu_attribution_valid": True,
        "pid": 321,
        "uid": process.stat().st_uid,
        "starttime": 99,
        "socket_inode": 17,
    }
    assert socket_calls == [8182, 8182]
    assert len(transport_calls) == 1
    assert transport_calls[0][0] == "http://127.0.0.1:8181/v1/systemone"
    assert "locale" not in transport_calls[0][1]
    assert "private-secret" not in json.dumps(observed)

    def valid_stat(_pid: int, _root: Path) -> dict[str, int]:
        return {"starttime": 99, "utime": 1, "stime": 1}

    failure_boundaries: list[dict[str, Any]] = [
        {
            "stat_reader": lambda *_args: (_ for _ in ()).throw(
                FileNotFoundError("/private/proc/stat")
            )
        },
        {
            "cmdline_reader": lambda *_args: (_ for _ in ()).throw(
                PermissionError("/private/proc/cmdline")
            )
        },
        {"environ_reader": lambda *_args: (_ for _ in ()).throw(OSError("/private/proc/environ"))},
        {
            "cwd_reader": lambda *_args: (_ for _ in ()).throw(
                FileNotFoundError("/private/proc/cwd")
            )
        },
        {"cwd_reader": lambda *_args: (_ for _ in ()).throw(PermissionError("/private/proc/cwd"))},
    ]
    for boundary in failure_boundaries:
        injected = {"stat_reader": valid_stat, "socket_reader": socket_reader} | boundary
        with pytest.raises(EvidenceBlocked) as failure:
            validate_runtime_identity(
                pid=321,
                expected_checkpoint=str(checkpoint),
                gateway_url="http://127.0.0.1:8181",
                getuid=lambda: process.stat().st_uid,
                geteuid=lambda: 1000,
                proc_root=proc_root,
                **_identity_test_seams(executable, approved_system_roots=(system_root,)),
                transport=transport,
                **injected,
            )
        assert str(failure.value) == "candidate runtime identity became unavailable"

    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            pid=321,
            expected_checkpoint=str(checkpoint),
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(executable),
            cmdline_reader=lambda *_args: "\0".join([*command[:-2], "--unknown", "x"]) + "\0",
            stat_reader=lambda *_args: {"starttime": 99, "utime": 1, "stime": 1},
            socket_reader=socket_reader,
        )


def test_resource_sampler_uses_concurrent_injected_readers() -> None:
    proc_count = 0
    gpu_count = 0
    lock = threading.Lock()

    def proc_reader(_pid: int) -> dict[str, int]:
        nonlocal proc_count
        with lock:
            proc_count += 1
        return {"VmRSS": 100, "VmHWM": 200, "VmSwap": 30}

    def gpu_reader(pid: int) -> dict[str, Any]:
        nonlocal gpu_count
        with lock:
            gpu_count += 1
        return {
            "started": time.monotonic(),
            "ended": time.monotonic(),
            "duration_ms": 1.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 2000.0,
            "device_total_mib": 24576.0,
        }

    result = sample_resources(
        99,
        duration_seconds=0.08,
        proc_reader=proc_reader,
        gpu_reader=gpu_reader,
        swap_reader=iter((100, 100)).__next__,
        identity_reader=lambda _pid: {"starttime": 7, "utime": 1, "stime": 1},
        owner_reader=lambda _pid: 1000,
    )
    assert proc_count >= 1 and gpu_count >= 1
    assert result["candidate_swap_delta_bytes"] == 0
    assert result["device_total_mib"] == 24576
    assert result["identity_survived"] is True


def test_resource_sampler_blocks_foreign_pid_and_host_swap_ambiguity() -> None:
    common: dict[str, Any] = {
        "duration_seconds": 0.02,
        "proc_reader": lambda _pid: {"VmRSS": 1, "VmHWM": 1, "VmSwap": 10},
        "identity_reader": lambda _pid: {"starttime": 7},
        "owner_reader": lambda _pid: 1000,
    }
    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99,
            gpu_reader=lambda _pid: (_ for _ in ()).throw(EvidenceBlocked("foreign GPU PID")),
            swap_reader=lambda: 0,
            **common,
        )
    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99,
            gpu_reader=lambda pid: {
                "started": time.monotonic(),
                "ended": time.monotonic(),
                "duration_ms": 1.0,
                "compute_pids": [pid],
                "candidate_gpu_mib": 100.0,
                "device_total_mib": 24576.0,
            },
            swap_reader=iter((0, 1)).__next__,
            **common,
        )


def test_fixed_nvidia_smi_sample_attributes_pid_and_device_memory() -> None:
    commands: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> Any:
        commands.append(argv)
        text = (
            "GPU-1, 24576\n"
            if any("--query-gpu=" in argument for argument in argv)
            else "GPU-1, 42, 1024\n"
        )
        return SimpleNamespace(returncode=0, stdout=text)

    observed = _nvidia_sample(42, run=run)
    assert observed["compute_pids"] == [42]
    assert observed["device_total_mib"] == 24576
    assert all(command[0] == "/usr/bin/nvidia-smi" for command in commands)


def _runtime_case(tmp_path: Any) -> tuple[dict[str, str], list[str], Any, Any]:
    capsule = tmp_path / ("a" * 64)
    executable = capsule / "environment/bin/python"
    module = capsule / "payload" / "source" / "kev" / "kev" / "serve.py"
    checkpoint = capsule / "checkpoint"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(capsule, target_is_directory=True)
    (process / "exe").symlink_to(executable)
    executable.write_text("", encoding="utf-8")
    executable.chmod(0o755)
    module.write_text("", encoding="utf-8")
    checkpoint.write_text("", encoding="utf-8")
    command = [
        str(executable),
        "-m",
        "kev.serve",
        "--run",
        str(checkpoint),
        "--fallback",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "8182",
    ]
    environment = {
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
        "HOME": f"{capsule}/derived",
        "XDG_CACHE_HOME": f"{capsule}/cache",
        "HF_HOME": f"{capsule}/hf-home",
        "HUGGINGFACE_HUB_CACHE": f"{capsule}/hf-home/hub",
        "TRANSFORMERS_CACHE": f"{capsule}/cache",
        "TORCH_HOME": f"{capsule}/cache",
        "TRITON_CACHE_DIR": f"{capsule}/cache",
        "TMPDIR": f"{capsule}/tmp",
        "PYTHONPYCACHEPREFIX": f"{capsule}/cache",
        "KEV_BENCHMARK_OUTPUT": f"{capsule}/reports",
        "PYTHONPATH": f"{capsule}/payload/source/kev",
        "KEV_API_KEY": "private-secret",
    }
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    return environment, command, process, proc_root


def _identity_test_seams(
    executable: Any,
    *,
    proc_exe: Any | None = None,
    approved_system_roots: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    final_python = executable.resolve()

    def path_stat(path: Any, follow: bool) -> Any:
        info = (
            final_python.stat() if path.name == "exe" else (path.stat() if follow else path.lstat())
        )
        return SimpleNamespace(
            st_uid=0,
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_ino=info.st_ino,
        )

    seams = {
        "path_stat_reader": path_stat,
        "proc_exe_reader": lambda _pid, _root: str(proc_exe or final_python),
    }
    if approved_system_roots is not None:
        return {
            **seams,
            "approved_system_roots": approved_system_roots,
        }
    return seams


def test_python_entry_accepts_real_two_hop_venv_symlink(tmp_path: Any) -> None:
    capsule = tmp_path / ("b" * 64)
    entry = capsule / "environment/bin/python"
    system = tmp_path / "system/usr/bin"
    system.mkdir(parents=True)
    (capsule / "environment/bin").mkdir(parents=True)
    target = system / "python3.12"
    target.write_text("python", encoding="utf-8")
    target.chmod(0o755)
    (entry.parent / "python3").symlink_to(target)
    entry.symlink_to("python3")
    proc = tmp_path / "proc"
    (proc / "42").mkdir(parents=True)
    executable_info = target.stat()

    def path_stat(path: Any, follow: bool) -> Any:
        info = target.stat() if path.name == "exe" else (path.stat() if follow else path.lstat())
        return SimpleNamespace(
            st_uid=0, st_mode=info.st_mode, st_dev=info.st_dev, st_ino=info.st_ino
        )

    capsule_root, resolved = _validate_python_entry(
        str(entry),
        pid=42,
        proc_root=proc,
        path_stat_reader=path_stat,
        proc_exe_reader=lambda *_: str(target),
        approved_system_roots=(system,),
    )
    assert capsule_root == capsule
    assert resolved == target
    assert executable_info.st_ino == target.stat().st_ino


def test_python_entry_accepts_eight_real_symlink_hops(tmp_path: Any) -> None:
    capsule = tmp_path / ("f" * 64)
    entry = capsule / "environment/bin/python"
    system = tmp_path / "system/usr/bin"
    system.mkdir(parents=True)
    entry.parent.mkdir(parents=True)
    target = system / "python3"
    target.write_text("python", encoding="utf-8")
    target.chmod(0o755)
    for index in range(7):
        link = entry.parent / f"hop{index}"
        link.symlink_to(f"hop{index + 1}" if index < 6 else target)
    entry.symlink_to("hop0")
    proc = tmp_path / "proc"
    (proc / "44").mkdir(parents=True)

    def path_stat(path: Any, follow: bool) -> Any:
        info = target.stat() if path.name == "exe" else (path.stat() if follow else path.lstat())
        return SimpleNamespace(
            st_uid=0, st_mode=info.st_mode, st_dev=info.st_dev, st_ino=info.st_ino
        )

    _, resolved = _validate_python_entry(
        str(entry),
        pid=44,
        proc_root=proc,
        path_stat_reader=path_stat,
        proc_exe_reader=lambda *_: str(target),
        approved_system_roots=(system,),
    )
    assert resolved == target


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("relative", "normalized absolute path"),
        ("dotdot", "normalized absolute path"),
        ("double_slash", "normalized absolute path"),
        ("missing_root", "exactly one content-addressed"),
        ("multiple_roots", "exactly one content-addressed"),
        ("wrong_entry", "exact capsule environment/bin/python"),
        ("symlink_capsule", "real normalized directory"),
        ("symlink_environment", "real directory"),
        ("symlink_bin", "real directory"),
        ("writable_capsule", "root-owned and non-writable"),
        ("groupwrite_environment", "root-owned and non-writable"),
        ("worldwrite_bin", "root-owned and non-writable"),
        ("nonroot_capsule", "root-owned and non-writable"),
        ("nonroot_environment", "root-owned and non-writable"),
        ("nonroot_bin", "root-owned and non-writable"),
        ("dangling", "metadata is unavailable"),
        ("loop", "loops"),
        ("too_many", "exceeds eight hops"),
        ("escape", "escapes approved executable roots"),
        ("non_normalized_hop", "symlink target is not normalized"),
        ("unapproved", "unapproved system prefix"),
        ("non_executable", "not a regular executable"),
        ("writable_target", "root-owned and non-writable"),
        ("groupwrite_target", "root-owned and non-writable"),
        ("worldwrite_target", "root-owned and non-writable"),
        ("nonroot_target", "root-owned and non-writable"),
        ("nonregular_target", "not a regular executable"),
        ("proc_path", "does not match proc executable path and inode"),
        ("proc_inode", "does not match proc executable path and inode"),
        ("proc_non_normalized", "proc executable path is not normalized"),
    ],
)
def test_python_entry_rejects_unsafe_identity_branches(
    tmp_path: Any, failure: str, reason: str
) -> None:
    capsule = tmp_path / ("c" * 64)
    entry = capsule / "environment/bin/python"
    system = tmp_path / "approved"
    system.mkdir()
    (capsule / "environment/bin").mkdir(parents=True)
    proc = tmp_path / "proc"
    (proc / "43").mkdir(parents=True)
    target = system / "python"
    target.write_text("python", encoding="utf-8")
    target.chmod(0o755)
    proc_target = target
    if failure == "non_executable":
        target.chmod(0o644)
    elif failure == "writable_target":
        target.chmod(0o777)
    elif failure == "groupwrite_target":
        target.chmod(0o770)
    elif failure == "worldwrite_target":
        target.chmod(0o757)
    elif failure == "nonregular_target":
        target.unlink()
        target.mkdir()
    if failure == "dangling":
        entry.symlink_to(system / "missing")
    elif failure == "loop":
        first, second = entry.parent / "one", entry.parent / "two"
        entry.symlink_to("one")
        first.symlink_to("two")
        second.symlink_to("one")
    elif failure == "too_many":
        for index in range(9):
            (entry.parent / f"hop{index}").symlink_to(f"hop{index + 1}" if index < 8 else target)
        entry.symlink_to("hop0")
    elif failure == "escape":
        outside = tmp_path / "outside"
        outside.write_text("python", encoding="utf-8")
        outside.chmod(0o755)
        entry.symlink_to(outside)
    elif failure == "unapproved":
        entry.symlink_to("/usr/sbin/python")
    elif failure == "non_normalized_hop":
        entry.symlink_to("./python")
    elif failure in {
        "non_executable",
        "writable_target",
        "groupwrite_target",
        "worldwrite_target",
        "nonroot_target",
        "nonregular_target",
        "proc_path",
        "proc_inode",
    }:
        entry.symlink_to(target)
    else:
        entry.write_text("python", encoding="utf-8")
        entry.chmod(0o755)
    if failure == "symlink_capsule":
        real_capsule = tmp_path / ("e" * 64)
        capsule.rename(real_capsule)
        capsule.symlink_to(real_capsule, target_is_directory=True)
    elif failure == "symlink_environment":
        environment = capsule / "environment"
        environment.rename(capsule / "environment-real")
        environment.symlink_to(capsule / "environment-real", target_is_directory=True)
    elif failure == "symlink_bin":
        bin_directory = capsule / "environment/bin"
        bin_directory.rename(capsule / "environment/bin-real")
        bin_directory.symlink_to(capsule / "environment/bin-real", target_is_directory=True)

    def path_stat(path: Any, follow: bool) -> Any:
        info = target.stat() if path.name == "exe" else (path.stat() if follow else path.lstat())
        if path.name == "exe" and failure == "proc_inode":
            info = SimpleNamespace(
                st_mode=info.st_mode,
                st_dev=info.st_dev,
                st_ino=info.st_ino + 1,
            )
        nonroot_paths = {
            "nonroot_capsule": capsule,
            "nonroot_environment": capsule / "environment",
            "nonroot_bin": capsule / "environment/bin",
        }
        uid = 1000 if path == nonroot_paths.get(failure) or failure == "nonroot_target" else 0
        mode = info.st_mode
        writable_paths = {
            "writable_capsule": capsule,
            "groupwrite_environment": capsule / "environment",
            "worldwrite_bin": capsule / "environment/bin",
        }
        if path == writable_paths.get(failure):
            mode |= 0o020 if failure == "groupwrite_environment" else 0o002
        return SimpleNamespace(st_uid=uid, st_mode=mode, st_dev=info.st_dev, st_ino=info.st_ino)

    argv0 = str(entry)
    if failure == "relative":
        argv0 = "relative/python"
    elif failure == "dotdot":
        argv0 = str(entry.parent / ".." / "bin/python")
    elif failure == "double_slash":
        argv0 = "/" + str(entry)
    elif failure == "missing_root":
        argv0 = str(tmp_path / "ordinary/environment/bin/python")
    elif failure == "multiple_roots":
        nested = capsule / ("d" * 64) / "environment/bin/python"
        argv0 = str(nested)
    elif failure == "wrong_entry":
        argv0 = str(entry.parent / "python3")
    elif failure == "proc_path":
        proc_target = system / "other-python"
    elif failure == "proc_non_normalized":
        proc_target = system / ".." / "approved/python"

    with pytest.raises(EvidenceBlocked, match=reason):
        _validate_python_entry(
            argv0,
            pid=43,
            proc_root=proc,
            path_stat_reader=path_stat,
            proc_exe_reader=lambda *_: str(proc_target),
            approved_system_roots=(system,),
        )


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_flag",
        "unknown_kev",
        "provider_env",
        "proxy_env",
        "pythonpath",
        "pythonhome",
    ],
)
def test_runtime_identity_rejects_duplicate_flags_and_unapproved_environment(
    tmp_path: Any, failure: str
) -> None:
    environment, command, process, proc_root = _runtime_case(tmp_path)
    if failure == "unknown_kev":
        environment["KEV_UNKNOWN"] = "x"
    elif failure == "provider_env":
        environment["OPENAI_API_KEY"] = "private-provider-token"
    elif failure == "proxy_env":
        environment["HTTPS_PROXY"] = "http://private-proxy.invalid"
    elif failure == "pythonpath":
        environment["PYTHONPATH"] = "/external/imports"
    elif failure == "pythonhome":
        environment["PYTHONHOME"] = "/external/python"
    raw_command = command[:]
    if failure == "duplicate_flag":
        raw_command[9] = "--run"

    def stat_reader(_pid: int, _root: Any) -> dict[str, int]:
        return {"starttime": 7, "utime": 1, "stime": 1}

    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            321,
            command[4],
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(Path(command[0])),
            cmdline_reader=lambda *_: "\0".join(raw_command) + "\0",
            environ_reader=lambda *_: (environment, True),
            stat_reader=stat_reader,
            socket_reader=lambda *_: 17,
            transport=lambda _url, payload, **_kw: HttpResult(200, {}, _valid_response(payload), 1),
        )


@pytest.mark.parametrize("escape", ["executable", "checkpoint", "fallback", "module"])
def test_runtime_identity_rejects_capsule_escape_or_disagreement(
    tmp_path: Any, escape: str
) -> None:
    environment, command, process, proc_root = _runtime_case(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("", encoding="utf-8")
    altered = command[:]
    if escape == "executable":
        altered[0] = str(outside)
    elif escape == "checkpoint":
        altered[5] = str(outside)
    elif escape == "fallback":
        altered[7] = str(outside)
    else:
        module = tmp_path / ("a" * 64) / "payload" / "source" / "kev" / "kev" / "serve.py"
        module.unlink()
        module.symlink_to(outside)
    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            321,
            command[4],
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(Path(command[0])),
            cmdline_reader=lambda *_: "\0".join(altered) + "\0",
            environ_reader=lambda *_: (environment, True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=lambda _url, payload, **_kw: HttpResult(200, {}, _valid_response(payload), 1),
        )


@pytest.mark.parametrize("cwd_kind", ["external", "capsule_child"])
def test_runtime_identity_rejects_non_root_cwd(tmp_path: Any, cwd_kind: str) -> None:
    environment, command, process, proc_root = _runtime_case(tmp_path)
    capsule = Path(command[0]).parents[2]
    target_cwd = tmp_path / "external-cwd"
    expected_reason = "working directory is not the capsule root"
    if cwd_kind == "capsule_child":
        target_cwd = capsule / "child"
        target_cwd.mkdir()
    else:
        shadow = target_cwd / "kev/serve.py"
        shadow.parent.mkdir(parents=True)
        shadow.write_text("# shadow module", encoding="utf-8")
    (process / "cwd").unlink()
    (process / "cwd").symlink_to(target_cwd, target_is_directory=True)
    with pytest.raises(EvidenceBlocked, match=expected_reason):
        validate_runtime_identity(
            321,
            command[4],
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(Path(command[0])),
            environ_reader=lambda *_: (environment, True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=lambda _url, payload, **_kw: HttpResult(200, {}, _valid_response(payload), 1),
        )


@pytest.mark.parametrize("failure", ["missing", "inaccessible"])
def test_runtime_identity_normalizes_unavailable_process_cwd(tmp_path: Any, failure: str) -> None:
    environment, command, process, proc_root = _runtime_case(tmp_path)
    error_type = FileNotFoundError if failure == "missing" else PermissionError

    def missing_cwd(_path: Any) -> Any:
        raise error_type("/private/proc/321/cwd")

    with pytest.raises(EvidenceBlocked) as error:
        validate_runtime_identity(
            321,
            command[4],
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(Path(command[0])),
            environ_reader=lambda *_: (environment, True),
            cwd_reader=missing_cwd,
        )
    assert str(error.value) == "candidate runtime identity became unavailable"


@pytest.mark.parametrize("socket_state", ["missing", "shared", "ambiguous"])
def test_runtime_identity_requires_unique_stable_socket_ownership(
    tmp_path: Any, socket_state: str
) -> None:
    environment, command, process, proc_root = _runtime_case(tmp_path)
    calls = 0

    def socket_reader(*_args: Any) -> int:
        nonlocal calls
        calls += 1
        if socket_state == "missing":
            raise EvidenceBlocked("no direct listener")
        if socket_state == "shared":
            return 17 if calls == 1 else 18
        raise EvidenceBlocked("ambiguous direct listener")

    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            321,
            command[4],
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(Path(command[0])),
            environ_reader=lambda *_: (environment, True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=socket_reader,
            transport=lambda _url, payload, **_kw: HttpResult(200, {}, _valid_response(payload), 1),
        )


@pytest.mark.parametrize("transition", ["owner", "starttime"])
def test_runtime_identity_rejects_pid_owner_or_starttime_transition(
    tmp_path: Any, transition: str
) -> None:
    environment, command, process, proc_root = _runtime_case(tmp_path)
    owners = iter((process.stat().st_uid, process.stat().st_uid + 1))
    starts = iter((7, 8))
    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            321,
            command[4],
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            **_identity_test_seams(Path(command[0])),
            environ_reader=lambda *_: (environment, True),
            owner_reader=(lambda _path: next(owners)) if transition == "owner" else None,
            stat_reader=lambda *_: {"starttime": next(starts), "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=lambda _url, payload, **_kw: HttpResult(200, {}, _valid_response(payload), 1),
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_command",
        "nonzero",
        "malformed",
        "low_capacity",
        "missing_pid",
        "foreign_pid",
        "delayed",
    ],
)
def test_nvidia_sample_blocks_missing_malformed_foreign_and_delayed_evidence(failure: str) -> None:
    def run(argv: list[str], **_kwargs: Any) -> Any:
        if failure == "missing_command":
            raise FileNotFoundError("nvidia-smi")
        if failure == "nonzero":
            return SimpleNamespace(returncode=1, stdout="")
        if any("query-gpu=" in item for item in argv):
            value = (
                "bad"
                if failure == "malformed"
                else "GPU-1, 20000\n"
                if failure == "low_capacity"
                else "GPU-1, 24576\n"
            )
        elif failure == "missing_pid":
            value = ""
        elif failure == "foreign_pid":
            value = "GPU-1, 42, 100\nGPU-1, 43, 50\n"
        else:
            value = "GPU-1, 42, 100\n"
        return SimpleNamespace(returncode=0, stdout=value)

    ticks = iter((0.0, 4.0))
    with pytest.raises(EvidenceBlocked):
        _nvidia_sample(42, run=run, clock=lambda: next(ticks))


def test_sampler_rejects_swap_reset_pid_change_and_reports_candidate_swap() -> None:
    base: dict[str, Any] = {
        "duration_seconds": 0.35,
        "proc_reader": lambda _pid: {"VmRSS": 1, "VmHWM": 1, "VmSwap": 0},
        "gpu_reader": lambda pid: {
            "started": 0.0,
            "ended": 0.01,
            "duration_ms": 10.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 1.0,
            "device_total_mib": 24576.0,
        },
        "owner_reader": lambda _pid: 1000,
        "identity_reader": lambda _pid: {"starttime": 7},
    }
    read_count = 0

    def candidate_swap(_pid: int) -> dict[str, int]:
        nonlocal read_count
        read_count += 1
        return {"VmRSS": 1, "VmHWM": 1, "VmSwap": 0 if read_count == 1 else 2 * 1024**3}

    result = sample_resources(
        99,
        swap_reader=lambda: 0,
        proc_reader=candidate_swap,
        **{key: value for key, value in base.items() if key != "proc_reader"},
    )
    assert result["candidate_swap_delta_bytes"] > 1024**3
    swaps = iter((10, 5, 5, 5))
    with pytest.raises(EvidenceBlocked):
        sample_resources(99, swap_reader=lambda: next(swaps), **base)
    candidate_reset = iter((10, 5, 5, 5))
    with pytest.raises(EvidenceBlocked, match="negative swap delta"):
        sample_resources(
            99,
            swap_reader=lambda: 10,
            **(
                base
                | {
                    "proc_reader": lambda _pid: {
                        "VmRSS": 1,
                        "VmHWM": 1,
                        "VmSwap": next(candidate_reset),
                    }
                }
            ),
        )
    starts = iter((7, 8, 8, 8))
    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99,
            swap_reader=lambda: 0,
            **(base | {"identity_reader": lambda _pid: {"starttime": next(starts)}}),
        )
    owners = iter((1000, 1001, 1001))
    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99, swap_reader=lambda: 0, **(base | {"owner_reader": lambda _pid: next(owners)})
        )


def test_sampler_uses_transient_candidate_swap_peak_after_recovery() -> None:
    swaps: list[int] = []

    def proc_reader(_pid: int) -> dict[str, int]:
        value = 100 if not swaps else 2 * 1024**3 + 100 if len(swaps) == 1 else 100
        swaps.append(value)
        return {"VmRSS": 1, "VmHWM": 1, "VmSwap": value}

    evidence = sample_resources(
        99,
        duration_seconds=0.55,
        proc_reader=proc_reader,
        gpu_reader=lambda pid: {
            "started": time.monotonic(),
            "ended": time.monotonic() + 0.01,
            "duration_ms": 10.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 1.0,
            "device_total_mib": 24576.0,
        },
        owner_reader=lambda _pid: 1000,
        identity_reader=lambda _pid: {"starttime": 7},
        swap_reader=lambda: 0,
    )
    assert len(swaps) >= 3 and swaps[0] == swaps[-1] == 100
    assert evidence["candidate_swap_delta_bytes"] == 2 * 1024**3


def test_sampler_blocks_procfs_and_nvidia_cadence_gaps() -> None:
    def proc_reader(_pid: int) -> dict[str, int]:
        time.sleep(1.05)
        return {"VmRSS": 1, "VmHWM": 1, "VmSwap": 0}

    def gpu(pid: int) -> dict[str, Any]:
        return {
            "started": time.monotonic(),
            "ended": time.monotonic(),
            "duration_ms": 1.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 1.0,
            "device_total_mib": 24576.0,
        }

    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99,
            2.2,
            proc_reader=proc_reader,
            gpu_reader=gpu,
            swap_reader=lambda: 0,
            identity_reader=lambda _pid: {"starttime": 7},
            owner_reader=lambda _pid: 1000,
            socket_reader=lambda _pid, _port, _root, _host: 17,
        )

    def delayed_gpu(pid: int) -> dict[str, Any]:
        time.sleep(3.05)
        return gpu(pid)

    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99,
            4.2,
            proc_reader=lambda _pid: {"VmRSS": 1, "VmHWM": 1, "VmSwap": 0},
            gpu_reader=delayed_gpu,
            swap_reader=lambda: 0,
            identity_reader=lambda _pid: {"starttime": 7},
            owner_reader=lambda _pid: 1000,
            socket_reader=lambda _pid, _port, _root, _host: 17,
        )


def test_resource_monitor_waits_for_both_initial_samples_and_binds_identity() -> None:
    expected = {"pid": 99, "uid": 1000, "starttime": 7, "socket_inode": 17}
    sampler = partial(
        sample_resources,
        proc_reader=lambda _pid: {"VmRSS": 10, "VmHWM": 10, "VmSwap": 0},
        gpu_reader=lambda pid: {
            "started": time.monotonic(),
            "ended": time.monotonic(),
            "duration_ms": 1.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 1.0,
            "device_total_mib": 24576.0,
        },
        swap_reader=lambda: 0,
        identity_reader=lambda _pid: {"starttime": 7, "utime": 1, "stime": 1},
        owner_reader=lambda _pid: 1000,
        socket_reader=lambda _pid, _port, _root, _host: 17,
    )
    monitor = ResourceMonitor(99, sampler=sampler, expected_identity=expected, readiness_timeout=1)
    started_at = time.monotonic()
    monitor.start()
    assert time.monotonic() >= started_at
    evidence = monitor.stop()
    assert evidence["identity_survived"] is True
    assert evidence["proc_timestamps"] and evidence["nvidia_samples"]
    assert evidence["closing_proc_timestamp"] >= evidence["proc_timestamps"][-1]
    assert evidence["closing_nvidia_sample"]["started"] >= evidence["nvidia_samples"][-1]["started"]


def test_short_resource_window_captures_higher_closing_gpu_peak() -> None:
    calls = 0

    def gpu(pid: int) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        value = 100.0 if calls == 1 else 3000.0
        now = time.monotonic()
        return {
            "started": now,
            "ended": now,
            "duration_ms": 1.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": value,
            "device_total_mib": 24576.0,
        }

    monitor = ResourceMonitor(
        99,
        sampler=partial(
            sample_resources,
            proc_reader=lambda _pid: {"VmRSS": 10, "VmHWM": 10, "VmSwap": 0},
            gpu_reader=gpu,
            swap_reader=lambda: 0,
            identity_reader=lambda _pid: {"starttime": 7},
            owner_reader=lambda _pid: 1000,
            socket_reader=lambda _pid, _port, _root, _host: 17,
        ),
        expected_identity={"pid": 99, "uid": 1000, "starttime": 7, "socket_inode": 17},
    ).start()
    evidence = monitor.stop()
    assert evidence["closing_nvidia_sample"]["candidate_gpu_mib"] == 3000.0
    assert evidence["gpu_peak_mib"] == 3000.0


@pytest.mark.parametrize("failure", ["malformed", "delayed", "foreign"])
def test_resource_monitor_blocks_invalid_closing_gpu_observation(failure: str) -> None:
    calls = 0

    def gpu(pid: int) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        now = time.monotonic()
        sample: dict[str, Any] = {
            "started": now,
            "ended": now,
            "duration_ms": 1.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 100.0,
            "device_total_mib": 24576.0,
        }
        if calls > 1:
            if failure == "malformed":
                sample.pop("ended")
            elif failure == "delayed":
                sample["duration_ms"] = 3001.0
            else:
                sample["compute_pids"] = [pid, 444]
        return sample

    monitor = ResourceMonitor(
        99,
        sampler=partial(
            sample_resources,
            proc_reader=lambda _pid: {"VmRSS": 10, "VmHWM": 10, "VmSwap": 0},
            gpu_reader=gpu,
            swap_reader=lambda: 0,
            identity_reader=lambda _pid: {"starttime": 7},
            owner_reader=lambda _pid: 1000,
            socket_reader=lambda _pid, _port, _root, _host: 17,
        ),
        expected_identity={"pid": 99, "uid": 1000, "starttime": 7, "socket_inode": 17},
    ).start()
    with pytest.raises(EvidenceBlocked):
        monitor.stop()


def test_resource_monitor_blocks_identity_drift_at_closure() -> None:
    identity_calls = 0
    lock = threading.Lock()

    def identity(_pid: int) -> dict[str, int]:
        nonlocal identity_calls
        with lock:
            identity_calls += 1
            starttime = 7 if identity_calls < 4 else 8
        return {"starttime": starttime}

    monitor = ResourceMonitor(
        99,
        sampler=partial(
            sample_resources,
            proc_reader=lambda _pid: {"VmRSS": 10, "VmHWM": 10, "VmSwap": 0},
            gpu_reader=lambda pid: {
                "started": time.monotonic(),
                "ended": time.monotonic(),
                "duration_ms": 1.0,
                "compute_pids": [pid],
                "candidate_gpu_mib": 100.0,
                "device_total_mib": 24576.0,
            },
            swap_reader=lambda: 0,
            identity_reader=identity,
            owner_reader=lambda _pid: 1000,
            socket_reader=lambda _pid, _port, _root, _host: 17,
        ),
        expected_identity={"pid": 99, "uid": 1000, "starttime": 7, "socket_inode": 17},
    ).start()
    with pytest.raises(EvidenceBlocked):
        monitor.stop()


@pytest.mark.parametrize("failure", ["malformed", "delayed", "missing"])
def test_resource_monitor_blocks_invalid_closing_procfs_observation(failure: str) -> None:
    calls = 0

    def proc(_pid: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls > 1:
            if failure == "malformed":
                return {"VmRSS": 10, "VmHWM": 10}
            if failure == "delayed":
                time.sleep(1.05)
            else:
                raise FileNotFoundError("injected closing procfs disappearance")
        return {"VmRSS": 10, "VmHWM": 10, "VmSwap": 0}

    monitor = ResourceMonitor(
        99,
        sampler=partial(
            sample_resources,
            proc_reader=proc,
            gpu_reader=lambda pid: {
                "started": time.monotonic(),
                "ended": time.monotonic(),
                "duration_ms": 1.0,
                "compute_pids": [pid],
                "candidate_gpu_mib": 100.0,
                "device_total_mib": 24576.0,
            },
            swap_reader=lambda: 0,
            identity_reader=lambda _pid: {"starttime": 7},
            owner_reader=lambda _pid: 1000,
            socket_reader=lambda _pid, _port, _root, _host: 17,
        ),
        expected_identity={"pid": 99, "uid": 1000, "starttime": 7, "socket_inode": 17},
    ).start()
    with pytest.raises(EvidenceBlocked):
        monitor.stop()


def test_resource_monitor_readiness_failure_and_timeout_are_blocked() -> None:
    def failed_sampler(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise EvidenceBlocked("injected first sample failure")

    with pytest.raises(EvidenceBlocked, match="before initial samples"):
        ResourceMonitor(99, sampler=failed_sampler, readiness_timeout=1).start()

    def delayed_sampler(*_args: Any, stop_event: threading.Event, **_kwargs: Any) -> dict[str, Any]:
        stop_event.wait(1)
        return {}

    with pytest.raises(EvidenceBlocked, match="readiness timed out"):
        ResourceMonitor(99, sampler=delayed_sampler, readiness_timeout=0.02).start()


def test_sampling_rejects_restart_or_socket_drift_from_validated_identity() -> None:
    expected = {"pid": 99, "uid": 1000, "starttime": 7, "socket_inode": 17}
    common: dict[str, Any] = {
        "duration_seconds": 0.02,
        "proc_reader": lambda _pid: {"VmRSS": 1, "VmHWM": 1, "VmSwap": 0},
        "gpu_reader": lambda pid: {
            "started": time.monotonic(),
            "ended": time.monotonic(),
            "duration_ms": 1.0,
            "compute_pids": [pid],
            "candidate_gpu_mib": 1.0,
            "device_total_mib": 24576.0,
        },
        "swap_reader": lambda: 0,
        "identity_reader": lambda _pid: {"starttime": 8, "utime": 1, "stime": 1},
        "owner_reader": lambda _pid: 1000,
        "socket_reader": lambda _pid, _port, _root, _host: 17,
        "expected_identity": expected,
    }
    with pytest.raises(EvidenceBlocked, match="changed after identity validation"):
        sample_resources(99, **common)

    identities = iter((7, 7, 7, 7))
    inodes = iter((17, 18, 18, 18))
    with pytest.raises(EvidenceBlocked):
        sample_resources(
            99,
            **(
                common
                | {
                    "identity_reader": lambda _pid: {"starttime": next(identities)},
                    "socket_reader": lambda _pid, _port, _root, _host: next(inodes),
                }
            ),
        )


def _complete_launcher_environment(capsule_root: Path) -> dict[str, str]:
    return {
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


def test_launcher_environment_fixture_has_exact_30_keys() -> None:
    capsule = Path("/capsule/abcd1234" * 8)
    env = _complete_launcher_environment(capsule)
    assert len(env) == 30
    assert set(env.keys()) == {
        "LANG",
        "LC_ALL",
        "PATH",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "KEV_BACKEND",
        "KEV_DTYPE",
        "KEV_MERGE",
        "KEV_FUSED",
        "KEV_CUDA_GRAPHS",
        "KEV_DATE_FACTS",
        "KEV_LORA_SCALE",
        "KEV_PREFIX_CACHE",
        "KEV_PREFIX_MIN_TOKENS",
        "KEV_PREFIX_MAX_TOKENS",
        "HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "TMPDIR",
        "PYTHONPYCACHEPREFIX",
        "KEV_BENCHMARK_OUTPUT",
        "PYTHONPATH",
        "KEV_API_KEY",
    }
    assert env["KEV_API_KEY"] == "<nonempty>"
    assert env["PYTHONPATH"] == f"{capsule}/payload/source/kev"


def test_launcher_environment_parity_with_runner_required_map(tmp_path: Any) -> None:
    capsule = tmp_path / ("p" * 64)
    env = _complete_launcher_environment(capsule)
    proc_root = tmp_path / "proc"
    process = proc_root / "1"
    process.mkdir(parents=True)
    raw = "\0".join(f"{k}={v}" for k, v in env.items()).encode("utf-8") + b"\0"
    (process / "environ").write_bytes(raw)
    environ, secret_present = _read_proc_environ(1, proc_root)
    assert secret_present is True
    assert environ == env
    assert environ["KEV_API_KEY"] == "<nonempty>"
    assert len(environ) == 30


@pytest.mark.parametrize(
    ("raw_bytes", "reason"),
    [
        (b"", "framing is malformed"),
        (b"KEY=val", "framing is malformed"),
        (b"KEY=val\0\0", "framing is malformed"),
        (b"KEY=val1\0\0KEY=val2\0", "empty interior entry"),
        (b"\0KEY2=val2\0", "empty interior entry"),
        (b"\0", "empty interior entry"),
        (b"KEY1=val1\0KEY2\0KEY3=val3\0", "malformed"),
        (b"=val\0", "empty key"),
        (b"KEY1=val1\0KEY1=val2\0", "duplicate"),
        (
            b"KEY=val1\0KEY=val2\0KEY3=val3\0KEY4=val4\0KEY5=val5\0KEY6=val6\0KEY7=val7\0KEY8=val8\0KEY9=val9\0KEY10=val10\0KEY11=val11\0KEY12=val12\0KEY13=val13\0KEY14=val14\0KEY15=val15\0KEY16=val16\0KEY17=val17\0KEY18=val18\0KEY19=val19\0KEY20=val20\0KEY21=val21\0KEY22=val22\0KEY23=val23\0KEY24=val24\0KEY25=val25\0KEY26=val26\0KEY27=val27\0KEY28=val28\0KEY29=val29\0KEY30=val30\0KEY31=val31\0KEY32=val32\0KEY33=val33\0KEY34=val34\0KEY35=val35\0KEY36=val36\0KEY37=val37\0KEY38=val38\0KEY39=val39\0KEY40=val40\0KEY41=val41\0KEY42=val42\0KEY43=val43\0KEY44=val44\0KEY45=val45\0KEY46=val46\0KEY47=val47\0KEY48=val48\0KEY49=val49\0KEY50=val50\0",
            "duplicate",
        ),
    ],
)
def test_read_proc_environ_rejects_malformed_framing(
    tmp_path: Any, raw_bytes: bytes, reason: str
) -> None:
    proc_root = tmp_path / "proc"
    process = proc_root / "1"
    process.mkdir(parents=True)
    (process / "environ").write_bytes(raw_bytes)
    with pytest.raises(EvidenceBlocked, match=reason):
        _read_proc_environ(1, proc_root)


def test_read_proc_environ_accepts_valid_30key_encoding(tmp_path: Any) -> None:
    entries = {
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
        "HOME": "/capsule/derived",
        "XDG_CACHE_HOME": "/capsule/cache",
        "HF_HOME": "/capsule/hf-home",
        "HUGGINGFACE_HUB_CACHE": "/capsule/hf-home/hub",
        "TRANSFORMERS_CACHE": "/capsule/cache",
        "TORCH_HOME": "/capsule/cache",
        "TRITON_CACHE_DIR": "/capsule/cache",
        "TMPDIR": "/capsule/tmp",
        "PYTHONPYCACHEPREFIX": "/capsule/cache",
        "KEV_BENCHMARK_OUTPUT": "/capsule/reports",
        "PYTHONPATH": "/capsule/payload/source/kev",
    }
    raw = "\0".join(f"{k}={v}" for k, v in entries.items()).encode("utf-8") + b"\0"
    proc_root = tmp_path / "proc"
    process = proc_root / "1"
    process.mkdir(parents=True)
    (process / "environ").write_bytes(raw)
    result, secret_present = _read_proc_environ(1, proc_root)
    assert secret_present is False
    assert len(result) == 29
    assert result == entries


def test_read_proc_environ_accepts_kev_api_key_present(tmp_path: Any) -> None:
    entries = {
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
        "HOME": "/capsule/derived",
        "XDG_CACHE_HOME": "/capsule/cache",
        "HF_HOME": "/capsule/hf-home",
        "HUGGINGFACE_HUB_CACHE": "/capsule/hf-home/hub",
        "TRANSFORMERS_CACHE": "/capsule/cache",
        "TORCH_HOME": "/capsule/cache",
        "TRITON_CACHE_DIR": "/capsule/cache",
        "TMPDIR": "/capsule/tmp",
        "PYTHONPYCACHEPREFIX": "/capsule/cache",
        "KEV_BENCHMARK_OUTPUT": "/capsule/reports",
        "PYTHONPATH": "/capsule/payload/source/kev",
        "KEV_API_KEY": "real-secret",
    }
    raw = "\0".join(f"{k}={v}" for k, v in entries.items()).encode("utf-8") + b"\0"
    proc_root = tmp_path / "proc"
    process = proc_root / "1"
    process.mkdir(parents=True)
    (process / "environ").write_bytes(raw)
    result, secret_present = _read_proc_environ(1, proc_root)
    assert secret_present is True
    assert len(result) == 30
    assert result["KEV_API_KEY"] == "<nonempty>"
    assert all(v != "real-secret" for k, v in result.items() if k == "KEV_API_KEY")


def test_read_proc_environ_rejects_invalid_utf8(tmp_path: Any) -> None:
    raw = b"KEY=val\xc3\x28\0"
    proc_root = tmp_path / "proc"
    process = proc_root / "1"
    process.mkdir(parents=True)
    (process / "environ").write_bytes(raw)
    with pytest.raises(EvidenceBlocked, match="invalid UTF-8"):
        _read_proc_environ(1, proc_root)


def test_runtime_identity_rejects_every_environment_drift(tmp_path: Any) -> None:
    capsule = tmp_path / ("a" * 64)
    executable = capsule / "environment/bin/python"
    module = capsule / "payload" / "source" / "kev" / "kev" / "serve.py"
    checkpoint = capsule / "checkpoint"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(capsule, target_is_directory=True)
    final_python = tmp_path / "python3.12"
    final_python.write_text("python", encoding="utf-8")
    final_python.chmod(0o755)
    (executable.parent / "python3").symlink_to(final_python)
    executable.symlink_to("python3")
    (process / "exe").symlink_to(final_python)
    module.write_text("", encoding="utf-8")
    checkpoint.write_text("", encoding="utf-8")
    command = [
        str(executable),
        "-m",
        "kev.serve",
        "--run",
        str(checkpoint),
        "--fallback",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "8182",
    ]
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")

    def validate(env: dict[str, str]) -> None:
        validate_runtime_identity(
            pid=321,
            expected_checkpoint=str(checkpoint),
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            path_stat_reader=_make_path_stat_reader(tmp_path / "python3.12"),
            proc_exe_reader=lambda *_: str(final_python),
            approved_system_roots=(tmp_path,),
            cmdline_reader=lambda *_: "\0".join(command) + "\0",
            environ_reader=lambda *_: (env, True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=lambda *_a, **_kw: HttpResult(
                200,
                {},
                b'{"model":"kev-latest","answers":{},"usage":{"input_tokens":1,"output_tokens":0},"latency_ms":1}',
                1.0,
            ),
        )

    canonical = _complete_launcher_environment(capsule)
    for key in canonical:
        missing = canonical.copy()
        del missing[key]
        with pytest.raises(EvidenceBlocked, match="environment identity mismatch"):
            validate(missing)

        changed = canonical.copy()
        changed[key] = "<changed>"
        with pytest.raises(EvidenceBlocked, match="environment identity mismatch"):
            validate(changed)

    extra = canonical | {"EXTRA_KEY": "drift"}
    with pytest.raises(EvidenceBlocked, match="environment identity mismatch"):
        validate(extra)


def _make_path_stat_reader(final_target: Path) -> Callable[[Path, bool], Any]:
    def reader(path: Path, follow: bool) -> Any:
        info = (
            final_target.stat() if path.name == "exe" else (path.stat() if follow else path.lstat())
        )
        return SimpleNamespace(
            st_uid=0,
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_ino=info.st_ino,
        )

    return reader


def test_runtime_identity_rejects_top_level_kev_shadow(tmp_path: Any) -> None:
    capsule = tmp_path / ("a" * 64)
    executable = capsule / "environment/bin/python"
    module = capsule / "payload" / "source" / "kev" / "kev" / "serve.py"
    checkpoint = capsule / "checkpoint"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(capsule, target_is_directory=True)
    final_python = tmp_path / "python3.12"
    final_python.write_text("python", encoding="utf-8")
    final_python.chmod(0o755)
    (executable.parent / "python3").symlink_to(final_python)
    executable.symlink_to("python3")
    (process / "exe").symlink_to(final_python)
    module.write_text("", encoding="utf-8")
    checkpoint.write_text("", encoding="utf-8")
    command = [
        str(executable),
        "-m",
        "kev.serve",
        "--run",
        str(checkpoint),
        "--fallback",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "8182",
    ]
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    top_kev = capsule / "kev"
    top_kev.mkdir()
    (top_kev / "serve.py").write_bytes(b"# shadow")

    def transport(*_args: Any, **_kw: Any) -> Any:
        return HttpResult(
            200,
            {},
            b'{"model":"kev-latest","answers":{},"usage":{"input_tokens":1,"output_tokens":0},"latency_ms":1}',
            1.0,
        )

    with pytest.raises(EvidenceBlocked, match="top-level kev entry"):
        validate_runtime_identity(
            pid=321,
            expected_checkpoint=str(checkpoint),
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            path_stat_reader=_make_path_stat_reader(tmp_path / "python3.12"),
            proc_exe_reader=lambda *_: str(final_python),
            approved_system_roots=(tmp_path,),
            cmdline_reader=lambda *_: "\0".join(command) + "\0",
            environ_reader=lambda *_: (_complete_launcher_environment(capsule), True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=transport,
        )


def test_runtime_identity_rejects_top_level_kev_dot_shadow(tmp_path: Any) -> None:
    capsule = tmp_path / ("a" * 64)
    executable = capsule / "environment/bin/python"
    module = capsule / "payload" / "source" / "kev" / "kev" / "serve.py"
    checkpoint = capsule / "checkpoint"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(capsule, target_is_directory=True)
    final_python = tmp_path / "python3.12"
    final_python.write_text("python", encoding="utf-8")
    final_python.chmod(0o755)
    (executable.parent / "python3").symlink_to(final_python)
    executable.symlink_to("python3")
    (process / "exe").symlink_to(final_python)
    module.write_text("", encoding="utf-8")
    checkpoint.write_text("", encoding="utf-8")
    command = [
        str(executable),
        "-m",
        "kev.serve",
        "--run",
        str(checkpoint),
        "--fallback",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "8182",
    ]
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    top_kev_pyc = capsule / "kev.pyc"
    top_kev_pyc.write_bytes(b"shadow")

    def transport(*_args: Any, **_kw: Any) -> Any:
        return HttpResult(
            200,
            {},
            b'{"model":"kev-latest","answers":{},"usage":{"input_tokens":1,"output_tokens":0},"latency_ms":1}',
            1.0,
        )

    with pytest.raises(EvidenceBlocked, match="top-level kev entry"):
        validate_runtime_identity(
            pid=321,
            expected_checkpoint=str(checkpoint),
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
            path_stat_reader=_make_path_stat_reader(tmp_path / "python3.12"),
            proc_exe_reader=lambda *_: str(final_python),
            approved_system_roots=(tmp_path,),
            cmdline_reader=lambda *_: "\0".join(command) + "\0",
            environ_reader=lambda *_: (_complete_launcher_environment(capsule), True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=transport,
        )
