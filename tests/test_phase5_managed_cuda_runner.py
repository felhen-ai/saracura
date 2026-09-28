from __future__ import annotations

import json
import threading
import time
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.phase5_managed_cuda.runner import (
    CandidateRejected,
    EvidenceBlocked,
    HttpResult,
    ResourceMonitor,
    _nvidia_sample,
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
    from pathlib import Path

    capsule = tmp_path / ("a" * 64)
    executable = capsule / "bin/python"
    module = capsule / "kev/serve.py"
    checkpoint = capsule / "checkpoint"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
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
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
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
    with pytest.raises(EvidenceBlocked):
        validate_runtime_identity(
            pid=321,
            expected_checkpoint=str(checkpoint),
            gateway_url="http://127.0.0.1:8181",
            getuid=lambda: process.stat().st_uid,
            geteuid=lambda: 1000,
            proc_root=proc_root,
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
    executable = capsule / "bin/python"
    module = capsule / "kev/serve.py"
    checkpoint = capsule / "checkpoint"
    proc_root = tmp_path / "proc"
    process = proc_root / "321"
    executable.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    process.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
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
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
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
        "KEV_API_KEY": "private-secret",
    }
    (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    return environment, command, process, proc_root


@pytest.mark.parametrize("failure", ["duplicate_flag", "unknown_kev", "provider_env", "proxy_env"])
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
        module = tmp_path / ("a" * 64) / "kev/serve.py"
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
            cmdline_reader=lambda *_: "\0".join(altered) + "\0",
            environ_reader=lambda *_: (environment, True),
            stat_reader=lambda *_: {"starttime": 7, "utime": 1, "stime": 1},
            socket_reader=lambda *_: 17,
            transport=lambda _url, payload, **_kw: HttpResult(200, {}, _valid_response(payload), 1),
        )


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
