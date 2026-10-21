"""Network-free tests for research/ptbr_bench.py and research/validate_submission.py.

The research scripts are untyped on purpose: they are loaded by name, not imported statically.
Dataset bytes and the model server are self-authored fakes; nothing leaves the process.
"""

from __future__ import annotations

import copy
import http.client
import importlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

RESEARCH = Path(__file__).parents[1] / "research"
DATASET_SHA = "a" * 40
CHECKPOINT_SHA = "b" * 40
NOUL_TASKS = {"faquad_resposta", "olid_ofensivo"}

Behavior = Callable[[dict[str, Any]], bytes | BaseException]


@pytest.fixture
def bench(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.syspath_prepend(str(RESEARCH))
    for name in ("ptbr_bench", "validate_submission"):
        sys.modules.pop(name, None)
    module = importlib.import_module("ptbr_bench")
    monkeypatch.setattr(module, "RETRY_DELAY", 0)
    yield module
    for name in ("ptbr_bench", "validate_submission"):
        sys.modules.pop(name, None)


@pytest.fixture
def validator(bench: Any) -> Any:
    return importlib.import_module("validate_submission")


def make_data(tasks: list[str]) -> dict[str, bytes]:
    data = {}
    for task in tasks:
        rows = []
        for i in range(4):
            if task in NOUL_TASKS:
                gold: Any = i % 2 == 0
                row = {"type": "noul", "instructions": "ofensivo?", "gold": gold}
            else:
                gold = "a" if i % 2 == 0 else "b"
                row = {
                    "type": "choice",
                    "instructions": "qual?",
                    "criteria": {"a": "primeira", "b": "segunda"},
                    "gold": gold,
                }
            rows.append({**row, "text": f"gold={gold} item {i}"})
        data[task] = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    return data


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def perfect(body: dict[str, Any]) -> bytes:
    gold = body["state"].split()[0].removeprefix("gold=")
    q = body["questions"]["q"]
    if q["type"] == "noul":
        answer: dict[str, Any] = {"noul": 0.9 if gold == "True" else 0.1}
    else:
        answer = {"choice": gold}
    return json.dumps({"answers": {"q": answer}}).encode()


def install_server(monkeypatch: pytest.MonkeyPatch, behavior: Behavior) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0) -> FakeResponse:
        body = json.loads(req.data)
        calls.append(body)
        result = behavior(body)
        if isinstance(result, BaseException):
            raise result
        return FakeResponse(result)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


@pytest.fixture
def source(bench: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    data = make_data(bench.TASKS)
    monkeypatch.setattr(bench, "download", lambda revision, task: data[task])
    return data


def cli(tmp_path: Path, *extra: str, name: str = "run") -> list[str]:
    return [
        "--base-url",
        "http://127.0.0.1:9/",
        "--model",
        "alias",
        "--name",
        name,
        "--revision",
        DATASET_SHA,
        "--cache",
        str(tmp_path / "cache"),
        *extra,
    ]


FULL_META = (
    "--checkpoint-id",
    "org/model",
    "--checkpoint-revision",
    CHECKPOINT_SHA,
    "--hardware",
    "test cpu",
)


def run_full(bench: Any, tmp_path: Path, *extra: str) -> tuple[Path, dict[str, Any]]:
    out = tmp_path / "report.json"
    assert bench.main(cli(tmp_path, "--output", str(out), *FULL_META, *extra)) == 0
    return out, json.loads(out.read_text(encoding="utf-8"))


def test_full_run_is_submission_ready_and_validates(
    bench: Any,
    validator: Any,
    source: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = install_server(monkeypatch, perfect)
    out, report = run_full(bench, tmp_path)

    assert len(calls) == 28
    assert report["schema_version"] == 2
    assert report["revision"] == DATASET_SHA
    assert report["run"]["mode"] == "full"
    assert report["run"]["submission_ready"] is True
    assert report["mean_balanced_accuracy"] == 1.0
    assert report["checkpoint"]["revision"] == CHECKPOINT_SHA
    assert report["checkpoint"]["quantization"] == "unknown"
    assert report["evaluator"]["sha256"] == bench.self_sha256()
    assert set(report["items"]) == set(bench.TASKS)  # legacy fields stay
    assert "127.0.0.1" not in out.read_text(encoding="utf-8")
    assert all(len(v) == 4 for v in report["items"].values())

    errors, warnings = validator.validate(report, source=lambda rev, task: source[task])
    assert errors == []
    assert "checkpoint.quantization is unknown" in warnings


def test_validator_cli_uses_fresh_download(
    bench: Any,
    validator: Any,
    source: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_server(monkeypatch, perfect)
    out, _ = run_full(bench, tmp_path)
    assert validator.main([str(out)]) == 0
    assert "VALID" in capsys.readouterr().out


def test_limit_run_is_smoke_and_never_ready(
    bench: Any,
    validator: Any,
    source: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_server(monkeypatch, perfect)
    _, report = run_full(bench, tmp_path, "--limit", "2")

    assert report["run"]["mode"] == "smoke"
    assert report["run"]["partial"] is True
    assert report["run"]["submission_ready"] is False
    assert report["tasks"]["olid_alvo"]["n"] == 2
    assert report["tasks"]["olid_alvo"]["source_n"] == 4
    errors, _ = validator.validate(report, source=lambda rev, task: source[task])
    assert any("partial or smoke" in e for e in errors)
    assert any("source rows" in e for e in errors)


def test_missing_checkpoint_provenance_is_unknown_and_not_ready(
    bench: Any, source: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_server(monkeypatch, perfect)
    out = tmp_path / "r.json"
    assert bench.main(cli(tmp_path, "--output", str(out), "--checkpoint-revision", "main")) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["checkpoint"]["id"] == "unknown"
    assert report["run"]["submission_ready"] is False
    assert any("checkpoint id" in i for i in report["run"]["readiness_issues"])
    assert any("revision" in i for i in report["run"]["readiness_issues"])


def test_digest_revision_is_accepted_as_immutable(bench: Any) -> None:
    assert bench.is_immutable_revision("sha256:" + "c" * 64)
    assert bench.is_immutable_revision("d" * 40)
    for bad in ("main", "latest", "unknown", "", "sha256:abc", "D" * 40, None):
        assert not bench.is_immutable_revision(bad)


@pytest.mark.parametrize(
    ("behavior", "code"),
    [
        (lambda body: b"not json", "bad_response"),
        (lambda body: b"[1, 2]", "bad_answer"),
        (lambda body: b'{"answers": {"q": null}}', "bad_answer"),
        (lambda body: b'{"answers": {"q": {}}}', "no_answer"),
        (lambda body: b'{"answers": {"q": {"choice": "zzz", "noul": "x"}}}', None),
        (lambda body: http.client.RemoteDisconnected("gone"), "transport"),
        (lambda body: urllib.error.URLError("refused"), "transport"),
        (
            lambda body: urllib.error.HTTPError("u", 503, "x", None, None),  # type: ignore[arg-type]
            "http_server_error",
        ),
        (
            lambda body: urllib.error.HTTPError("u", 400, "x", None, None),  # type: ignore[arg-type]
            "http_client_error",
        ),
    ],
)
def test_bad_server_answers_count_wrong_with_bounded_codes(
    bench: Any,
    source: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    behavior: Behavior,
    code: str | None,
) -> None:
    install_server(monkeypatch, behavior)
    _, report = run_full(bench, tmp_path)

    assert report["run"]["submission_ready"] is False
    for task in bench.TASKS:
        assert report["tasks"][task]["failed"] == 4
        assert all(p is None for _, p in report["items"][task])
        assert set(report["errors"][task]) <= set(bench.ERROR_CODES)
        assert report["tasks"][task]["balanced_accuracy"] == 0.0
    if code:
        assert set(report["errors"]["olid_alvo"]) == {code}
    assert any("failed items" in i for i in report["run"]["readiness_issues"])


def test_invalid_choice_and_probability_answers_are_wrong(bench: Any) -> None:
    q = {"type": "choice", "criteria": {"a": "x", "b": "y"}}
    assert bench.pick({"choice": "b"}, q) == ("b", None)
    assert bench.pick({"choice": "zzz"}, q) == (None, "invalid_choice")
    assert bench.pick({"choice": 1}, q) == (None, "invalid_choice")
    assert bench.pick({"probabilities": {"a": 0.2, "b": 0.8}}, q) == ("b", None)
    assert bench.pick({"probabilities": {"z": 1.0}}, q) == (None, "invalid_choice")
    assert bench.pick({"probabilities": {"a": "high"}}, q) == (None, "invalid_choice")
    assert bench.pick({"probabilities": {}}, q) == (None, "no_prediction")
    n = {"type": "noul"}
    assert bench.pick({"noul": 0.7}, n) == (True, None)
    assert bench.pick({"probability": 0.2}, n) == (False, None)
    assert bench.pick({"noul": "yes"}, n) == (None, "invalid_noul")
    assert bench.pick({"noul": float("nan")}, n) == (None, "invalid_noul")


@pytest.mark.parametrize(
    "bad",
    [
        ["--limit", "-1"],
        ["--limit", "two"],
        ["--timeout", "0"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--timeout", "x"],
        ["--name", "../escape"],
        ["--name", "dir/file"],
        ["--name", ".hidden"],
        ["--checkpoint-id", "http://host/model"],
        ["--model", "https://private.example/v1"],
        ["--hardware", ""],
        ["--revision", "../x"],
        ["--base-url", "ftp://host"],
        ["--base-url", "http://user:pw@host"],
    ],
)
def test_invalid_cli_values_are_rejected_before_any_work(
    bench: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: list[str]
) -> None:
    def forbidden(*a: object, **k: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    args = [
        "--base-url",
        "http://127.0.0.1:9",
        "--name",
        "ok",
        "--cache",
        str(tmp_path / "c"),
    ]
    flag = bad[0]
    args = [a for i, a in enumerate(args) if not (a == flag or (i and args[i - 1] == flag))] + bad
    with pytest.raises(SystemExit) as exc:
        bench.main(args)
    assert exc.value.code == 2
    assert not (tmp_path / "c").exists()


def test_model_alias_accepts_empty_and_public_ids_but_never_echoes_urls(
    bench: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = bench.build_parser()
    base = ["--base-url", "http://127.0.0.1:9", "--name", "ok"]
    assert parser.parse_args(base).model == ""
    assert parser.parse_args([*base, "--model", "  "]).model == ""
    assert parser.parse_args([*base, "--model", "org/model-1.5"]).model == "org/model-1.5"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--model", "https://private.example/v1"])
    assert "private.example" not in capsys.readouterr().err


def test_existing_output_is_rejected_before_network_and_left_untouched(
    bench: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*a: object, **k: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    target = tmp_path / "taken.json"
    target.write_text("published")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        bench.main(cli(tmp_path, "--output", str(target)))
    assert target.read_text() == "published"

    monkeypatch.chdir(tmp_path)
    (tmp_path / "run.json").write_text("older")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        bench.main(cli(tmp_path))
    with pytest.raises(SystemExit, match=r"not writable|does not exist"):
        bench.main(cli(tmp_path, "--output", str(tmp_path / "missing" / "r.json")))


def test_default_output_is_name_json_and_no_temp_files_remain(
    bench: Any, source: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_server(monkeypatch, perfect)
    monkeypatch.chdir(tmp_path)
    assert bench.main(cli(tmp_path, name="legacy-name")) == 0
    assert (tmp_path / "legacy-name.json").is_file()
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["legacy-name.json"]


def test_publish_exclusive_never_replaces(bench: Any, tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    bench.publish_exclusive(path, b"one")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        bench.publish_exclusive(path, b"two")
    assert path.read_bytes() == b"one"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_revision_is_resolved_once_to_an_immutable_commit(
    bench: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake(req: Any, timeout: float = 0) -> FakeResponse:
        seen.append(req if isinstance(req, str) else req.full_url)
        return FakeResponse(json.dumps({"sha": DATASET_SHA}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert bench.resolve_revision("main") == DATASET_SHA
    assert seen == [f"https://huggingface.co/api/datasets/{bench.REPO}/revision/main"]
    assert bench.resolve_revision("c" * 40) == "c" * 40  # already immutable: no request
    assert len(seen) == 1

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResponse(b'{"sha": "main"}'))
    with pytest.raises(SystemExit, match="did not return a commit"):
        bench.resolve_revision("main")


def test_certificate_failure_gets_actionable_help_without_disabling_tls(
    bench: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*a: object, **k: object) -> None:
        raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer"))

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(SystemExit) as exc:
        bench.download(DATASET_SHA, "olid_alvo")
    assert "uv run --python 3.12" in str(exc.value)
    assert "Do not disable" in str(exc.value)


def test_malformed_dataset_aborts_before_inference(
    bench: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = make_data(bench.TASKS)
    data["olid_alvo"] = (
        b'{"type": "choice", "text": "t", "instructions": "i", "gold": "z",'
        b' "criteria": {"a": "x"}}\n'
    )
    monkeypatch.setattr(bench, "download", lambda revision, task: data[task])
    calls = install_server(monkeypatch, perfect)
    with pytest.raises(SystemExit, match="invalid dataset"):
        bench.main(cli(tmp_path, "--output", str(tmp_path / "r.json")))
    assert calls == []
    assert not (tmp_path / "r.json").exists()
    data["olid_alvo"] = b""
    with pytest.raises(ValueError, match="no rows"):
        bench.parse_rows(b"\n", "olid_alvo")


def test_shuffle_protocol_is_unchanged(bench: Any) -> None:
    assert bench.SEED == 20261009
    assert bench.MAX_CHARS == 3500
    assert sorted(bench.TASKS) == bench.TASKS and len(bench.TASKS) == 7
    row = {"type": "choice", "instructions": "i", "criteria": {k: k for k in "abcdef"}}
    first = bench.question(row, bench.random.Random(bench.SEED))
    again = bench.question(row, bench.random.Random(bench.SEED))
    assert list(first["criteria"]) == list(again["criteria"])
    assert set(first["criteria"]) == set("abcdef")


def test_long_text_is_cut_at_protocol_length(
    bench: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = make_data(bench.TASKS)
    for task, raw in data.items():
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        for row in rows:
            row["text"] = f"gold={row['gold']} " + "x" * 9000
        data[task] = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    monkeypatch.setattr(bench, "download", lambda revision, task: data[task])
    calls = install_server(monkeypatch, perfect)
    bench.main(cli(tmp_path, "--limit", "1", "--output", str(tmp_path / "r.json")))
    assert {len(c["state"]) for c in calls} == {3500}


# --- validator: tampering --------------------------------------------------


@pytest.fixture
def good(
    bench: Any, source: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    install_server(monkeypatch, perfect)
    _, report = run_full(bench, tmp_path)
    return report


def check(validator: Any, source: dict[str, bytes], report: dict[str, Any]) -> list[str]:
    errors: list[str] = validator.validate(report, source=lambda rev, task: source[task])[0]
    return errors


def tamper_gold(r: dict[str, Any]) -> None:
    r["items"]["olid_alvo"][0][0] = "b" if r["items"]["olid_alvo"][0][0] == "a" else "a"


def tamper_pred_illegal(r: dict[str, Any]) -> None:
    r["items"]["olid_alvo"][0][1] = "not-an-option"


def tamper_score(r: dict[str, Any]) -> None:
    r["tasks"]["scielo_area"]["balanced_accuracy"] = 0.5


def tamper_mean(r: dict[str, Any]) -> None:
    r["mean_balanced_accuracy"] = 0.99


def tamper_ci(r: dict[str, Any]) -> None:
    r["mean_ci95"] = [0.1, 0.2]


def tamper_truncate(r: dict[str, Any]) -> None:
    del r["items"]["camara_tema"][-1]
    del r["errors"]["camara_tema"][-1]


def tamper_failure_hidden(r: dict[str, Any]) -> None:
    r["items"]["olid_alvo"][1][1] = None  # no error code recorded for the missing answer


def tamper_checkpoint(r: dict[str, Any]) -> None:
    r["checkpoint"]["revision"] = "main"


def tamper_digest(r: dict[str, Any]) -> None:
    r["dataset_files"]["olid_alvo"]["sha256"] = "0" * 64


def tamper_revision(r: dict[str, Any]) -> None:
    r["revision"] = "main"


def tamper_protocol(r: dict[str, Any]) -> None:
    r["protocol"]["max_chars"] = 100


def tamper_mode(r: dict[str, Any]) -> None:
    r["run"]["mode"] = "smoke"


def tamper_missing_task(r: dict[str, Any]) -> None:
    del r["items"]["olid_alvo"]


def tamper_url(r: dict[str, Any]) -> None:
    r["checkpoint"]["server"] = "http://10.0.0.5:8000"


def tamper_not_ready(r: dict[str, Any]) -> None:
    r["run"]["submission_ready"] = False


def tamper_ids(r: dict[str, Any]) -> None:
    r["checkpoint"]["id"] = "unknown"


@pytest.mark.parametrize(
    "tamper",
    [
        tamper_gold,
        tamper_pred_illegal,
        tamper_score,
        tamper_mean,
        tamper_ci,
        tamper_truncate,
        tamper_failure_hidden,
        tamper_checkpoint,
        tamper_digest,
        tamper_revision,
        tamper_protocol,
        tamper_mode,
        tamper_missing_task,
        tamper_url,
        tamper_not_ready,
        tamper_ids,
    ],
)
def test_validator_rejects_tampered_reports(
    validator: Any,
    source: dict[str, bytes],
    good: dict[str, Any],
    tamper: Callable[[dict[str, Any]], None],
) -> None:
    assert check(validator, source, good) == []
    broken = copy.deepcopy(good)
    tamper(broken)
    assert check(validator, source, broken) != []


def test_validator_rejects_gold_that_disagrees_with_source_even_if_self_consistent(
    validator: Any, bench: Any, source: dict[str, bytes], good: dict[str, Any]
) -> None:
    forged = copy.deepcopy(good)
    for pair in forged["items"]["olid_alvo"]:
        pair[0] = "a"
        pair[1] = "a"
    forged["tasks"]["olid_alvo"]["balanced_accuracy"] = 1.0
    assert check(validator, source, forged) != []


def test_validator_authority_is_the_source_not_the_report(
    validator: Any, bench: Any, source: dict[str, bytes], good: dict[str, Any]
) -> None:
    changed = dict(source)
    changed["olid_alvo"] = source["olid_alvo"].replace(b"gold=a", b"gold=b", 1)
    errors = validator.validate(good, source=lambda rev, task: changed[task])[0]
    assert any("dataset_files" in e for e in errors)


def test_validator_rejects_legacy_and_non_reports(validator: Any, source: dict[str, bytes]) -> None:
    legacy = {"name": "old", "tasks": {}, "items": {}, "mean_balanced_accuracy": 0.7}
    errors = check(validator, source, legacy)
    assert errors and "legacy" in errors[0]
    assert check(validator, source, ["x"])  # type: ignore[arg-type]


def test_validator_flags_unknown_evaluator_and_never_writes(
    bench: Any, validator: Any, source: dict[str, bytes], good: dict[str, Any], tmp_path: Path
) -> None:
    other = copy.deepcopy(good)
    other["evaluator"]["sha256"] = "e" * 64
    errors, warnings = validator.validate(other, source=lambda rev, task: source[task])
    assert errors == []
    assert any("evaluator script differs" in w for w in warnings)
    before = sorted(tmp_path.iterdir())
    validator.validate(good, source=lambda rev, task: source[task])
    assert sorted(tmp_path.iterdir()) == before


def test_validator_cli_rejects_missing_oversized_and_nonfinite_files(
    validator: Any, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="not a file"):
        validator.load_report(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"mean_balanced_accuracy": NaN}')
    with pytest.raises(SystemExit, match="not valid JSON"):
        validator.load_report(bad)
