"""Network-free tests for research/llm_bridge.py.

The bridge is a research script loaded by name. Tokenizer, generator, torch and transformers are
self-authored fakes; no weights, no benchmark data and no sockets beyond loopback.
"""

from __future__ import annotations

import hashlib
import http.client
import http.server
import importlib
import json
import random
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

RESEARCH = Path(__file__).parents[1] / "research"
REVISION = "c" * 40
MODEL = "org/model"


@pytest.fixture
def mod(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.syspath_prepend(str(RESEARCH))
    sys.modules.pop("llm_bridge", None)
    heavy_before = {n for n in ("torch", "transformers") if n in sys.modules}
    module = importlib.import_module("llm_bridge")
    assert {n for n in ("torch", "transformers") if n in sys.modules} == heavy_before
    yield module
    sys.modules.pop("llm_bridge", None)


class FakeBackend:
    """Tokenizer = one token per character; generator replies from a script."""

    def __init__(self, replies: list[str] | None = None, context: int = 100_000) -> None:
        self.replies = list(replies or [])
        self.context = context
        self.prompts: list[str] = []
        self.generated = 0
        self.fail: BaseException | None = None
        self.on_generate: Any = None

    def encode(self, text: str) -> list[int]:
        self.prompts.append(text)
        return [1] * len(text)

    def generate(self, ids: list[int], deadline: float) -> str:
        self.generated += 1
        if self.on_generate:
            self.on_generate(deadline)
        if self.fail:
            raise self.fail
        return self.replies.pop(0)


def make_bridge(mod: Any, backend: FakeBackend, clock: Any = None, timeout: float = 90.0) -> Any:
    evidence = {"schema_version": 1, "model": {"id": MODEL}}
    kwargs = {"clock": clock} if clock else {}
    return mod.Bridge(backend, MODEL, evidence, timeout, **kwargs)


def body(question: dict[str, Any], state: str = "texto", model: str = MODEL) -> bytes:
    return json.dumps({"model": model, "state": state, "questions": {"q": question}}).encode()


def choice(n: int, shuffled: bool = False) -> dict[str, Any]:
    keys = [f"k{i:02d}" for i in range(n)]
    if shuffled:
        random.Random(3).shuffle(keys)
    return {"type": "choice", "instructions": "qual?", "criteria": {k: f"desc {k}" for k in keys}}


NOUL = {"type": "noul", "instructions": "ofensivo?"}


# C1: pure core


def test_letters_map_in_received_order_for_24_options(mod: Any) -> None:
    q = choice(24, shuffled=True)
    req = mod.parse_request(body(q), MODEL)
    question = req.questions[0]
    assert list(question.keys) == list(q["criteria"])  # not sorted
    text = mod.render_user_message("s", question)
    lines = [ln for ln in text.splitlines() if ln[1:3] == ") "]
    assert [ln[0] for ln in lines] == list(mod.LETTERS[:24])
    for i, key in enumerate(q["criteria"]):
        assert lines[i] == f"{mod.LETTERS[i]}) desc {key}"
    assert mod.answer_for(question, 23) == {"choice": list(q["criteria"])[23]}


def test_prompt_preserves_state_and_question_exactly(mod: Any) -> None:
    state = '  Olá — açãó \n\n tab\t e "aspas" {"json": [1]} \r\n  final  '
    q = {
        "type": "choice",
        "instructions": " Qual é\ta área? ",
        "criteria": {"b": "Saúdé ", "a": " Educação"},
    }
    question = mod.parse_request(body(q, state), MODEL).questions[0]
    text = mod.render_user_message(state, question)
    assert text.startswith(state + "\n\n")
    assert q["instructions"] in text
    assert "A) Saúdé \nB)  Educação\n" in text  # b before a: insertion order, strings untouched


def test_noul_is_fixed_no_yes_and_discrete(mod: Any) -> None:
    question = mod.parse_request(body(NOUL), MODEL).questions[0]
    text = mod.render_user_message("s", question)
    assert "A) Não\nB) Sim" in text
    assert mod.answer_for(question, 0) == {"noul": 0}
    assert mod.answer_for(question, 1) == {"noul": 1}
    assert mod.parse_letter("C", 2) is None


@pytest.mark.parametrize(
    ("text", "count", "expected"),
    [
        ("A", 3, 0),
        ("  C\n", 3, 2),
        ("Z", 26, 25),
        ("D", 3, None),
        ("a", 3, None),
        ("A.", 3, None),
        ("AB", 3, None),
        ("A\nB", 3, None),
        ("", 3, None),
        ("Resposta: A", 3, None),
        ("Á", 3, None),
        ("1", 3, None),
    ],
)
def test_parser_is_strict(mod: Any, text: str, count: int, expected: int | None) -> None:
    assert mod.parse_letter(text, count) == expected


# C2: closed validation


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        b'{"model": "org/model", "state": "x"}',
        json.dumps({"model": MODEL, "state": "x", "questions": {"q": NOUL}, "extra": 1}).encode(),
        json.dumps({"model": MODEL, "state": 1, "questions": {"q": NOUL}}).encode(),
        json.dumps({"model": MODEL, "state": "x", "questions": {}}).encode(),
        json.dumps(
            {"model": MODEL, "state": "x", "questions": {f"q{i}": NOUL for i in range(17)}}
        ).encode(),
        body({"type": "noul", "instructions": "x", "criteria": {}}),
        body({"type": "choice", "instructions": "x"}),
        body({"type": "choice", "instructions": "x", "criteria": {"a": "only"}}),
        body({"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": "t"}}),
        body(
            {"type": "choice", "instructions": "x", "criteria": {f"k{i}": "d" for i in range(27)}}
        ),
        body({"type": "choice", "instructions": "x", "criteria": {"a": "d", "b": "d"}, "extra": 1}),
        body({"type": "scale", "instructions": "x"}),
        body({"type": "noul", "instructions": 3}),
        b'{"model": "org/model", "model": "org/model", "state": "x", "questions": {"q": {}}}',
    ],
)
def test_malformed_requests_rejected_before_inference(mod: Any, raw: bytes) -> None:
    backend = FakeBackend()
    status, payload = make_bridge(mod, backend).handle(raw)
    assert status == 400
    assert set(payload) == {"error"}
    assert backend.prompts == [] and backend.generated == 0


def test_wrong_model_rejected_before_inference(mod: Any) -> None:
    backend = FakeBackend()
    status, payload = make_bridge(mod, backend).handle(body(NOUL, model="other/model"))
    assert (status, payload) == (400, {"error": "unknown_model"})
    assert backend.prompts == []


def test_oversize_body_rejected(mod: Any) -> None:
    backend = FakeBackend()
    status, payload = make_bridge(mod, backend).handle(b" " * (mod.MAX_BODY_BYTES + 1))
    assert (status, payload) == (413, {"error": "too_large"})
    assert backend.prompts == []


def test_choice_bounds_2_and_26_accepted(mod: Any) -> None:
    for n in (2, 26):
        assert len(mod.parse_request(body(choice(n)), MODEL).questions[0].keys) == n


# C7: failures, deadline, lock


def test_valid_choice_and_noul_answers(mod: Any) -> None:
    bridge = make_bridge(mod, FakeBackend(["C", "B"]))
    assert bridge.handle(body(choice(5))) == (200, {"answers": {"q": {"choice": "k02"}}})
    assert bridge.handle(body(NOUL)) == (200, {"answers": {"q": {"noul": 1}}})
    assert bridge.health()["counters"] == {"invalid_output": 0, "deadline": 0, "overbudget": 0}


@pytest.mark.parametrize(
    "reply", ["", "A.", "AB", "Resposta: A", "a", "F", "```A```", "<think>x</think>A"]
)
def test_invalid_output_is_empty_answer_without_retry(mod: Any, reply: str) -> None:
    backend = FakeBackend([reply, "A"])
    bridge = make_bridge(mod, backend)
    status, payload = bridge.handle(body(choice(4)))
    assert (status, payload) == (200, {"answers": {"q": {}}, "error": "invalid_output"})
    assert backend.generated == 1  # no repair, no retry, no fallback
    assert bridge.health()["counters"]["invalid_output"] == 1


def test_multi_question_partial_invalid(mod: Any) -> None:
    raw = json.dumps(
        {"model": MODEL, "state": "s", "questions": {"a": NOUL, "b": choice(3)}}
    ).encode()
    status, payload = make_bridge(mod, FakeBackend(["B", "??"])).handle(raw)
    assert status == 200
    assert payload == {"answers": {"a": {"noul": 1}, "b": {}}, "error": "invalid_output"}


def test_deadline_during_generation_blanks_all(mod: Any) -> None:
    backend = FakeBackend()
    backend.fail = mod.DeadlineExceeded()
    bridge = make_bridge(mod, backend)
    raw = json.dumps({"model": MODEL, "state": "s", "questions": {"a": NOUL, "b": NOUL}}).encode()
    assert bridge.handle(raw) == (200, {"answers": {"a": {}, "b": {}}, "error": "deadline"})
    assert backend.generated == 1
    assert bridge.health()["counters"]["deadline"] == 1


def test_deadline_wins_over_valid_partial_letter(mod: Any) -> None:
    now = [0.0]
    backend = FakeBackend(["A", "A"])
    backend.on_generate = lambda deadline: now.__setitem__(
        0, deadline + 0.1
    )  # slow generator, no sleeping
    bridge = make_bridge(mod, backend, clock=lambda: now[0], timeout=5)
    raw = json.dumps({"model": MODEL, "state": "s", "questions": {"a": NOUL, "b": NOUL}}).encode()
    assert bridge.handle(raw) == (200, {"answers": {"a": {}, "b": {}}, "error": "deadline"})
    assert (
        backend.generated == 1
    )  # the valid letter was discarded and the second question never ran


def test_deadline_is_whole_request_not_per_question(mod: Any) -> None:
    now = [0.0]
    backend = FakeBackend(["A", "A"])
    backend.on_generate = lambda deadline: now.__setitem__(0, now[0] + 3)
    bridge = make_bridge(mod, backend, clock=lambda: now[0], timeout=5)
    raw = json.dumps({"model": MODEL, "state": "s", "questions": {"a": NOUL, "b": NOUL}}).encode()
    status, payload = bridge.handle(raw)
    assert (status, payload["error"]) == (200, "deadline")  # 3 + 3 > 5 although each is under 5


def test_busy_returns_429_without_second_generation(mod: Any) -> None:
    backend = FakeBackend(["A", "A"])
    bridge = make_bridge(mod, backend)
    seen: list[tuple[int, dict[str, Any]]] = []
    backend.on_generate = lambda deadline: seen.append(bridge.handle(body(NOUL)))
    assert bridge.handle(body(NOUL)) == (200, {"answers": {"q": {"noul": 0}}})
    assert seen == [(429, {"error": "busy"})]
    assert backend.generated == 1
    assert bridge.handle(body(NOUL))[0] == 200  # lock released afterwards


def test_runtime_failure_is_bounded_500(mod: Any) -> None:
    backend = FakeBackend()
    backend.fail = RuntimeError("secret prompt text and /private/path")
    status, payload = make_bridge(mod, backend).handle(body(NOUL))
    assert (status, payload) == (500, {"error": "internal"})


# C3: context


def test_context_overflow_rejected_before_generation(mod: Any) -> None:
    backend = FakeBackend(["A"])
    bridge = make_bridge(mod, backend)
    prompt_len = len(
        mod.render_user_message("s", mod.parse_request(body(NOUL, "s"), MODEL).questions[0])
    )
    backend.context = prompt_len + mod.MAX_NEW_TOKENS - 1
    assert bridge.handle(body(NOUL, "s")) == (422, {"error": "context_overflow"})
    assert backend.generated == 0
    assert bridge.health()["counters"]["overbudget"] == 1
    backend.context = prompt_len + mod.MAX_NEW_TOKENS  # exactly at the limit still fits
    assert bridge.handle(body(NOUL, "s"))[0] == 200


def test_overflow_in_any_question_prevents_all_generation(mod: Any) -> None:
    backend = FakeBackend(["A", "A"], context=400)
    raw = json.dumps(
        {"model": MODEL, "state": "s", "questions": {"a": NOUL, "b": choice(26)}}
    ).encode()
    assert make_bridge(mod, backend).handle(raw)[0] == 422
    assert backend.generated == 0


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.device = "cpu"

    def __getitem__(self, i: int) -> list[int]:
        return self.rows[i]


def fake_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="0.0-fake",
        float32="f32",
        float16="f16",
        bfloat16="bf16",
        long="long",
        bool="bool",
        tensor=lambda rows, dtype=None, device=None: FakeTensor(rows),
        ones_like=lambda t: t,
        full=lambda shape, value, dtype=None, device=None: ("full", shape, value),
        inference_mode=lambda: __import__("contextlib").nullcontext(),
        set_num_threads=lambda n: calls.setdefault("threads", n),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )


calls: dict[str, Any] = {}


class FakeTokenizer:
    chat_template = "{{ messages[0].content }}"
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, template: str | None = None) -> None:
        if template is not None:
            self.chat_template = template
        self.rendered: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, repo: str, **kw: Any) -> FakeTokenizer:
        calls["tokenizer"] = (repo, kw)
        return cls(calls.get("template"))

    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.rendered.append({"messages": messages, **kw})
        return "<s>" + messages[0]["content"]

    def __call__(self, text: str, **kw: Any) -> dict[str, list[int]]:
        calls["tokenize"] = kw
        return {"input_ids": list(range(len(text)))}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        calls["decode"] = skip_special_tokens
        return "B"


class FakeModel:
    @classmethod
    def from_pretrained(cls, repo: str, **kw: Any) -> FakeModel:
        calls["model"] = (repo, kw)
        return cls()

    def to(self, device: str) -> FakeModel:
        calls["device"] = device
        return self

    def eval(self) -> None:
        pass

    def generate(self, input_ids: FakeTensor, **kw: Any) -> list[list[int]]:
        calls["generate"] = kw
        for criteria in kw["stopping_criteria"]:
            calls["stop"] = criteria(input_ids, None)
        return [[*input_ids.rows[0], 9]]


def fake_importer(name: str) -> Any:
    if name == "torch":
        return fake_torch()
    if name == "transformers":
        return types.SimpleNamespace(
            __version__="9.9",
            AutoTokenizer=FakeTokenizer,
            AutoModelForCausalLM=FakeModel,
            AutoModelForImageTextToText=FakeModel,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        )
    raise AssertionError(f"unexpected import {name}")


def config(mod: Any, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "context": 4096,
        "threads": 3,
        "template_kwargs": {"enable_thinking": False},
    }
    return mod.Config(**{**base, **kw})


def test_loader_is_pinned_local_only_and_never_trusts_remote_code(mod: Any) -> None:
    calls.clear()
    backend, info = mod.load_transformers_backend(config(mod), fake_importer)
    for key in ("tokenizer", "model"):
        repo, kw = calls[key]
        assert repo == MODEL
        assert kw["revision"] == REVISION
        assert kw["local_files_only"] is True
        assert kw["trust_remote_code"] is False
    assert calls["model"][1]["attn_implementation"] == "eager"
    assert calls["model"][1]["dtype"] == "f32"
    assert calls["threads"] == 3 and calls["device"] == "cpu"
    assert info["dtype"] == "float32" and info["device"] == "cpu" and info["attention"] == "eager"
    assert info["torch"] == "0.0-fake" and info["transformers"] == "9.9"
    assert info["template_kwargs"] == {"enable_thinking": False}
    assert (
        info["chat_template_sha256"]
        == hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest()
    )
    assert backend.context == 4096


def test_encode_uses_official_template_nonthinking_kwargs_and_no_truncation(mod: Any) -> None:
    calls.clear()
    backend, _ = mod.load_transformers_backend(config(mod), fake_importer)
    ids = backend.encode("olá")
    assert ids == list(range(len("<s>olá")))
    rendered = backend.tokenizer.rendered[-1]
    assert rendered["messages"] == [{"role": "user", "content": "olá"}]
    assert rendered["tokenize"] is False and rendered["add_generation_prompt"] is True
    assert rendered["enable_thinking"] is False
    assert calls["tokenize"] == {"add_special_tokens": False, "truncation": False}


def test_generate_is_greedy_capped_and_decodes_without_special_tokens(mod: Any) -> None:
    calls.clear()
    backend, _ = mod.load_transformers_backend(config(mod), fake_importer)
    assert backend.generate([1, 2, 3], deadline=float("inf")) == "B"
    kw = calls["generate"]
    assert kw["max_new_tokens"] == 32 and kw["do_sample"] is False
    assert calls["decode"] is True
    assert calls["stop"] == ("full", (1,), False)
    with pytest.raises(mod.DeadlineExceeded):
        backend.generate(
            [1, 2, 3], deadline=0.0
        )  # already expired: stopping criterion flips the flag


@pytest.mark.parametrize(
    ("device", "default"), [("cpu", "float32"), ("mps", "float16"), ("cuda", "bfloat16")]
)
def test_dtype_defaults_per_device_and_rejects_unsupported(
    mod: Any, device: str, default: str
) -> None:
    assert mod.resolve_dtype(device, None) == default
    with pytest.raises(mod.BridgeError):
        mod.resolve_dtype("cpu", "float16")


def test_unavailable_accelerator_fails_at_startup(mod: Any) -> None:
    for device in ("cuda", "mps"):
        with pytest.raises(mod.BridgeError):
            mod.load_transformers_backend(config(mod, device=device), fake_importer)


def test_template_without_official_string_is_rejected(mod: Any) -> None:
    calls.clear()
    calls["template"] = None
    FakeTokenizer.chat_template = None  # type: ignore[assignment]
    try:
        with pytest.raises(mod.BridgeError):
            mod.load_transformers_backend(config(mod), fake_importer)
    finally:
        FakeTokenizer.chat_template = "{{ messages[0].content }}"


def test_time_dependent_template_requires_frozen_clock(mod: Any) -> None:
    calls.clear()
    calls["template"] = "{{ strftime_now('%Y-%m-%d') }}{{ messages[0].content }}"
    clock_module = types.SimpleNamespace(datetime=__import__("datetime").datetime)

    def importer(name: str) -> Any:
        return (
            clock_module
            if name == "transformers.utils.chat_template_utils"
            else fake_importer(name)
        )

    def render(self: FakeTokenizer, messages: list[dict[str, str]], **kw: Any) -> str:
        if "chat_template" in kw:  # the startup probe
            return clock_module.datetime.now().strftime("%Y-%m-%d")
        return "<s>" + messages[0]["content"]

    original = FakeTokenizer.apply_chat_template
    FakeTokenizer.apply_chat_template = render  # type: ignore[method-assign,assignment]
    try:
        backend, info = mod.load_transformers_backend(config(mod), importer)
    finally:
        FakeTokenizer.apply_chat_template = original  # type: ignore[method-assign]
        calls.pop("template")
    assert info["strftime_now_frozen"] is True and info["template_date"] == "2026-10-21"
    assert (
        clock_module.datetime.now().year != 2026
        or clock_module.datetime is __import__("datetime").datetime
    )
    with mod.frozen_clock(clock_module, "2026-10-21"):
        assert clock_module.datetime.now().strftime("%Y-%m-%d") == "2026-10-21"
    assert backend.clock_module is clock_module


def test_time_dependent_template_rejected_when_not_deterministic(mod: Any) -> None:
    calls.clear()
    calls["template"] = "{{ strftime_now('%Y') }}"
    try:
        # transformers without a patchable datetime
        with pytest.raises(mod.BridgeError):
            mod.load_transformers_backend(
                config(mod),
                lambda n: (
                    types.SimpleNamespace()
                    if n.endswith("chat_template_utils")
                    else fake_importer(n)
                ),
            )
        # a helper that ignores the frozen clock fails the startup probe
        clock_module = types.SimpleNamespace(datetime=__import__("datetime").datetime)
        with pytest.raises(mod.BridgeError):
            mod.load_transformers_backend(
                config(mod),
                lambda n: clock_module if n.endswith("chat_template_utils") else fake_importer(n),
            )
    finally:
        calls.pop("template")


def test_template_kwargs_validation(mod: Any) -> None:
    assert mod.parse_template_kwargs('{"enable_thinking": false}') == {"enable_thinking": False}
    for bad in ("[]", "x", '{"messages": 1}', '{"a": {"b": 1}}', '{"a": 1, "a": 2}'):
        with pytest.raises(mod.BridgeError):
            mod.parse_template_kwargs(bad)


def test_cli_requires_pinned_revision_and_explicit_template_kwargs(mod: Any) -> None:
    ap = mod.build_parser()
    ok = [
        "--model",
        MODEL,
        "--device",
        "cpu",
        "--port",
        "1",
        "--context",
        "9",
        "--chat-template-kwargs",
        "{}",
    ]
    assert ap.parse_args([*ok, "--revision", REVISION]).revision == REVISION
    for argv in ([*ok, "--revision", "main"], [*ok], ["--revision", REVISION, "--model", MODEL]):
        with pytest.raises(SystemExit):
            ap.parse_args(argv)


# C4: HTTP, health, evidence


def request(
    server: Any,
    method: str,
    path: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request(method, path, body=data, headers=headers or {})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw)


@pytest.fixture
def served(mod: Any) -> Iterator[tuple[Any, FakeBackend, Any]]:
    backend = FakeBackend(["B"] * 8)
    bridge = make_bridge(mod, backend)
    server = mod.make_server(bridge, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, backend, bridge
    server.shutdown()
    server.server_close()
    thread.join(5)


def test_http_binds_loopback_only(served: Any) -> None:
    assert served[0].server_address[0] == "127.0.0.1"


def test_http_health_and_systemone_shape(served: Any, mod: Any) -> None:
    server, _, bridge = served
    status, health = request(server, "GET", "/health")
    assert status == 200
    assert health["evidence"]["model"]["id"] == MODEL
    assert health["evidence_sha256"] == bridge.evidence_sha256
    assert set(health["counters"]) == {"invalid_output", "deadline", "overbudget"}
    status, payload = request(
        server, "POST", "/v1/systemone", body(NOUL), {"content-type": "application/json"}
    )
    assert (status, payload) == (200, {"answers": {"q": {"noul": 1}}})
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "POST", "/nope", b"{}")[0] == 404


def test_http_bounded_failures(served: Any, mod: Any) -> None:
    server, backend, _ = served
    assert request(server, "POST", "/v1/systemone", b"nope")[0] == 400
    assert request(server, "POST", "/v1/systemone", body(NOUL, model="x/y"))[0] == 400
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/v1/systemone")
    conn.putheader("content-length", str(mod.MAX_BODY_BYTES + 1))
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413 and json.loads(resp.read()) == {"error": "too_large"}
    conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/v1/systemone")
    conn.endheaders()  # no content-length
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()
    assert backend.generated == 0


def test_http_logs_nothing(served: Any, capsys: pytest.CaptureFixture[str]) -> None:
    request(served[0], "POST", "/v1/systemone", body(NOUL, "SEGREDO-NO-ESTADO"))
    request(served[0], "POST", "/v1/systemone", b"SEGREDO-CRU")
    captured = capsys.readouterr()
    assert "SEGREDO" not in captured.out + captured.err


def test_evaluator_pick_accepts_bridge_answers(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(RESEARCH))
    sys.modules.pop("ptbr_bench", None)
    bench = importlib.import_module("ptbr_bench")
    try:
        backend = FakeBackend(["D", "B", "Z"])
        bridge = make_bridge(mod, backend)
        q = choice(5, shuffled=True)
        status, payload = bridge.handle(body(q))
        assert bench.pick(payload["answers"]["q"], q) == (list(q["criteria"])[3], None)
        status, payload = bridge.handle(body(NOUL))
        assert bench.pick(payload["answers"]["q"], NOUL) == (True, None)
        status, payload = bridge.handle(body(NOUL))  # "Z" is invalid for 2 options
        assert status == 200 and bench.pick(payload["answers"]["q"], NOUL) == (None, "no_answer")
    finally:
        sys.modules.pop("ptbr_bench", None)


def test_same_evaluator_calls_reach_native_fake_and_bridge_identically(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(RESEARCH))
    sys.modules.pop("ptbr_bench", None)
    bench = importlib.import_module("ptbr_bench")
    received: dict[str, list[bytes]] = {"native": [], "bridge": []}

    class Native(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            received["native"].append(self.rfile.read(int(self.headers["content-length"])))
            out = json.dumps({"answers": {"q": {"noul": 1}}}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    native = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Native)
    bridge = make_bridge(mod, FakeBackend(["B"] * 3))
    real = mod.make_server(bridge, 0)
    orig_handle = bridge.handle

    def recording(raw: bytes) -> Any:
        received["bridge"].append(raw)
        return orig_handle(raw)

    bridge.handle = recording
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (native, real)]
    for t in threads:
        t.start()
    try:
        row = {"type": "choice", "instructions": "i", "criteria": {"y": "dy", "x": "dx", "z": "dz"}}
        rows = [
            (row, "Texto ção"),
            ({"type": "noul", "instructions": "o?"}, "outro"),
            (row, "x" * 50),
        ]
        for server in (native, real):
            for q, state in rows:
                bench.call(f"http://127.0.0.1:{server.server_address[1]}", MODEL, state, q, 10)
    finally:
        for s in (native, real):
            s.shutdown()
            s.server_close()
        sys.modules.pop("ptbr_bench", None)
    assert received["native"] == received["bridge"]
    assert len(received["bridge"]) == 3


def test_evidence_written_exclusively_and_digest_is_stable(mod: Any, tmp_path: Path) -> None:
    bridge = make_bridge(mod, FakeBackend())
    target = tmp_path / "evidence.json"
    mod.publish_exclusive(target, bridge.evidence_bytes)
    assert target.read_bytes() == bridge.evidence_bytes
    assert (
        hashlib.sha256(target.read_bytes()).hexdigest()
        == bridge.evidence_sha256
        == make_bridge(mod, FakeBackend()).evidence_sha256
    )
    with pytest.raises(mod.BridgeError):
        mod.publish_exclusive(target, b"other")
    assert target.read_bytes() == bridge.evidence_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["evidence.json"]  # no temporary left behind


def test_evidence_content_has_no_paths_or_request_text(mod: Any) -> None:
    calls.clear()
    cfg = config(mod)
    _, info = mod.load_transformers_backend(cfg, fake_importer)
    evidence = mod.build_evidence(cfg, info, "d" * 64)
    text = json.dumps(evidence)
    assert "/Users" not in text and "/home" not in text
    for key in (
        "schema_version",
        "bridge_sha256",
        "model",
        "backend",
        "limits",
        "prompt",
        "decoding",
    ):
        assert key in evidence
    assert evidence["model"]["revision"] == REVISION
    assert evidence["limits"]["context"] == 4096 and evidence["limits"]["max_new_tokens"] == 32
    assert evidence["limits"]["generation_timeout_seconds"] == 90.0
    assert evidence["prompt"]["sha256"] == mod.PROMPT_SHA256
    assert mod.FROZEN_DATE == "2026-10-21"


def test_evidence_is_immutable_while_counters_change(mod: Any) -> None:
    backend = FakeBackend(["??"])
    bridge = make_bridge(mod, backend)
    before = bridge.health()
    bridge.handle(body(NOUL))
    after = bridge.health()
    assert (
        before["evidence"] == after["evidence"]
        and before["evidence_sha256"] == after["evidence_sha256"]
    )
    assert before["counters"]["invalid_output"] == 0 and after["counters"]["invalid_output"] == 1


def test_no_default_heavy_imports_or_downloads(mod: Any) -> None:
    source = (RESEARCH / "llm_bridge.py").read_text()
    top_level = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("torch" in ln or "transformers" in ln for ln in top_level)
    assert "snapshot_download" not in source and "hf_hub_download" not in source
    assert "trust_remote_code=True" not in source
