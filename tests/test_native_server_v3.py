import asyncio
import functools
import importlib.util
import json
import os
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest


def _load_research_module(name: str) -> Any:
    path = Path(__file__).parents[1] / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_research_module("ptbr_bench")
diagnostic = _load_research_module("full_text_diagnostic")
native_server = _load_research_module("native_server_v3")


def _canonical_full_legal(tmp_path: Path) -> tuple[Path, Path, dict[str, list[dict[str, Any]]]]:
    rows = {
        task: [
            {
                "type": "choice",
                "text": f"{task} text {index}",
                "instructions": "choose",
                "criteria": {"a": "A", "b": "B"},
                "gold": "a",
            }
            for index in range(count)
        ]
        for task, count in diagnostic.LEGACY_COUNTS.items()
    }
    data, prepared = tmp_path / "rows.json", tmp_path / "prepared.json"
    data.write_text(json.dumps(rows))
    diagnostic.prepare(data, prepared, mode="v3legalfull")
    return prepared, data, rows


def test_v1_preflight_materializes_exact_prefix_full_pairs(tmp_path: Path) -> None:
    rows, frozen = [], []
    for index in range(893):
        text = "x" * 3501 if index < 429 else "short"
        rows.append({"text": text, "instructions": "choose", "criteria": {"a": "A", "b": "B"}})
        frozen.append({"original_index": index, "question_order": ["a", "b"]})
    prepared = tmp_path / "prepared.json"
    data = tmp_path / "rows.json"
    prepared.write_text(json.dumps({"mode": "v1legacydiag", "tasks": {"juristcu_area": frozen}}))
    data.write_text(json.dumps({"juristcu_area": rows}))
    requests = native_server.load_frozen_requests(prepared, data)
    assert len(requests) == 858
    assert requests[0]["arm"] == "prefix"
    assert len(requests[0]["state"]) == 3500
    assert requests[1]["arm"] == "full"
    assert requests[0]["questions"]["q"]["criteria"] == {"a": "A", "b": "B"}


def test_v3_preflight_rejects_missing_32_options(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.json"
    data = tmp_path / "rows.json"
    prepared.write_text(json.dumps({"mode": "v3camara", "tasks": {"camara_tema": []}}))
    data.write_text(json.dumps({"camara_tema": []}))
    with pytest.raises(native_server.PreflightError, match="599"):
        native_server.load_frozen_requests(prepared, data)


def test_v3_full_legal_materializes_all_frozen_rows_in_original_order(tmp_path: Path) -> None:
    prepared, data, rows = _canonical_full_legal(tmp_path)

    requests = native_server.load_frozen_requests(prepared, data)

    assert len(requests) == 893
    assert [request["index"] for request in requests] == list(range(893))
    assert {request["arm"] for request in requests} == {"full"}
    assert requests[17]["state"] == rows["juristcu_area"][17]["text"]
    assert (
        list(requests[0]["questions"]["q"]["criteria"])
        == diagnostic.legacy_orders(rows)["juristcu_area"][0]
    )


def test_v3_full_legal_rejects_bad_identity_and_wrong_full_source(tmp_path: Path) -> None:
    prepared, data, _rows = _canonical_full_legal(tmp_path)
    forged = json.loads(prepared.read_text())
    forged["prepared_sha256"] = "0" * 64
    prepared.write_text(json.dumps(forged))
    with pytest.raises(native_server.PreflightError, match="identity is not canonical"):
        native_server.load_frozen_requests(prepared, data)

    prepared = tmp_path / "prepared-second.json"
    diagnostic.prepare(data, prepared, mode="v3legalfull")
    data_value = json.loads(data.read_text())
    data_value["juristcu_area"][1]["text"] = "wrong full source"
    data.write_text(json.dumps(data_value))
    with pytest.raises(native_server.PreflightError, match="source hash"):
        native_server.load_frozen_requests(prepared, data)


@pytest.mark.parametrize("field", ["gold", "row_sha256", "question_order"])
def test_v3_full_legal_rejects_self_rehashed_forged_prepared(tmp_path: Path, field: str) -> None:
    prepared, data, _rows = _canonical_full_legal(tmp_path)
    forged = json.loads(prepared.read_text())
    item = forged["tasks"]["juristcu_area"][0]
    if field == "gold":
        item[field] = "b"
    elif field == "row_sha256":
        item[field] = "0" * 64
    else:
        item[field] = list(reversed(item[field]))
    forged["prepared_sha256"] = diagnostic.sha(
        {key: value for key, value in forged.items() if key != "prepared_sha256"}
    )
    prepared.write_text(json.dumps(forged))

    with pytest.raises(native_server.PreflightError, match="row/order"):
        native_server.load_frozen_requests(prepared, data)


def test_immutable_evidence_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    native_server.publish_immutable(target, {"a": 1})
    with pytest.raises(native_server.PreflightError, match="overwrite"):
        native_server.publish_immutable(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}


def test_health_evidence_header_is_frozen_and_health_only() -> None:
    async def upstream(scope: Any, _receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"upstream":true}'})

    async def exchange(app: Any, path: str) -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await app({"type": "http", "path": path, "method": "GET", "headers": []}, receive, send)
        return sent

    binding = native_server.HealthEvidenceBinding()
    app = binding.app(upstream)
    unavailable = asyncio.run(exchange(app, "/health"))
    assert unavailable[0]["status"] == 503

    evidence = {"bound": ["after", "upstream", "preflight"]}
    binding.freeze(evidence)
    health = asyncio.run(exchange(app, "/health"))
    headers = dict(health[0]["headers"])
    assert health[0]["status"] == 200
    assert health[1]["body"] == b'{"upstream":true}'
    assert (
        headers[b"x-saracura-evidence-sha256"]
        == native_server.hashlib.sha256(native_server.canonical(evidence)).hexdigest().encode()
    )

    prediction = asyncio.run(exchange(app, "/v1/systemone"))
    assert prediction[0]["status"] == 200
    assert b"x-saracura-evidence-sha256" not in dict(prediction[0]["headers"])
    with pytest.raises(native_server.PreflightError, match="already frozen"):
        binding.freeze({"different": True})


def test_kev_installs_ready_health_after_live_server_with_bound_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.state = types.SimpleNamespace()
            self.handlers: dict[str, Any] = {}

        def get(self, path: str) -> Any:
            def register(handler: Any) -> Any:
                self.handlers[path] = handler
                return handler

            return register

        async def __call__(self, scope: dict[str, Any], _receive: Any, send: Any) -> None:
            if scope["path"] == "/health":
                value = self.handlers["/health"]()
            else:
                value = {"upstream_prediction": True}
            body = native_server.canonical(value)
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})

    class FakeCheckpoint:
        def __init__(self, path: str) -> None:
            self.path = path
            self.meta = types.SimpleNamespace(
                base="fixture/base", base_revision="fixture-base-revision"
            )

        def load(self, *_args: Any, **_kwargs: Any) -> tuple[str, str]:
            return "tokenizer", "model"

    class FakeServer:
        def __init__(self, *args: Any) -> None:
            self.args = args

    app = FakeApp()
    api: Any = types.ModuleType("kev.api")
    api.SystemOneRequest = types.SimpleNamespace(model_validate=lambda value: value)
    api.to_record = lambda value: (value, None)
    checkpoint: Any = types.ModuleType("kev.checkpoint")
    checkpoint.Checkpoint = FakeCheckpoint
    checkpoint.LoadOptions = lambda **kwargs: kwargs
    model: Any = types.ModuleType("kev.model")
    model.encode = lambda *_args, **_kwargs: {
        "ids": [1, 2],
        "state_tokens": 1,
        "state_truncated": False,
    }
    serve: Any = types.ModuleType("kev.serve")
    serve.Server = FakeServer
    serve.app = app
    package = types.ModuleType("kev")
    package.__path__ = []
    for name, module in {
        "kev": package,
        "kev.api": api,
        "kev.checkpoint": checkpoint,
        "kev.model": model,
        "kev.serve": serve,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    config = types.SimpleNamespace(
        checkpoint_path="/checkpoint",
        checkpoint_id="fixture/kev-checkpoint",
        base_id="fixture/base",
        base_revision="fixture-base-revision",
    )
    returned, _checks = native_server._preflight_kev(
        config,
        [{"state": "frozen", "questions": {"q": {}}}],
        types.SimpleNamespace(bfloat16="bfloat16"),
    )
    assert returned is app
    assert isinstance(app.state.server, FakeServer)
    assert app.state.server.args[0].path == "/checkpoint"
    assert app.handlers["/health"]() == {
        "status": "ready",
        "runtime": "kev",
        "model": "fixture/kev-checkpoint",
    }

    async def exchange(path: str) -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        binding = native_server.HealthEvidenceBinding()
        evidence = {"checkpoint": config.checkpoint_id}
        binding.freeze(evidence)
        await binding.app(app)({"type": "http", "path": path}, receive, send)
        return sent

    health = asyncio.run(exchange("/health"))
    assert json.loads(health[1]["body"]) == app.handlers["/health"]()
    assert dict(health[0]["headers"])[b"x-saracura-evidence-sha256"] == (
        native_server.hashlib.sha256(native_server.canonical({"checkpoint": config.checkpoint_id}))
        .hexdigest()
        .encode()
    )
    prediction = asyncio.run(exchange("/v1/systemone"))
    assert json.loads(prediction[1]["body"]) == {"upstream_prediction": True}
    assert b"x-saracura-evidence-sha256" not in dict(prediction[0]["headers"])


def test_native_parser_requires_explicit_identifiers() -> None:
    with pytest.raises(SystemExit):
        native_server.parser().parse_args(["--kind", "kev"])
    values = native_server.parser().parse_args(
        [
            "--kind",
            "kev",
            "--checkpoint-path",
            "/checkpoint",
            "--checkpoint-id",
            "org/model",
            "--checkpoint-revision",
            "a" * 40,
            "--base-id",
            "org/base",
            "--base-revision",
            "b" * 40,
            "--port",
            "18032",
            "--prepared-file",
            "/prepared",
            "--data-file",
            "/data",
            "--evidence-output",
            "/evidence",
            "--upstream-root",
            "/upstream",
            "--hf-cache",
            "/cache",
        ]
    )
    assert values.kind == "kev" and values.port == 18032


def test_native_decider_disables_warmup_and_deferred_lifespan_runs_after_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(sys, "path", list(sys.path))
    config = native_server.Config(
        kind="decider",
        checkpoint_path="/checkpoint",
        checkpoint_id="model",
        checkpoint_revision="rev",
        base_id="base",
        base_revision="base-rev",
        port=18032,
        prepared_file="/prepared",
        data_file="/data",
        evidence_output="/evidence",
        upstream_root="/upstream",
        hf_cache="/cache",
    )
    native_server.configure_environment(config)
    assert __import__("os").environ["DECIDER_WARMUP"] == "0"

    events: list[str] = []
    app = types.SimpleNamespace(router=types.SimpleNamespace())

    @asynccontextmanager
    async def original_lifespan(_instance: Any) -> Any:
        app.engine = "available-after-entry"
        events.append("entered")
        yield
        events.append("exited")

    app.router.lifespan_context = original_lifespan

    def publish() -> None:
        assert app.engine == "available-after-entry"
        events.append("preflight")

    native_server.install_checked_lifespan(app, publish)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            assert events == ["entered", "preflight"]

    asyncio.run(exercise())
    assert events == ["entered", "preflight", "exited"]


def test_decider_full_legal_uses_32768_and_rejects_native_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        @staticmethod
        def encode(value: str, **_kwargs: Any) -> list[int]:
            return list(range(len(value)))

    def prepared(
        state: str, _questions: Any, _debug: bool
    ) -> tuple[None, None, list[dict[str, Any]], None]:
        return None, None, [{"ids": list(range(len(state) + 20))}], None

    serve: Any = types.ModuleType("decider.serve")
    serve.eng = types.SimpleNamespace(
        tok=Tokenizer(),
        model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=32768)),
    )
    serve.S1 = types.SimpleNamespace(render_state=lambda state: state)
    serve.SCHEMA_FIRST = False
    serve._worth_caching = lambda *_args: False
    serve._prepare_s1 = prepared
    serve.MAX_STATE_TOKENS = 32768
    serve.MAX_ROW_TOKENS = 32768
    serve.check_size = lambda widths: None
    package = types.ModuleType("decider")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "decider", package)
    monkeypatch.setitem(sys.modules, "decider.serve", serve)
    requests = [
        {"task": "juristcu_area", "arm": "full", "state": "x" * 3501, "questions": {"q": {}}}
        for _ in range(893)
    ]

    checks = native_server._preflight_decider(types.SimpleNamespace(), requests, None)
    assert checks["context_check"]["declared_state_budget"] == 32768
    assert checks["context_check"]["model_position_limit"] == 32768

    serve.check_size = lambda _widths: (_ for _ in ()).throw(ValueError("hard limit"))
    with pytest.raises(native_server.PreflightError, match="hard context"):
        native_server._preflight_decider(types.SimpleNamespace(), requests, None)

    camara = [{"task": "camara_tema", "arm": "full", "state": "short", "questions": {"q": {}}}]
    serve.MAX_STATE_TOKENS = 8192
    serve.MAX_ROW_TOKENS = 8192
    serve.check_size = lambda widths: None
    assert (
        native_server._preflight_decider(types.SimpleNamespace(), camara, None)["context_check"][
            "declared_state_budget"
        ]
        == 8192
    )


def test_laya_preflight_pins_live_budgets_and_proves_complete_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        mask_token_id = 103
        mask_token = "[MASK]"

        def __call__(self, value: Any, **_kwargs: Any) -> dict[str, list[int]]:
            return {"input_ids": [1000 + ord(char) for char in str(value)]}

    class FakeAgent:
        def __init__(self) -> None:
            self.tok = FakeTokenizer()
            self.amp_enabled = True
            self.dtype = "amp-default"

        @staticmethod
        def _to_internal(question: dict[str, Any]) -> dict[str, Any]:
            return {
                "t": question["type"],
                "ins": question["instructions"],
                "options": list(question["criteria"].values()),
            }

    class FakeRouter:
        latest = None

        def __init__(self, **kwargs: Any) -> None:
            self.agent = FakeAgent()
            self.calls: list[dict[str, Any]] = []
            self.init = kwargs
            FakeRouter.latest = self

        def load(self, _model: Any) -> Any:
            return self.agent

        def predict(self, state: Any, questions: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
            self.calls.append({"state": state, "questions": questions, "kwargs": kwargs})
            return {"answers": {}}

    encoded: list[tuple[list[int], list[int], dict[str, Any]]] = []

    def fake_build_sequence(
        tokenizer: Any, _state: Any, internal: dict[str, Any], **kwargs: Any
    ) -> tuple[list[int], list[int]]:
        expected, markers = native_server._laya_expected_sequence(
            tokenizer,
            kwargs["state_ids"],
            internal,
            internal["options"],
            max_len=kwargs["max_len"],
            head_max_len=kwargs["head_max_len"],
        )
        encoded.append((expected, markers, kwargs))
        return expected, markers

    package: Any = types.ModuleType("laya")
    package.__path__ = []
    agent_module: Any = types.ModuleType("laya.agent")
    agent_module.Agent = FakeAgent
    common_module: Any = types.ModuleType("laya.common")
    common_module.build_sequence = fake_build_sequence
    common_module.render_options = lambda internal: internal["options"]
    common_module.serialize_state = lambda state: f"STATE:{state}"
    router_module: Any = types.ModuleType("laya.router")
    router_module.Router = FakeRouter
    serve_module: Any = types.ModuleType("laya.serve")
    serve_module.create_app = lambda router: types.SimpleNamespace(router=router)
    for name, module in {
        "laya": package,
        "laya.agent": agent_module,
        "laya.common": common_module,
        "laya.router": router_module,
        "laya.serve": serve_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    request = {
        "state": "full frozen state",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "use every instruction",
                "criteria": {
                    "a": "first option",
                    "b": "second option",
                },
            }
        },
    }
    fake_torch = types.SimpleNamespace(float32="float32")
    app, checks = native_server._preflight_laya(
        types.SimpleNamespace(checkpoint_path="/checkpoint", checkpoint_id="fixture/laya"),
        [request],
        fake_torch,
    )
    router = app.router
    assert router.init["models"] == {"typed-decisions": "/checkpoint"}
    assert router.init["default"] == "typed-decisions"
    assert router.agent.amp_enabled is False
    assert router.agent.dtype == "float32"
    router.predict("live state", {"q": {}})
    assert isinstance(router.predict, functools.partial)
    assert cast(Any, router.predict.func).__func__ is FakeRouter.predict
    assert router.calls[-1]["kwargs"] == {"max_len": 8192, "head_max_len": 2048}
    sequence, markers, kwargs = encoded[0]
    assert kwargs["max_len"] == 8192 and kwargs["head_max_len"] == 2048
    assert markers[0] == len(
        [101, *FakeTokenizer()("choice question: use every instruction")["input_ids"], 102]
    )
    assert markers[1] == markers[0] + 1 + len(FakeTokenizer()(" first option")["input_ids"])
    assert (
        FakeTokenizer()("STATE:full frozen state")["input_ids"]
        == sequence[-len("STATE:full frozen state") - 1 : -1]
    )
    assert checks["token_check"]["full_sequence_verified"] is True

    with pytest.raises(native_server.PreflightError, match="complete instructions/options/state"):
        native_server._laya_complete_sequence(
            lambda *_args, **_kwargs: (sequence[:-1], markers),
            FakeTokenizer(),
            request["state"],
            {
                "t": "choice",
                "ins": "use every instruction",
                "options": ["first option", "second option"],
            },
            FakeTokenizer()("STATE:full frozen state")["input_ids"],
            ["first option", "second option"],
            max_len=8192,
            head_max_len=8192,
        )


def test_context_limit_reads_real_nested_encoder_config() -> None:
    wrapper = types.SimpleNamespace(
        encoder=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=8192))
    )
    assert native_server._model_context_limit(wrapper, "Laya") == 8192


def test_context_limit_reads_decider_engine_v2_wrapper() -> None:
    engine = types.SimpleNamespace(
        m=types.SimpleNamespace(
            lm=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=262144))
        )
    )
    assert native_server._model_context_limit(engine, "Decider") == 262144
