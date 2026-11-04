#!/usr/bin/env python3
"""Task-owned CUDA launchers for unchanged Kev, Decider, Lev, and Laya servers.

This file deliberately contains no model download or fallback.  It prepares every
frozen request with the upstream runtime before exposing its original ASGI app.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import full_text_diagnostic as diagnostic
except ModuleNotFoundError:
    _diagnostic_spec = importlib.util.spec_from_file_location(
        "full_text_diagnostic", Path(__file__).with_name("full_text_diagnostic.py")
    )
    if _diagnostic_spec is None or _diagnostic_spec.loader is None:
        raise RuntimeError("cannot load canonical full-text diagnostic module") from None
    diagnostic = importlib.util.module_from_spec(_diagnostic_spec)
    sys.modules["full_text_diagnostic"] = diagnostic
    _diagnostic_spec.loader.exec_module(diagnostic)

PREFIX_CHARS = 3500
FULL_LEGAL_CONTEXT = 32768
LAYA_HEAD_MAX_LEN = 2048
LEGACY_CONTEXT = 8192
KINDS = ("kev", "decider", "lev", "laya")


class PreflightError(RuntimeError):
    """The selected native runtime cannot prove complete frozen input."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_source(path: str | Path) -> str:
    """Stable digest of one source file or an entire supplied upstream directory."""
    source = Path(path)
    if source.is_file():
        return sha_file(source)
    if not source.is_dir():
        raise PreflightError(f"missing upstream source: {source}")
    digest = hashlib.sha256()
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(source)).encode("utf-8") + b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def publish_immutable(path: str | Path, payload: dict[str, Any]) -> None:
    """Create evidence once using fsync plus hard-link publication, never replace it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PreflightError(f"refusing to overwrite immutable evidence: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


class HealthEvidenceBinding:
    """One immutable evidence digest exposed only by the upstream health route."""

    def __init__(self) -> None:
        self._digest: str | None = None

    def freeze(self, payload: dict[str, Any]) -> None:
        if self._digest is not None:
            raise PreflightError("health evidence binding is already frozen")
        self._digest = hashlib.sha256(canonical(payload)).hexdigest()

    def app(self, upstream: Any) -> Any:
        async def bound(scope: dict[str, Any], receive: Any, send: Any) -> None:
            if scope.get("type") != "http" or scope.get("path") != "/health":
                await upstream(scope, receive, send)
                return
            if self._digest is None:
                body = canonical({"detail": "health evidence binding unavailable"})
                await send(
                    {
                        "type": "http.response.start",
                        "status": 503,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return

            async def health_send(message: dict[str, Any]) -> None:
                if message.get("type") == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-saracura-evidence-sha256", self._digest.encode("ascii")))
                    message = {**message, "headers": headers}
                await send(message)

            await upstream(scope, receive, health_send)

        return bound


def read_json_or_jsonl(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise PreflightError("data file is not a non-empty JSON object or JSONL") from None
        return {"camara_tema": rows}
    if not isinstance(value, dict) or not all(isinstance(rows, list) for rows in value.values()):
        raise PreflightError("data file must be task-to-rows JSON or JSONL")
    return value


def load_frozen_requests(prepared_path: str | Path, data_path: str | Path) -> list[dict[str, Any]]:
    """Materialize the exact frozen diagnostic requests without changing their order.

    V1 diagnostic prepared files contribute both prefix and full variants for
    precisely the affected legal records.  V3 Câmara files contribute all 599
    32-choice requests.  Unknown shapes are rejected rather than guessed.
    """
    raw_prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    mode = raw_prepared.get("mode") if isinstance(raw_prepared, dict) else None
    if mode == "v3legalfull":
        try:
            prepared = diagnostic.load_prepared(prepared_path)
        except ValueError as exc:
            raise PreflightError("V3 full legal prepared identity is not canonical") from exc
    else:
        prepared = raw_prepared
    data = read_json_or_jsonl(data_path)
    mode = prepared.get("mode")
    tasks = prepared.get("tasks")
    if not isinstance(tasks, dict):
        raise PreflightError("prepared file has no frozen tasks")
    if mode == "v1legacydiag":
        frozen = tasks.get("juristcu_area")
        rows = data.get("juristcu_area")
        if not isinstance(frozen, list) or not isinstance(rows, list) or len(frozen) != len(rows):
            raise PreflightError("V1 frozen legal rows do not match supplied data")
        requests: list[dict[str, Any]] = []
        for item in frozen:
            index = item.get("original_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
                raise PreflightError("invalid frozen legal row index")
            row = rows[index]
            text, order, criteria = row.get("text"), item.get("question_order"), row.get("criteria")
            if (
                not isinstance(text, str)
                or not isinstance(order, list)
                or not isinstance(criteria, dict)
            ):
                raise PreflightError("frozen legal row lacks text/order/criteria")
            if set(order) != set(criteria):
                raise PreflightError("frozen legal question order does not match criteria")
            if len(text) <= PREFIX_CHARS:
                continue
            question = {
                "type": "choice",
                "instructions": row.get("instructions"),
                "criteria": {key: criteria[key] for key in order},
            }
            for arm, state in (("prefix", text[:PREFIX_CHARS]), ("full", text)):
                requests.append(
                    {
                        "task": "juristcu_area",
                        "index": index,
                        "arm": arm,
                        "state": state,
                        "questions": {"q": question},
                    }
                )
        if len(requests) != 858:
            raise PreflightError(
                f"V1 preflight requires 429 prefix/full pairs, got {len(requests)} requests"
            )
        return requests
    if mode == "v3legalfull":
        if diagnostic.source_hash(data_path) != prepared["rows_source_sha256"]:
            raise PreflightError("V3 full legal prepared source hash does not match supplied data")
        try:
            diagnostic._legacy(data)
        except ValueError as exc:
            raise PreflightError("V3 full legal source is not the canonical V2 rows") from exc
        if prepared.get("legacy_counts") != diagnostic.LEGACY_COUNTS:
            raise PreflightError("V3 full legal prepared legacy counts are not canonical")
        orders = diagnostic.legacy_orders(data)
        if prepared.get("legacy_order_stream_sha256") != diagnostic.sha(orders):
            raise PreflightError("V3 full legal prepared RNG order stream does not match source")
        frozen = tasks.get("juristcu_area")
        rows = data.get("juristcu_area")
        if (
            not isinstance(frozen, list)
            or not isinstance(rows, list)
            or len(frozen) != 893
            or len(rows) != 893
        ):
            raise PreflightError("V3 full legal preflight requires exactly 893 frozen rows")
        requests = []
        for expected_index, item in enumerate(frozen):
            index = item.get("original_index")
            if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
                raise PreflightError("V3 full legal frozen indices must retain original order")
            row = rows[index]
            text, order, criteria = row.get("text"), item.get("question_order"), row.get("criteria")
            if (
                not isinstance(text, str)
                or not isinstance(order, list)
                or not isinstance(criteria, dict)
                or item.get("gold") != row.get("gold")
                or not isinstance(item.get("text_sha256"), str)
                or diagnostic.sha(text) != item["text_sha256"]
                or item.get("row_sha256") != diagnostic.sha(row)
                or order != orders["juristcu_area"][index]
            ):
                raise PreflightError(
                    "V3 full legal row/order does not match its frozen full source"
                )
            if set(order) != set(criteria) or len(order) != len(criteria):
                raise PreflightError("V3 full legal question order does not match criteria")
            requests.append(
                {
                    "task": "juristcu_area",
                    "index": index,
                    "arm": "full",
                    "state": text,
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": row.get("instructions"),
                            "criteria": {key: criteria[key] for key in order},
                        }
                    },
                }
            )
        return requests
    if mode == "v3camara":
        frozen = tasks.get("camara_tema")
        rows = data.get("camara_tema")
        if (
            not isinstance(frozen, list)
            or not isinstance(rows, list)
            or len(frozen) != 599
            or len(rows) != 599
        ):
            raise PreflightError("V3 Câmara preflight requires exactly 599 frozen rows")
        requests = []
        for item in frozen:
            index = item.get("original_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
                raise PreflightError("invalid frozen Câmara row index")
            row, order, criteria = (
                rows[index],
                item.get("question_order"),
                rows[index].get("criteria"),
            )
            if (
                not isinstance(row.get("text"), str)
                or not isinstance(order, list)
                or not isinstance(criteria, dict)
            ):
                raise PreflightError("frozen Câmara row lacks text/order/criteria")
            if len(order) != 32 or set(order) != set(criteria):
                raise PreflightError("Câmara request does not retain all 32 options")
            question = {
                "type": "choice",
                "instructions": row.get("instructions"),
                "criteria": {key: criteria[key] for key in order},
            }
            requests.append(
                {
                    "task": "camara_tema",
                    "index": index,
                    "arm": "full",
                    "state": row["text"],
                    "questions": {"q": question},
                }
            )
        return requests
    raise PreflightError(f"unsupported prepared mode: {mode!r}")


@dataclass(frozen=True)
class Config:
    kind: str
    checkpoint_path: str
    checkpoint_id: str
    checkpoint_revision: str
    base_id: str
    base_revision: str
    port: int
    prepared_file: str
    data_file: str
    evidence_output: str
    upstream_root: str
    hf_cache: str


def configure_environment(config: Config) -> None:
    root = Path(config.upstream_root).resolve()
    os.environ.update(
        {
            "HF_HOME": config.hf_cache,
            "HF_HUB_CACHE": str(Path(config.hf_cache) / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "KEV_PREFIX_CACHE": "0",
        }
    )
    if config.kind == "decider":
        os.environ.update(
            {
                "DECIDER_MODEL": config.checkpoint_path,
                "DECIDER_DEVICE": "cuda",
                "DECIDER_IMAGES": "0",
                "DECIDER_WARMUP": "0",
                "DECIDER_MAX_STATE_TOKENS": str(
                    FULL_LEGAL_CONTEXT
                    if _prepared_is_full_legal(config.prepared_file)
                    else LEGACY_CONTEXT
                ),
            }
        )
    elif config.kind == "laya":
        os.environ.update({"LAYA_DEVICE": "cuda", "LAYA_PRELOAD": "0", "LAYA_AUTO_TASK": "0"})
    # Installed packages remain the runtime.  The optional supplied source tree is
    # provenance-only and is never patched or substituted for the installed wheel.
    sys.path[:0] = [str(root), str(root / "runtime-source")]


def _prepared_is_full_legal(path: str | Path) -> bool:
    """Read only the mode needed to configure the untouched upstream server."""
    source = Path(path)
    if not source.is_file():
        return False
    try:
        return json.loads(source.read_text(encoding="utf-8")).get("mode") == "v3legalfull"
    except (OSError, json.JSONDecodeError):
        return False


def _is_full_legal(requests: list[dict[str, Any]]) -> bool:
    return len(requests) == 893 and all(
        request.get("task") == "juristcu_area" and request.get("arm") == "full"
        for request in requests
    )


def _native_context_budget(requests: list[dict[str, Any]]) -> int:
    return FULL_LEGAL_CONTEXT if _is_full_legal(requests) else LEGACY_CONTEXT


def _model_context_limit(model: Any, runtime: str) -> int:
    """Return a declared model position limit; guesses are unsafe for full input."""
    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", candidate)
        for name in ("max_position_embeddings", "max_sequence_length", "n_positions"):
            value = getattr(config, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        for name in ("encoder", "model", "base_model", "backbone", "m", "lm", "core"):
            child = getattr(candidate, name, None)
            if child is not None and len(seen) < 12:
                pending.append(child)

    raise PreflightError(f"{runtime} model does not declare a usable position/context limit")


def upstream_source(config: Config) -> Path:
    root = Path(config.upstream_root)
    choices = {
        "kev": (root / "upstream-kev.tar.gz",),
        "lev": (root / "lev-runtime-source.tgz",),
        "decider": (
            root / "runtime-source" / "decider",
            root / "wheels" / "decider_ai-1.10.0-py3-none-any.whl",
        ),
        "laya": (
            root / "runtime-source" / "laya",
            root / "wheels" / "laya-0.3.20-py3-none-any.whl",
        ),
    }[config.kind]
    return next((choice for choice in choices if choice.exists()), choices[0])


def package_versions(kind: str) -> dict[str, str]:
    names = {
        "kev": ("torch", "transformers", "peft"),
        "decider": ("decider-ai", "torch"),
        "lev": ("lev", "torch"),
        "laya": ("laya", "torch"),
    }[kind]
    return {name: importlib.metadata.version(name) for name in names}


def _request_hash(requests: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(b"".join(canonical(request) + b"\n" for request in requests)).hexdigest()


def _preflight_kev(
    config: Config, requests: list[dict[str, Any]], torch: Any
) -> tuple[Any, dict[str, Any]]:
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import encode
    from kev.serve import Server, app

    checkpoint = Checkpoint(config.checkpoint_path)
    if checkpoint.meta.base != config.base_id or checkpoint.meta.base_revision not in (
        None,
        config.base_revision,
    ):
        raise PreflightError("Kev checkpoint base or pinned base revision differs from CLI")
    checkpoint.meta.base_revision = config.base_revision
    tokenizer, model = checkpoint.load("cuda", LoadOptions(dtype=torch.bfloat16, backend="torch"))
    maximum = state_max = 0
    for request in requests:
        req = SystemOneRequest.model_validate(
            {
                "model": config.checkpoint_id,
                "state": request["state"],
                "questions": request["questions"],
            }
        )
        record, _ = to_record(req)
        encoded = encode(tokenizer, record, max_state=65536, max_branch=73728, strict=True)
        if encoded.get("state_truncated"):
            raise PreflightError("Kev upstream encoder truncated a frozen request")
        maximum = max(maximum, len(encoded["ids"]))
        state_max = max(state_max, int(encoded["state_tokens"]))
    app.state.server = Server(checkpoint, tokenizer, model, "cuda")

    # Kev exposes model discovery but no health route. Add readiness only;
    # prediction endpoints and their preprocessing stay upstream-owned.
    @app.get("/health")
    def kev_health():
        return {"status": "ready", "runtime": "kev", "model": config.checkpoint_id}

    return app, {
        "token_check": {
            "strict": True,
            "max_record_tokens": maximum,
            "max_state_tokens": state_max,
        },
        "context_check": {"max_state": 65536, "max_branch": 73728},
    }


def _preflight_decider(
    config: Config, requests: list[dict[str, Any]], _torch: Any
) -> dict[str, Any]:
    import decider.serve as serve

    # `serve.eng` exists only after the original app lifespan has entered.  The caller
    # wraps that lifespan, not the upstream route or preprocessing implementation.
    if serve.eng is None:
        raise PreflightError(
            "Decider engine unavailable; preflight must run after original lifespan startup"
        )
    budget = _native_context_budget(requests)
    configured = getattr(serve, "MAX_STATE_TOKENS", None)
    if _is_full_legal(requests) and configured != budget:
        raise PreflightError(
            f"Decider effective state budget is {configured}, expected declared {budget}; full input unsupported"
        )
    engine_limit = (
        _model_context_limit(getattr(serve.eng, "model", serve.eng), "Decider")
        if _is_full_legal(requests)
        else int(getattr(serve, "MAX_ROW_TOKENS", LEGACY_CONTEXT))
    )
    maximum = state_max = 0
    for request in requests:
        count = len(
            serve.eng.tok.encode(serve.S1.render_state(request["state"]), add_special_tokens=False)
        )
        if count > budget:
            raise PreflightError(f"Decider state exceeds declared {budget}-token budget")
        if serve.SCHEMA_FIRST and serve._worth_caching(request["questions"], False):
            prefixes, suffix = serve._plan_schema(request["questions"], False, request["state"])
            width = max(prefixes) + len(suffix[0])
        else:
            _, _, items, _ = serve._prepare_s1(request["state"], request["questions"], False)
            width = max(len(item["ids"]) for item in items)
        if width > engine_limit:
            raise PreflightError(
                f"Decider complete request needs {width} tokens but model limit is {engine_limit}"
            )
        try:
            serve.check_size([width])
        except Exception as exc:
            raise PreflightError(
                f"Decider upstream hard context rejects complete {width}-token request"
            ) from exc
        maximum, state_max = max(maximum, width), max(state_max, count)
    return {
        "token_check": {
            "max_record_tokens": maximum,
            "max_state_tokens": state_max,
            "schema_first": bool(serve.SCHEMA_FIRST),
        },
        "context_check": {
            "declared_state_budget": budget,
            "model_position_limit": engine_limit,
            "max_row_tokens": serve.MAX_ROW_TOKENS,
        },
    }


def _extract_lev_engine(app: Any) -> Any:
    endpoint = next(
        (route.endpoint for route in app.routes if getattr(route, "path", "") == "/health"), None
    )
    state = [
        cell.cell_contents
        for cell in (getattr(endpoint, "__closure__", ()) or ())
        if isinstance(cell.cell_contents, dict) and "engine" in cell.cell_contents
    ]
    if len(state) != 1 or state[0].get("engine") is None:
        raise PreflightError("Lev original health closure did not expose a live engine")
    return state[0]["engine"]


def _preflight_lev(app: Any, config: Config, requests: list[dict[str, Any]]) -> dict[str, Any]:
    engine = _extract_lev_engine(app)
    budget = _native_context_budget(requests)
    engine_limit = (
        _model_context_limit(getattr(engine, "model", engine), "Lev")
        if _is_full_legal(requests)
        else LEGACY_CONTEXT
    )
    maximum = state_max = 0
    for request in requests:
        kwargs: dict[str, int] = {}
        if _is_full_legal(requests):
            parameters = inspect.signature(engine.prepare).parameters
            for name in ("max_len", "max_tokens", "max_context_tokens", "context_window"):
                if name in parameters:
                    kwargs[name] = budget
                    break
        prepared = engine.prepare(request["state"], request["questions"], **kwargs)
        width = int(prepared.width)
        state_tokens = len(prepared.prefix_ids)
        if width > engine_limit:
            raise PreflightError(
                f"Lev complete request needs {width} tokens but model limit is {engine_limit}"
            )
        maximum, state_max = max(maximum, width), max(state_max, state_tokens)
    return {
        "token_check": {"max_record_tokens": maximum, "max_state_tokens": state_max},
        "context_check": {
            "declared_budget": budget,
            "model_position_limit": engine_limit,
            "engine_budget_parameter": next(iter(kwargs), None),
        },
    }


def _laya_expected_sequence(
    tokenizer: Any,
    state_ids: list[int],
    internal: dict[str, Any],
    options: list[str],
    *,
    max_len: int,
    head_max_len: int,
) -> tuple[list[int], list[int]]:
    """Return the untruncated upstream sequence for an already-admitted request."""
    mask_token = tokenizer.mask_token
    head_ids = tokenizer(
        f"{internal['t']} question: {str(internal['ins']).replace(mask_token, ' ')}",
        add_special_tokens=False,
    )["input_ids"]
    option_ids = [
        tokenizer(" " + option.replace(mask_token, " "), add_special_tokens=False)["input_ids"]
        for option in options
    ]
    if any(len(option) > 48 for option in option_ids):
        raise PreflightError("Laya upstream build_sequence would truncate an option")
    head_size = len(head_ids) + sum(1 + len(option) for option in option_ids)
    if head_size > head_max_len:
        raise PreflightError("Laya upstream build_sequence would truncate instructions or options")
    expected = [tokenizer.cls_token_id, *head_ids, tokenizer.sep_token_id]
    markers: list[int] = []
    for option in option_ids:
        markers.append(len(expected))
        expected.extend((tokenizer.mask_token_id, *option))
    expected.extend((tokenizer.sep_token_id, *state_ids, tokenizer.sep_token_id))
    if len(expected) > max_len:
        raise PreflightError("Laya upstream build_sequence would truncate frozen state")
    return expected, markers


def _laya_complete_sequence(
    build_sequence: Any,
    tokenizer: Any,
    state: Any,
    internal: dict[str, Any],
    state_ids: list[int],
    options: list[str],
    *,
    max_len: int,
    head_max_len: int,
) -> tuple[list[int], list[int]]:
    """Call upstream with serving budgets and prove its bytes equal the complete input."""
    expected, expected_markers = _laya_expected_sequence(
        tokenizer, state_ids, internal, options, max_len=max_len, head_max_len=head_max_len
    )
    sequence, markers = build_sequence(
        tokenizer,
        state,
        internal,
        max_len=max_len,
        head_max_len=head_max_len,
        state_ids=state_ids,
    )
    if sequence != expected or markers != expected_markers:
        raise PreflightError(
            "Laya build_sequence differs from complete instructions/options/state sequence"
        )
    return sequence, markers


def _preflight_laya(
    config: Config, requests: list[dict[str, Any]], torch: Any
) -> tuple[Any, dict[str, Any]]:
    from laya.agent import Agent
    from laya.common import build_sequence, render_options, serialize_state
    from laya.router import Router
    from laya.serve import create_app

    router = Router(
        models={"typed-decisions": config.checkpoint_path},
        device="cuda",
        max_loaded=1,
        default="typed-decisions",
        auto_task_detection=False,
    )
    agent = router.load("typed-decisions")
    requested_budget = _native_context_budget(requests)
    model_limit = (
        _model_context_limit(getattr(agent, "model", agent), "Laya")
        if _is_full_legal(requests)
        else LEGACY_CONTEXT
    )
    # A model need only fit the actual complete requests, not our generous
    # ceiling. The exact encoder preflight below rejects any real overflow.
    effective_budget = min(requested_budget, model_limit)
    # The upstream CUDA default enables AMP. This run declares float32, so disable
    # AMP explicitly before the app can receive a request.
    agent.amp_enabled = False
    agent.dtype = torch.float32
    router.predict = functools.partial(
        router.predict, max_len=effective_budget, head_max_len=LAYA_HEAD_MAX_LEN
    )
    maximum = state_max = 0
    for request in requests:
        state_ids = agent.tok(
            serialize_state(request["state"]).replace(agent.tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        for question in request["questions"].values():
            internal = Agent._to_internal(question)
            options = render_options(internal)
            sequence, _markers = _laya_complete_sequence(
                build_sequence,
                agent.tok,
                request["state"],
                internal,
                state_ids,
                options,
                max_len=effective_budget,
                head_max_len=LAYA_HEAD_MAX_LEN,
            )
            maximum, state_max = max(maximum, len(sequence)), max(state_max, len(state_ids))
    return create_app(router), {
        "token_check": {
            "max_record_tokens": maximum,
            "max_state_tokens": state_max,
            "options_cap": 48,
            "full_sequence_verified": True,
        },
        "context_check": {
            "declared_budget": requested_budget,
            "model_position_limit": model_limit,
            "effective_budget": effective_budget,
            "head_max_len": LAYA_HEAD_MAX_LEN,
        },
        "dtype_check": {"agent_amp_enabled": False, "agent_dtype": "float32"},
    }


def install_checked_lifespan(app: Any, publish: Any) -> None:
    """Run the original lifespan first, then publish deferred native preflight evidence."""
    from contextlib import asynccontextmanager

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def checked_lifespan(instance: Any):
        async with original_lifespan(instance):
            publish()
            yield

    app.router.lifespan_context = checked_lifespan


def launch(config: Config) -> None:
    if config.kind not in KINDS:
        raise PreflightError("unsupported native runtime")
    configure_environment(config)
    import torch

    if not torch.cuda.is_available():
        raise PreflightError("CUDA is required for native GPU launch")
    torch.set_num_threads(2)
    requests = load_frozen_requests(config.prepared_file, config.data_file)
    if config.kind == "decider":
        import decider.serve as decider

        # This assignment is made before the untouched original lifespan starts.
        decider.resolve_device = lambda requested=None: ("cuda", torch.bfloat16)
        app, deferred = decider.app, lambda: _preflight_decider(config, requests, torch)
    elif config.kind == "lev":
        from lev.server import create_app

        app = create_app(
            checkpoint_dir=config.checkpoint_path,
            model_cache=os.environ["HF_HUB_CACHE"],
            timeout=180,
        )

        def deferred():
            return _preflight_lev(app, config, requests)
    else:
        runners = {"kev": _preflight_kev, "laya": _preflight_laya}
        app, checks = runners[config.kind](config, requests, torch)

        def deferred():
            return checks

    def evidence(checks: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": "saracura-native-server-evidence-v1",
            "checkpoint": {
                "id": config.checkpoint_id,
                "revision": config.checkpoint_revision,
                "base": config.base_id,
                "base_revision": config.base_revision,
            },
            "runtime": {
                "kind": config.kind,
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "packages": package_versions(config.kind),
                "upstream_source_sha256": sha_source(upstream_source(config)),
            },
            "backend": "torch-cuda-native-upstream",
            "dtype": "bfloat16" if config.kind != "laya" else "float32",
            "context": checks["context_check"],
            "prompt": {
                "native_preprocessing": config.kind,
                "prepared_sha256": sha_file(config.prepared_file),
                "data_sha256": sha_file(config.data_file),
            },
            "server": {
                "source_sha256": sha_file(__file__),
                "port": config.port,
                "health": "original-upstream-health",
            },
            "preflight": {
                **checks,
                "requests": len(requests),
                "requests_sha256": _request_hash(requests),
                "no_truncation": True,
            },
        }

    health_binding = HealthEvidenceBinding()

    def publish_evidence(checks: dict[str, Any]) -> None:
        payload = evidence(checks)
        publish_immutable(config.evidence_output, payload)
        health_binding.freeze(payload)

    if config.kind in {"decider", "lev"}:
        install_checked_lifespan(app, lambda: publish_evidence(deferred()))
    else:
        publish_evidence(deferred())
    import uvicorn

    uvicorn.run(health_binding.app(app), host="127.0.0.1", port=config.port, access_log=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--kind", required=True, choices=KINDS)
    result.add_argument("--checkpoint-path", required=True)
    result.add_argument("--checkpoint-id", required=True)
    result.add_argument("--checkpoint-revision", required=True)
    result.add_argument("--base-id", required=True)
    result.add_argument("--base-revision", required=True)
    result.add_argument("--port", required=True, type=int)
    result.add_argument("--prepared-file", required=True)
    result.add_argument("--data-file", required=True)
    result.add_argument("--evidence-output", required=True)
    result.add_argument("--upstream-root", required=True)
    result.add_argument("--hf-cache", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    values = parser().parse_args(argv)
    if not 1 <= values.port <= 65535:
        raise SystemExit("--port must be 1..65535")
    launch(Config(**vars(values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
