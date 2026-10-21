"""Loopback System One bridge for a pinned, offline Transformers chat model.

The evaluator (research/ptbr_bench.py) speaks POST /v1/systemone. This server turns each question into one chat
message with letter-coded options (A..Z, in the order received), greedily generates at most 32 tokens and accepts
exactly one uppercase letter. Anything else fails the item: no repair, no retry, no fallback, no constrained decoding.
Boolean questions use A = Não, B = Sim and return a discrete 0/1, never a probability or a confidence.

The core (request validation, prompt rendering, parsing, HTTP) is standard library only. torch and transformers are
imported only by the CLI at startup, from a pinned local snapshot (local_files_only, no trust_remote_code); nothing is
ever downloaded by this process. Failure contract: malformed request or wrong model 400, body over 1 MiB 413, prompt
plus output over the context 422, busy 429 (one inference at a time), internal failure 500. Invalid model output and
the whole-request deadline answer HTTP 200 with {} per question and a bounded error code, so the evaluator records
no_answer once and never re-generates through its 5xx retry. The deadline is checked between decoding steps and
between questions; a single prefill or kernel cannot be interrupted.

Usage (see research/evaluation-guide.md for the whole recipe):

    python research/llm_bridge.py --model ORG/NAME --revision <40 hex> --device cpu --port 18031 \
        --context 8192 --chat-template-kwargs '{"enable_thinking": false}' --evidence-output evidence.json
"""

import argparse
import contextlib
import hashlib
import http.server
import importlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
PROMPT_VERSION = "ptbr-systemone-letter-v1"
MAX_NEW_TOKENS = 32
MAX_BODY_BYTES = 1024 * 1024
MAX_QUESTIONS = 16
MIN_CHOICES, MAX_CHOICES = 2, 26
MAX_NAME_CHARS = 128
FROZEN_DATE = "2026-10-21"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOUL_OPTIONS = ("Não", "Sim")
HOST = "127.0.0.1"
SOCKET_TIMEOUT = 30
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
RESERVED_TEMPLATE_KWARGS = frozenset(
    {"messages", "conversation", "tools", "documents", "add_generation_prompt", "tokenize"}
)
DTYPES = {
    "cpu": ("float32", "bfloat16"),
    "mps": ("float16", "float32"),
    "cuda": ("bfloat16", "float16", "float32"),
}
AUTO_CLASSES = ("AutoModelForCausalLM", "AutoModelForImageTextToText")

# Prompt pieces are constants so that their hash is part of the execution evidence.
PROMPT_QUESTION = "Pergunta: "
PROMPT_OPTIONS = "Opções:"
PROMPT_ANSWER = (
    "Responda com exatamente uma letra maiúscula, a da opção correta, "
    "sem pontuação, sem explicação e sem nenhum outro texto."
)
PROMPT_SHA256 = hashlib.sha256(
    "\0".join(
        (PROMPT_VERSION, PROMPT_QUESTION, PROMPT_OPTIONS, PROMPT_ANSWER, *NOUL_OPTIONS)
    ).encode()
).hexdigest()

COUNTERS = ("invalid_output", "deadline", "overbudget")


class BridgeError(Exception):
    """Startup or configuration failure with a bounded message."""


class RequestError(Exception):
    def __init__(self, status, code):
        super().__init__(code)
        self.status = status
        self.code = code


class DeadlineExceeded(Exception):
    """Raised by a generator when the whole-request deadline passed during decoding."""


class Overbudget(Exception):
    """Prompt tokens plus the output budget exceed the configured context."""


# ----------------------------------------------------------------------------------------------------------------
# Pure core


@dataclass(frozen=True)
class Question:
    name: str
    kind: str  # "choice" or "noul"
    instructions: str
    keys: tuple = ()
    descriptions: tuple = ()


@dataclass(frozen=True)
class ParsedRequest:
    state: str
    questions: tuple = field(default_factory=tuple)


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def _closed(obj, allowed):
    if not isinstance(obj, dict) or set(obj) != set(allowed):
        raise RequestError(400, "invalid_request")


def _text(value):
    if not isinstance(value, str):
        raise RequestError(400, "invalid_request")
    return value


def parse_question(name, raw):
    if not isinstance(raw, dict):
        raise RequestError(400, "invalid_request")
    if raw.get("type") == "noul":
        _closed(raw, ("type", "instructions"))
        return Question(name, "noul", _text(raw["instructions"]))
    if raw.get("type") != "choice":
        raise RequestError(400, "invalid_request")
    _closed(raw, ("type", "instructions", "criteria"))
    criteria = raw["criteria"]
    if not isinstance(criteria, dict) or not MIN_CHOICES <= len(criteria) <= MAX_CHOICES:
        raise RequestError(400, "invalid_request")
    keys = tuple(criteria)  # insertion order is the order received; never sorted or shuffled
    descriptions = tuple(_text(criteria[k]) for k in keys)
    return Question(name, "choice", _text(raw["instructions"]), keys, descriptions)


def parse_request(raw, served_model):
    """Validate the closed request. Raises RequestError before any inference."""
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError(413, "too_large")
    try:
        body = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise RequestError(400, "invalid_json") from None
    _closed(body, ("model", "state", "questions"))
    if _text(body["model"]) != served_model:
        raise RequestError(400, "unknown_model")
    state = _text(body["state"])
    questions = body["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(400, "invalid_request")
    parsed = []
    for name, value in questions.items():
        if not isinstance(name, str) or not 1 <= len(name) <= MAX_NAME_CHARS:
            raise RequestError(400, "invalid_request")
        parsed.append(parse_question(name, value))
    return ParsedRequest(state, tuple(parsed))


def option_labels(question):
    """Option texts in received order (noul is fixed: A Não, B Sim)."""
    return NOUL_OPTIONS if question.kind == "noul" else question.descriptions


def render_user_message(state, question):
    """The single user chat message: state and question verbatim, options letter-coded in received order."""
    options = "\n".join(f"{LETTERS[i]}) {text}" for i, text in enumerate(option_labels(question)))
    return f"{state}\n\n{PROMPT_QUESTION}{question.instructions}\n\n{PROMPT_OPTIONS}\n{options}\n\n{PROMPT_ANSWER}"


def parse_letter(text, option_count):
    """Index of the answered option, or None. Exactly one uppercase A..Z inside the option count."""
    stripped = text.strip()
    if len(stripped) != 1 or stripped not in LETTERS:
        return None
    index = LETTERS.index(stripped)
    return index if index < option_count else None


def answer_for(question, index):
    if question.kind == "noul":
        return {
            "noul": index
        }  # 0 = Não, 1 = Sim: a discrete decision, not a calibrated probability
    return {"choice": question.keys[index]}


# ----------------------------------------------------------------------------------------------------------------
# Bridge: validation + one inference at a time + counters


def canonical_bytes(obj):
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def publish_exclusive(path, payload):
    """Write payload so the final path appears complete or not at all, and never replaces an existing file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise BridgeError("refusing to overwrite existing evidence file") from None
        except OSError as e:
            raise BridgeError(
                f"could not create evidence file without overwriting: {e.strerror}"
            ) from None
    finally:
        Path(tmp).unlink(missing_ok=True)


class Bridge:
    """backend needs: context (int), encode(text) -> list[int], generate(ids, deadline) -> str."""

    def __init__(self, backend, served_model, evidence, generation_timeout, clock=time.monotonic):
        self.backend = backend
        self.served_model = served_model
        self.evidence = evidence
        self.evidence_bytes = canonical_bytes(evidence)
        self.evidence_sha256 = hashlib.sha256(self.evidence_bytes).hexdigest()
        self.generation_timeout = generation_timeout
        self.clock = clock
        self._inference = threading.Lock()
        self._counter_lock = threading.Lock()
        self._counters = dict.fromkeys(COUNTERS, 0)

    def health(self):
        with self._counter_lock:
            counters = dict(self._counters)
        return {
            "evidence": self.evidence,
            "evidence_sha256": self.evidence_sha256,
            "counters": counters,
        }

    def _count(self, name):
        with self._counter_lock:
            self._counters[name] += 1

    def handle(self, raw):
        """(HTTP status, JSON payload). Never raises; payloads carry bounded codes, never texts."""
        try:
            request = parse_request(raw, self.served_model)
        except RequestError as e:
            return e.status, {"error": e.code}
        if not self._inference.acquire(blocking=False):
            return 429, {"error": "busy"}
        try:
            return self._run(request)
        except Overbudget:
            self._count("overbudget")
            return 422, {"error": "context_overflow"}
        except Exception:
            return 500, {"error": "internal"}
        finally:
            self._inference.release()

    def _run(self, request):
        deadline = self.clock() + self.generation_timeout
        encoded = []
        for q in request.questions:
            ids = self.backend.encode(render_user_message(request.state, q))
            if len(ids) + MAX_NEW_TOKENS > self.backend.context:
                raise Overbudget
            encoded.append(ids)
        answers = {}
        error = None
        for q, ids in zip(request.questions, encoded, strict=True):
            try:
                if self.clock() >= deadline:
                    raise DeadlineExceeded
                text = self.backend.generate(ids, deadline)
                if self.clock() >= deadline:
                    raise DeadlineExceeded  # the deadline wins even over a valid partial letter
            except DeadlineExceeded:
                self._count("deadline")
                return 200, {
                    "answers": {r.name: {} for r in request.questions},
                    "error": "deadline",
                }
            index = parse_letter(text, len(option_labels(q)))
            if index is None:
                self._count("invalid_output")
                answers[q.name] = {}
                error = "invalid_output"
            else:
                answers[q.name] = answer_for(q, index)
        payload = {"answers": answers}
        if error:
            payload["error"] = error
        return 200, payload


# ----------------------------------------------------------------------------------------------------------------
# HTTP


class _Handler(http.server.BaseHTTPRequestHandler):
    timeout = SOCKET_TIMEOUT
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # no request, body or error text is ever logged
        pass

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # client went away; nothing is re-queued
        self.close_connection = True

    def do_GET(self):
        if self.path == "/health":
            self._send(200, self.server.bridge.health())
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("content-length", ""))
        except ValueError:
            self._send(400, {"error": "invalid_request"})
            return
        if length < 0:
            self._send(400, {"error": "invalid_request"})
        elif length > MAX_BODY_BYTES:
            self._send(413, {"error": "too_large"})
        else:
            try:
                raw = self.rfile.read(length)
            except OSError:
                return
            if len(raw) != length:
                self._send(400, {"error": "invalid_request"})
                return
            self._send(*self.server.bridge.handle(raw))


def make_server(bridge, port=0):
    """Threaded HTTP server bound to loopback only."""
    server = http.server.ThreadingHTTPServer((HOST, port), _Handler)
    server.daemon_threads = True
    server.bridge = bridge
    return server


# ----------------------------------------------------------------------------------------------------------------
# Optional Transformers backend (heavy imports happen only here, at explicit CLI startup)


@dataclass(frozen=True)
class Config:
    model: str
    revision: str
    device: str
    context: int
    threads: int
    template_kwargs: dict
    dtype: str | None = None
    auto_class: str = "AutoModelForCausalLM"
    template_date: str = FROZEN_DATE
    generation_timeout: float = 90.0
    alias: str | None = None


def resolve_dtype(device, requested):
    allowed = DTYPES[device]
    if requested is None:
        return allowed[0]
    if requested not in allowed:
        raise BridgeError(f"dtype {requested} is not supported on {device}")
    return requested


@contextlib.contextmanager
def frozen_clock(module, date):
    """Make `datetime.now()` inside `module` return `date`, which is what the library's strftime_now helper reads."""
    original = getattr(module, "datetime", None)
    if original is None or not hasattr(original, "now"):
        raise BridgeError(
            "chat template helper strftime_now cannot be frozen with this transformers"
        )
    year, month, day = (int(p) for p in date.split("-"))

    class Frozen(original):
        @classmethod
        def now(cls, tz=None):
            return cls(year, month, day, tzinfo=tz)

    module.datetime = Frozen
    try:
        yield
    finally:
        module.datetime = original


class TransformersBackend:
    def __init__(self, torch, transformers, tokenizer, model, config, kwargs, clock_module):
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = config.device
        self.context = config.context
        self.kwargs = kwargs
        self.date = config.template_date
        self.clock_module = clock_module
        self._stopping_base = transformers.StoppingCriteria
        self._criteria_list = transformers.StoppingCriteriaList

    def _render(self, text, **extra):
        messages = [{"role": "user", "content": text}]
        cm = (
            frozen_clock(self.clock_module, self.date)
            if self.clock_module is not None
            else contextlib.nullcontext()
        )
        with cm:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.kwargs, **extra
            )

    def probe_clock(self):
        """Fail at startup unless a time-dependent template renders the frozen date."""
        if self.clock_module is None:
            return
        out = self._render("x", chat_template="{{ strftime_now('%Y-%m-%d') }}")
        if out != self.date:
            raise BridgeError(
                "chat template strftime_now is not deterministic with this transformers"
            )

    def encode(self, text):
        prompt = self._render(text)
        return list(self.tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])

    def generate(self, ids, deadline, clock=time.monotonic):
        torch = self.torch
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        state = {"expired": False}

        class Deadline(self._stopping_base):
            def __call__(self, input_ids, scores, **kwargs):
                state["expired"] = state["expired"] or clock() >= deadline
                return torch.full(
                    (input_ids.shape[0],),
                    state["expired"],
                    dtype=torch.bool,
                    device=input_ids.device,
                )

        with torch.inference_mode():
            out = self.model.generate(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                stopping_criteria=self._criteria_list([Deadline()]),
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            )
        if state["expired"]:
            raise DeadlineExceeded
        return self.tokenizer.decode(out[0][len(ids) :], skip_special_tokens=True)


def parse_template_kwargs(text):
    try:
        value = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except ValueError:
        raise BridgeError("--chat-template-kwargs must be a JSON object") from None
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise BridgeError("--chat-template-kwargs must be a JSON object")
    if RESERVED_TEMPLATE_KWARGS & set(value):
        raise BridgeError("--chat-template-kwargs uses a reserved name")
    if any(not isinstance(v, str | int | float | bool) for v in value.values()):
        raise BridgeError("--chat-template-kwargs values must be scalars")
    return {unicodedata.normalize("NFC", k): v for k, v in value.items()}


def load_transformers_backend(config, import_module=importlib.import_module):
    """Load the pinned snapshot from the local cache only. Returns (backend, backend evidence)."""
    torch = import_module("torch")
    transformers = import_module("transformers")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise BridgeError("device cuda is not available")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise BridgeError("device mps is not available")
    dtype = resolve_dtype(config.device, config.dtype)
    torch.set_num_threads(config.threads)
    pinned = {"revision": config.revision, "local_files_only": True, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, **pinned)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise BridgeError("tokenizer has no single official chat template")
    model = getattr(transformers, config.auto_class).from_pretrained(
        config.model, attn_implementation="eager", dtype=getattr(torch, dtype), **pinned
    )
    model.to(config.device)
    model.eval()
    kwargs = dict(config.template_kwargs)
    if "date_string" in template and "date_string" not in kwargs:
        kwargs["date_string"] = config.template_date
    clock_module = None
    if "strftime_now" in template:
        clock_module = import_module("transformers.utils.chat_template_utils")
    backend = TransformersBackend(
        torch, transformers, tokenizer, model, config, kwargs, clock_module
    )
    backend.probe_clock()
    info = {
        "runtime": "transformers",
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "python": ".".join(str(p) for p in sys.version_info[:3]),
        "dtype": dtype,
        "device": config.device,
        "threads": config.threads,
        "attention": "eager",
        "auto_class": config.auto_class,
        "template_kwargs": kwargs,
        "template_date": config.template_date,
        "strftime_now_frozen": clock_module is not None,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }
    return backend, info


def build_evidence(config, info, code_sha256):
    """Immutable execution manifest. No request text, endpoint, or local path."""
    return {
        "schema_version": SCHEMA_VERSION,
        "bridge_sha256": code_sha256,
        "model": {
            "id": config.model,
            "alias": config.alias or config.model,
            "revision": config.revision,
            "tokenizer_revision": config.revision,
        },
        "backend": info,
        "limits": {
            "context": config.context,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation_timeout_seconds": config.generation_timeout,
            "max_body_bytes": MAX_BODY_BYTES,
            "max_questions": MAX_QUESTIONS,
        },
        "prompt": {"version": PROMPT_VERSION, "sha256": PROMPT_SHA256},
        "decoding": "greedy",
    }


# ----------------------------------------------------------------------------------------------------------------
# CLI


def _model(text):
    if not MODEL_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected ORG/NAME")
    return text


def _revision(text):
    if not REVISION_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a 40-hex commit")
    return text


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _positive_float(text):
    value = float(text)
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError("expected a positive number")
    return value


def _date(text):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD")
    return text


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model", required=True, type=_model, help="Hub repo id of the already downloaded snapshot"
    )
    ap.add_argument(
        "--revision", required=True, type=_revision, help="40-hex commit of that snapshot"
    )
    ap.add_argument("--device", required=True, choices=sorted(DTYPES))
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument(
        "--context", required=True, type=_positive_int, help="effective context in tokens"
    )
    ap.add_argument(
        "--chat-template-kwargs",
        required=True,
        help="JSON object, e.g. '{\"enable_thinking\": false}'",
    )
    ap.add_argument("--threads", type=_positive_int, default=4)
    ap.add_argument("--dtype", choices=sorted({d for v in DTYPES.values() for d in v}))
    ap.add_argument("--auto-class", choices=AUTO_CLASSES, default=AUTO_CLASSES[0])
    ap.add_argument(
        "--alias", type=_model, help="model name the evaluator sends (default: --model)"
    )
    ap.add_argument("--generation-timeout", type=_positive_float, default=90.0)
    ap.add_argument("--template-date", type=_date, default=FROZEN_DATE)
    ap.add_argument(
        "--evidence-output", type=Path, help="write the execution evidence here, never overwriting"
    )
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be 1..65535")
    # Offline before any heavy import: this process never downloads anything.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        config = Config(
            model=args.model,
            revision=args.revision,
            device=args.device,
            context=args.context,
            threads=args.threads,
            template_kwargs=parse_template_kwargs(args.chat_template_kwargs),
            dtype=args.dtype,
            auto_class=args.auto_class,
            template_date=args.template_date,
            generation_timeout=args.generation_timeout,
            alias=args.alias,
        )
        backend, info = load_transformers_backend(config)
        evidence = build_evidence(
            config, info, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        bridge = Bridge(backend, config.alias or config.model, evidence, config.generation_timeout)
        if args.evidence_output:
            publish_exclusive(args.evidence_output, bridge.evidence_bytes)
    except BridgeError as e:
        raise SystemExit(f"error: {e}") from None
    server = make_server(bridge, args.port)
    print(
        f"listening on {HOST}:{server.server_address[1]} evidence_sha256={bridge.evidence_sha256}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
