"""V3-only 32-choice SystemOne bridge.

This is deliberately a small stdlib wrapper around :mod:`llm_bridge`; it never
changes the V1 26-letter protocol.  Invoke this file, rather than the legacy
bridge, when serving the V3 Câmara task.
"""

import hashlib
import sys
from pathlib import Path

import llm_bridge as legacy

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
PROMPT_VERSION = "ptbr-systemone-letter-v3-camara32"
LEGACY_BRIDGE_SHA256 = hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()
PROMPT_ANSWER = (
    "Responda com exatamente um identificador sensível a maiúsculas e minúsculas, "
    "o da opção correta, sem pontuação, sem explicação e sem nenhum outro texto."
)
PROMPT_SHA256 = hashlib.sha256(
    "\0".join(
        (
            PROMPT_VERSION,
            legacy.PROMPT_QUESTION,
            legacy.PROMPT_OPTIONS,
            PROMPT_ANSWER,
            ALPHABET,
            LEGACY_BRIDGE_SHA256,
            *legacy.NOUL_OPTIONS,
        )
    ).encode()
).hexdigest()


def parse_identifier(text, option_count):
    """Strict V3 readout: ``f`` is option 31 while ``F`` remains option 5."""
    value = text.strip()
    if len(value) != 1 or value not in ALPHABET:
        return None
    index = ALPHABET.index(value)
    return index if index < option_count else None


def render_user_message(state, question):
    options = "\n".join(
        f"{ALPHABET[i]}) {text}" for i, text in enumerate(legacy.option_labels(question))
    )
    return f"{state}\n\n{legacy.PROMPT_QUESTION}{question.instructions}\n\n{legacy.PROMPT_OPTIONS}\n{options}\n\n{PROMPT_ANSWER}"


def validate_32_single_tokens(tokenizer):
    """Return immutable token evidence, rejecting non-reversible or shared tokens."""
    mapping, seen = {}, set()
    for identifier in ALPHABET:
        ids = list(tokenizer(identifier, add_special_tokens=False, truncation=False)["input_ids"])
        if (
            len(ids) != 1
            or tokenizer.decode(ids, skip_special_tokens=False) != identifier
            or ids[0] in seen
        ):
            raise legacy.BridgeError("V3 identifier is not one distinct reversible token")
        raw = tokenizer.convert_ids_to_tokens(ids[0])
        if not isinstance(raw, str):
            raise legacy.BridgeError("V3 identifier has no raw token")
        mapping[identifier] = {"id": ids[0], "token": raw}
        seen.add(ids[0])
    return mapping


def parse_question(name, raw):
    """V3's sole request-schema difference is the 32-choice ceiling."""
    if not isinstance(raw, dict):
        raise legacy.RequestError(400, "invalid_request")
    if raw.get("type") == "noul":
        legacy._closed(raw, ("type", "instructions"))
        return legacy.Question(name, "noul", legacy._text(raw["instructions"]))
    if raw.get("type") != "choice":
        raise legacy.RequestError(400, "invalid_request")
    legacy._closed(raw, ("type", "instructions", "criteria"))
    criteria = raw["criteria"]
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= len(ALPHABET):
        raise legacy.RequestError(400, "invalid_request")
    keys = tuple(legacy._text(key) for key in criteria)
    return legacy.Question(
        name,
        "choice",
        legacy._text(raw["instructions"]),
        keys,
        tuple(legacy._text(criteria[key]) for key in keys),
    )


def parse_request(raw, served_model):
    if len(raw) > legacy.MAX_BODY_BYTES:
        raise legacy.RequestError(413, "too_large")
    try:
        body = legacy.json.loads(raw.decode("utf-8"), object_pairs_hook=legacy._no_duplicate_keys)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise legacy.RequestError(400, "invalid_json") from None
    legacy._closed(body, ("model", "state", "questions"))
    if legacy._text(body["model"]) != served_model:
        raise legacy.RequestError(400, "unknown_model")
    questions = body["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= legacy.MAX_QUESTIONS:
        raise legacy.RequestError(400, "invalid_request")
    parsed = []
    for name, value in questions.items():
        if not 1 <= len(legacy._text(name)) <= legacy.MAX_NAME_CHARS:
            raise legacy.RequestError(400, "invalid_request")
        parsed.append(parse_question(name, value))
    return legacy.ParsedRequest(legacy._text(body["state"]), tuple(parsed))


class V3Bridge(legacy.Bridge):
    def handle(self, raw):
        try:
            request = parse_request(raw, self.served_model)
        except legacy.RequestError as error:
            return error.status, {"error": error.code}
        if not self._inference.acquire(blocking=False):
            return 429, {"error": "busy"}
        try:
            return self._run(request)
        except legacy.Overbudget:
            self._count("overbudget")
            return 422, {"error": "context_overflow"}
        except legacy.UnsupportedLetterTokens:
            return 422, {"error": "unsupported_letter_tokens"}
        except Exception:
            return 500, {"error": "internal"}
        finally:
            self._inference.release()

    def _run(self, request):
        deadline = self.clock() + self.generation_timeout
        encoded = []
        for q in request.questions:
            letters = ALPHABET[: len(legacy.option_labels(q))]
            if q.kind == "choice" and len(letters) > len(ALPHABET):
                raise legacy.RequestError(400, "invalid_request")
            letter_ids = None
            if self.readout == "constrained-letter":
                letter_ids = self.backend.letter_token_ids(letters)
                if tuple(letter_ids) != tuple(letters) or len(set(letter_ids.values())) != len(
                    letter_ids
                ):
                    raise legacy.UnsupportedLetterTokens
            ids = self.backend.encode(render_user_message(request.state, q))
            budget = (
                legacy.CONSTRAINED_NEW_TOKENS
                if self.readout == "constrained-letter"
                else legacy.MAX_NEW_TOKENS
            )
            if len(ids) + budget > self.backend.context:
                raise legacy.Overbudget
            encoded.append((ids, letter_ids))
        answers, error = {}, None
        for q, (ids, letter_ids) in zip(request.questions, encoded, strict=True):
            try:
                if self.clock() >= deadline:
                    raise legacy.DeadlineExceeded
                index = (
                    self.backend.generate_choice(ids, deadline, letter_ids)
                    if self.readout == "constrained-letter"
                    else parse_identifier(
                        self.backend.generate(ids, deadline), len(legacy.option_labels(q))
                    )
                )
                if self.clock() >= deadline:
                    raise legacy.DeadlineExceeded
            except legacy.DeadlineExceeded:
                self._count("deadline")
                return 200, {
                    "answers": {r.name: {} for r in request.questions},
                    "error": "deadline",
                }
            except legacy.NonfiniteLogits:
                self._count("invalid_output")
                return 200, {
                    "answers": {r.name: {} for r in request.questions},
                    "error": "invalid_output",
                }
            if index is None:
                self._count("invalid_output")
                answers[q.name] = {}
                error = "invalid_output"
            else:
                answers[q.name] = legacy.answer_for(q, index)
        return 200, {"answers": answers, **({"error": error} if error else {})}


def build_evidence(config, info, code_sha256):
    result = legacy.build_evidence(config, info, code_sha256)
    result["prompt"] = {
        "version": PROMPT_VERSION,
        "sha256": PROMPT_SHA256,
        "alphabet": ALPHABET,
        "answer": PROMPT_ANSWER,
    }
    result["protocol"] = "v3-camara32-only"
    result["source_hashes"] = {
        "wrapper": code_sha256,
        "legacy_bridge": LEGACY_BRIDGE_SHA256,
    }
    return result


def main(argv=None):
    args = legacy.build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be 1..65535")
    import os

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    try:
        config = legacy.Config(
            args.model,
            args.revision,
            args.device,
            args.context,
            args.threads,
            legacy.parse_template_kwargs(args.chat_template_kwargs),
            args.dtype,
            args.auto_class,
            args.template_date,
            args.generation_timeout,
            args.alias,
            args.readout,
        )
        backend, info = legacy.load_transformers_backend(config)
        if args.readout == "constrained-letter":
            backend._letter_tokens = validate_32_single_tokens(backend.tokenizer)
            info["letter_tokens"] = backend._letter_tokens
        evidence = build_evidence(
            config,
            info,
            hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest(),
        )
        bridge = V3Bridge(
            backend, args.alias or args.model, evidence, args.generation_timeout, args.readout
        )
        if args.evidence_output:
            legacy.publish_exclusive(args.evidence_output, bridge.evidence_bytes)
    except legacy.BridgeError as e:
        raise SystemExit(f"error: {e}") from None
    server = legacy.make_server(bridge, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
