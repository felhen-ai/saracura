"""Local-only v1alpha1 CLI for explicit research-only backends."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Never, cast

from saracura.backends import (
    DeterministicFixtureBackend,
    LayaUniversalBackend,
    MiniLMRoutingBackend,
)
from saracura.calibration import (
    ResearchCalibrationArtifact,
    create_identity_calibration,
    load_calibration_envelope,
    validate_calibration_compatibility,
)
from saracura.contracts import ErrorCode, SaracuraError, parse_request_json
from saracura.laya_snapshot import load_laya_candidate
from saracura.runtime import (
    MINILM_ROUTING_WORKFLOW_ID,
    MINILM_ROUTING_WORKFLOW_REVISION,
    PHASE4A_IDENTITY_PROFILE,
    DecisionEngine,
    default_workflows,
)
from saracura.runtime.engine import MAX_STATE_PAYLOAD_BYTES, calibration_context
from saracura.serialization import serialize_state
from saracura.verified_bytes import read_public_external_file

MAX_REQUEST_BYTES = 1_000_000


class _ArgumentFailure(ValueError):
    """A parser failure rendered through the normal JSON error envelope."""


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise _ArgumentFailure(message)

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        if status == 0:
            super().exit(status, message)
        raise _ArgumentFailure(message or "invalid command arguments")


def _add_minilm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-snapshot", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-capsule", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"))


def _add_laya_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-snapshot", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(prog="saracura")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_NonExitingArgumentParser,
    )

    decide = commands.add_parser("decide", help="run a local research-only backend")
    decide.add_argument(
        "--backend", choices=("fixture", "minilm-routing", "laya-universal"), default="fixture"
    )
    decide.add_argument("--request", type=Path, required=True)
    decide.add_argument("--calibration", type=Path)
    decide.add_argument("--timing", action="store_true")
    _add_minilm_arguments(decide)
    _add_laya_arguments(decide)

    describe = commands.add_parser(
        "describe-backend",
        help="load a local backend and emit its public immutable reference",
    )
    describe.add_argument("--backend", choices=("minilm-routing", "laya-universal"), required=True)
    _add_minilm_arguments(describe)
    _add_laya_arguments(describe)

    identity = commands.add_parser(
        "create-identity-calibration",
        help="create the Phase 4A no-fit identity calibration artifact",
    )
    identity.add_argument("--backend", choices=("minilm-routing",), required=True)
    identity.add_argument("--request", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    identity.add_argument("--created-at", required=True)
    _add_minilm_arguments(identity)
    return parser


def _require_minilm_arguments(values: argparse.Namespace) -> MiniLMRoutingBackend:
    if any(getattr(values, name, None) is None for name in ("encoder_snapshot", "device")):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "MiniLM commands require explicit local artifact paths and a device.",
            "/",
        )
    capsule = values.training_capsule
    synthetic_pair = values.training_manifest is not None and values.checkpoint is not None
    if (capsule is None and not synthetic_pair) or (
        capsule is not None
        and (values.training_manifest is not None or values.checkpoint is not None)
    ):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "MiniLM commands require one sealed human capsule or the Phase 4A synthetic pair.",
            "/",
        )
    return MiniLMRoutingBackend(
        encoder_snapshot=values.encoder_snapshot,
        training_manifest=values.training_manifest,
        checkpoint=values.checkpoint,
        training_capsule=capsule,
        device=values.device,
    )


def _require_laya_arguments(values: argparse.Namespace) -> LayaUniversalBackend:
    if values.model_snapshot is None or values.device is None:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Laya universal commands require an explicit local snapshot and device.",
            "/",
        )
    if getattr(values, "calibration", None) is not None or any(
        getattr(values, name, None) is not None
        for name in ("encoder_snapshot", "training_manifest", "checkpoint", "training_capsule")
    ):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Laya universal commands reject calibration and MiniLM-only arguments.",
            "/backend",
        )
    return LayaUniversalBackend(model_snapshot=values.model_snapshot, device=values.device)


def _backend_for_decide(
    values: argparse.Namespace,
) -> DeterministicFixtureBackend | MiniLMRoutingBackend | LayaUniversalBackend:
    if values.backend == "fixture":
        if values.calibration is None:
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Fixture decisions require a calibration artifact.",
                "/calibration",
            )
        if any(
            getattr(values, name, None) is not None
            for name in (
                "encoder_snapshot",
                "training_manifest",
                "checkpoint",
                "training_capsule",
                "device",
                "model_snapshot",
            )
        ):
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Fixture decisions reject MiniLM-only arguments.",
                "/backend",
            )
        return DeterministicFixtureBackend()
    if values.backend == "laya-universal":
        return _require_laya_arguments(values)
    if values.calibration is None:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "MiniLM decisions require a calibration artifact.",
            "/calibration",
        )
    if values.model_snapshot is not None:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "MiniLM decisions reject Laya-only arguments.",
            "/backend",
        )
    return _require_minilm_arguments(values)


def _prevalidate_request(
    request_path: Path, *, execution_tier: Literal["compiled", "universal"]
) -> tuple[Any, bytes]:
    """Validate every request-only Laya gate before optional backend construction."""

    request = parse_request_json(read_public_external_file(request_path, maximum=MAX_REQUEST_BYTES))
    if execution_tier == "compiled" and len(request.questions) != 1:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "The local CLI accepts one calibration artifact and one question.",
            "/questions",
        )
    default_workflows().validate(request, execution_tier=execution_tier)
    if request.model in {"latest", "main", "master"}:
        raise SaracuraError(
            ErrorCode.MODEL_ALIAS_FORBIDDEN,
            "The immutable model revision is not available.",
            "/model",
        )
    state_payload = serialize_state(request)
    if len(state_payload) > MAX_STATE_PAYLOAD_BYTES:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Serialized state exceeds the research runtime byte limit.",
            "/state",
            details={"max_bytes": MAX_STATE_PAYLOAD_BYTES},
        )
    if execution_tier == "universal":
        if not _is_nfc(request.state) or not all(
            _is_nfc(value)
            for value in (
                request.locale,
                request.domain,
                *(question.instruction for question in request.questions),
                *(
                    criterion.id
                    for question in request.questions
                    for criterion in question.criteria
                ),
                *(
                    criterion.description
                    for question in request.questions
                    for criterion in question.criteria
                ),
            )
        ):
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID, "Universal input must already be NFC.", "/"
            )
        candidate = load_laya_candidate()
        if request.model != candidate.model_revision:
            raise SaracuraError(
                ErrorCode.MODEL_NOT_FOUND,
                "The immutable model revision is not available.",
                "/model",
            )
    return request, state_payload


def _is_nfc(value: object) -> bool:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value) == value
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_nfc(key) and _is_nfc(child) for key, child in value.items()
        )
    if isinstance(value, list):
        return all(_is_nfc(child) for child in value)
    return True


def _run_decide(
    request_path: Path,
    calibration_path: Path,
    include_timing: bool,
    *,
    backend: DeterministicFixtureBackend | MiniLMRoutingBackend | None = None,
    backend_factory: (
        Callable[[], DeterministicFixtureBackend | MiniLMRoutingBackend] | None
    ) = None,
) -> int:
    try:
        request, _state_payload = _prevalidate_request(request_path, execution_tier="compiled")
        # This is deliberately request-only validation.  It must complete
        # before the optional backend can import ML packages, inspect a device,
        # or open a model/weight path.
        # The sole envelope read classifies the lane and, for v2, verifies the
        # bundled trust receipt.  It runs before constructing MiniLM and is
        # retained for later compatibility validation without a reread.
        artifact = load_calibration_envelope(calibration_path)
        resolved_backend = backend or (
            backend_factory() if backend_factory is not None else DeterministicFixtureBackend()
        )
        question = request.questions[0]
        profile = (
            PHASE4A_IDENTITY_PROFILE if isinstance(resolved_backend, MiniLMRoutingBackend) else None
        )
        if isinstance(resolved_backend, MiniLMRoutingBackend) and isinstance(
            artifact, ResearchCalibrationArtifact
        ):
            profile = artifact.dataset_profile()
        expected = calibration_context(
            request,
            question.id,
            len(question.criteria),
            resolved_backend,
            profile,
        )
        validate_calibration_compatibility(artifact, expected)
        engine = DecisionEngine(
            backend=resolved_backend,
            workflows=default_workflows(),
            calibrations={question.id: artifact},
            calibration_profiles={question.id: profile} if profile is not None else None,
        )
        response = engine.decide(request, include_timing=include_timing)
        print(response.model_dump_json(indent=2, exclude_none=False))
        return 0
    except FileNotFoundError:
        public = SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Input file was not found.",
            "/",
        )
    except SaracuraError as error:
        public = error
    except OSError:
        public = SaracuraError(
            ErrorCode.INTERNAL_ERROR,
            "Local input could not be read.",
            "/",
        )
    except Exception:
        public = SaracuraError(
            ErrorCode.INTERNAL_ERROR,
            "Unexpected internal error.",
            "/",
        )
    _print_error(public)
    return 2


def _run_describe(values: argparse.Namespace) -> int:
    if values.backend == "laya-universal":
        backend = _require_laya_arguments(values)
        try:
            backend.prepare()
            payload = {
                "model": backend.model.model_dump(mode="json"),
                "execution_tier": "universal",
                "workflow": {
                    "id": "universal-choice",
                    "revision": "phase4d-laya.v1",
                    "locales": ["pt-BR", "en"],
                    "question_type": "choice",
                    "questions": {"minimum": 1, "maximum": 10},
                    "criteria_per_question": {"minimum": 2, "maximum": 20},
                },
                "capacity": {
                    "input_tokens_maximum": 1024,
                    "decision_head_prefix_tokens_maximum": 256,
                    "microbatch_questions": 1,
                },
                "response": {
                    "status": "uncalibrated",
                    "score_semantics": "ranking_weights",
                    "abstained": True,
                    "calibration": None,
                    "automation_allowed": False,
                },
                "runtime_disposition": "research_only_unresolved_provenance",
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 0
        finally:
            backend.close()
    if values.model_snapshot is not None:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "MiniLM commands reject Laya-only arguments.",
            "/backend",
        )
    minilm_backend = _require_minilm_arguments(values)
    minilm_payload: dict[str, Any] = {
        "model": minilm_backend.model.model_dump(mode="json"),
        "architecture_config_sha256": (
            minilm_backend.calibration_metadata.architecture_config_sha256
        ),
        "execution_path": minilm_backend.execution_path,
        "workflow": {
            "id": MINILM_ROUTING_WORKFLOW_ID,
            "revision": MINILM_ROUTING_WORKFLOW_REVISION,
        },
        "identity_profile": PHASE4A_IDENTITY_PROFILE.model_dump(mode="json"),
    }
    print(json.dumps(minilm_payload, ensure_ascii=False, sort_keys=True))
    return 0


def _run_laya_decide(values: argparse.Namespace) -> int:
    """Run the explicit Laya lane, constructing it only after request-only gates."""

    try:
        request, _state_payload = _prevalidate_request(values.request, execution_tier="universal")
        backend = _require_laya_arguments(values)
        try:
            response = DecisionEngine(
                backend=backend,
                workflows=default_workflows(),
                calibrations={},
            ).decide(request, include_timing=values.timing)
            print(response.model_dump_json(indent=2, exclude_none=False))
            return 0
        finally:
            backend.close()
    except FileNotFoundError:
        public = SaracuraError(ErrorCode.REQUEST_INVALID, "Input file was not found.", "/")
    except SaracuraError as error:
        public = error
    except OSError:
        public = SaracuraError(ErrorCode.INTERNAL_ERROR, "Local input could not be read.", "/")
    except Exception:
        public = SaracuraError(ErrorCode.INTERNAL_ERROR, "Unexpected internal error.", "/")
    _print_error(public)
    return 2


def _run_create_identity(values: argparse.Namespace) -> int:
    request = parse_request_json(
        read_public_external_file(values.request, maximum=MAX_REQUEST_BYTES)
    )
    backend = _require_minilm_arguments(values)
    if request.model != backend.model.revision:
        raise SaracuraError(
            ErrorCode.MODEL_NOT_FOUND,
            "The immutable model revision is not available.",
            "/model",
        )
    if len(request.questions) != 1:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Identity calibration accepts one immutable MiniLM question.",
            "/questions",
        )
    default_workflows().validate(request)
    question = request.questions[0]
    context = calibration_context(
        request,
        question.id,
        len(question.criteria),
        backend,
        PHASE4A_IDENTITY_PROFILE,
    )
    artifact = create_identity_calibration(values.output, context, values.created_at)
    print(artifact.model_dump_json(indent=2))
    return 0


def _print_error(error: SaracuraError) -> None:
    print(json.dumps(error.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    try:
        values = _parser().parse_args(argv)
        if values.command == "decide":
            if values.backend == "laya-universal":
                return _run_laya_decide(values)
            if values.calibration is None:
                raise SaracuraError(
                    ErrorCode.REQUEST_INVALID,
                    "Fixture and MiniLM decisions require a calibration artifact.",
                    "/calibration",
                )
            return _run_decide(
                values.request,
                values.calibration,
                values.timing,
                backend_factory=lambda: cast(
                    DeterministicFixtureBackend | MiniLMRoutingBackend,
                    _backend_for_decide(values),
                ),
            )
        if values.command == "describe-backend":
            return _run_describe(values)
        if values.command == "create-identity-calibration":
            return _run_create_identity(values)
        raise _ArgumentFailure("unknown command")
    except _ArgumentFailure:
        _print_error(
            SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Command arguments are invalid.",
                "/",
            )
        )
    except FileNotFoundError:
        _print_error(SaracuraError(ErrorCode.REQUEST_INVALID, "Input file was not found.", "/"))
    except SaracuraError as error:
        _print_error(error)
    except OSError:
        _print_error(SaracuraError(ErrorCode.INTERNAL_ERROR, "Local input could not be read.", "/"))
    except Exception:
        _print_error(SaracuraError(ErrorCode.INTERNAL_ERROR, "Unexpected internal error.", "/"))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
