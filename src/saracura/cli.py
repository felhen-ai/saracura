"""Local-only v1alpha1 CLI for the fixture and explicit MiniLM research path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Never

from saracura.backends import DeterministicFixtureBackend, MiniLMRoutingBackend
from saracura.calibration import create_identity_calibration, load_calibration
from saracura.contracts import ErrorCode, SaracuraError, parse_request_json
from saracura.runtime import (
    MINILM_ROUTING_WORKFLOW_ID,
    MINILM_ROUTING_WORKFLOW_REVISION,
    PHASE4A_IDENTITY_PROFILE,
    DecisionEngine,
    default_workflows,
)
from saracura.runtime.engine import calibration_context


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
    parser.add_argument("--device", choices=("mps", "cpu"))


def _parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(prog="saracura")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_NonExitingArgumentParser,
    )

    decide = commands.add_parser("decide", help="run a local research-only backend")
    decide.add_argument("--backend", choices=("fixture", "minilm-routing"), default="fixture")
    decide.add_argument("--request", type=Path, required=True)
    decide.add_argument("--calibration", type=Path, required=True)
    decide.add_argument("--timing", action="store_true")
    _add_minilm_arguments(decide)

    describe = commands.add_parser(
        "describe-backend",
        help="load a local backend and emit its public immutable reference",
    )
    describe.add_argument("--backend", choices=("minilm-routing",), required=True)
    _add_minilm_arguments(describe)

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
    if any(
        getattr(values, name, None) is None
        for name in ("encoder_snapshot", "training_manifest", "checkpoint", "device")
    ):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "MiniLM commands require explicit local artifact paths and a device.",
            "/",
        )
    return MiniLMRoutingBackend(
        encoder_snapshot=values.encoder_snapshot,
        training_manifest=values.training_manifest,
        checkpoint=values.checkpoint,
        device=values.device,
    )


def _backend_for_decide(
    values: argparse.Namespace,
) -> DeterministicFixtureBackend | MiniLMRoutingBackend:
    if values.backend == "fixture":
        if any(
            getattr(values, name, None) is not None
            for name in ("encoder_snapshot", "training_manifest", "checkpoint", "device")
        ):
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Fixture decisions reject MiniLM-only arguments.",
                "/backend",
            )
        return DeterministicFixtureBackend()
    return _require_minilm_arguments(values)


def _run_decide(
    request_path: Path,
    calibration_path: Path,
    include_timing: bool,
    *,
    backend: DeterministicFixtureBackend | MiniLMRoutingBackend | None = None,
) -> int:
    try:
        request = parse_request_json(request_path.read_bytes())
        resolved_backend = backend or DeterministicFixtureBackend()
        if len(request.questions) != 1:
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "The local CLI accepts one calibration artifact and one question.",
                "/questions",
            )
        question = request.questions[0]
        profile = (
            PHASE4A_IDENTITY_PROFILE if isinstance(resolved_backend, MiniLMRoutingBackend) else None
        )
        expected = calibration_context(
            request,
            question.id,
            len(question.criteria),
            resolved_backend,
            profile,
        )
        artifact = load_calibration(calibration_path, expected)
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
    backend = _require_minilm_arguments(values)
    payload: dict[str, Any] = {
        "model": backend.model.model_dump(mode="json"),
        "architecture_config_sha256": backend.calibration_metadata.architecture_config_sha256,
        "execution_path": backend.execution_path,
        "workflow": {
            "id": MINILM_ROUTING_WORKFLOW_ID,
            "revision": MINILM_ROUTING_WORKFLOW_REVISION,
        },
        "identity_profile": PHASE4A_IDENTITY_PROFILE.model_dump(mode="json"),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _run_create_identity(values: argparse.Namespace) -> int:
    request = parse_request_json(values.request.read_bytes())
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
            return _run_decide(
                values.request,
                values.calibration,
                values.timing,
                backend=_backend_for_decide(values),
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
