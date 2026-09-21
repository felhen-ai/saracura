"""Minimal local CLI for the self-authored research fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from saracura.backends import DeterministicFixtureBackend
from saracura.calibration import load_calibration
from saracura.contracts import ErrorCode, SaracuraError, parse_request_json
from saracura.runtime import DecisionEngine, default_workflows
from saracura.runtime.engine import calibration_context


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saracura")
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide", help="run the local research-only fixture")
    decide.add_argument("--request", type=Path, required=True)
    decide.add_argument("--calibration", type=Path, required=True)
    decide.add_argument("--timing", action="store_true")
    return parser


def _run_decide(request_path: Path, calibration_path: Path, include_timing: bool) -> int:
    try:
        request = parse_request_json(request_path.read_bytes())
        backend = DeterministicFixtureBackend()
        if len(request.questions) != 1:
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "The minimal CLI accepts one calibration artifact and one question.",
                "/questions",
            )
        question = request.questions[0]
        expected = calibration_context(request, question.id, len(question.criteria), backend)
        artifact = load_calibration(calibration_path, expected)
        engine = DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations={question.id: artifact},
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
    print(json.dumps(public.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "decide":
        return _run_decide(args.request, args.calibration, args.timing)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
