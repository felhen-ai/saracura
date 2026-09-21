"""Fail-closed calibration loading and durable atomic writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError

from saracura.calibration.models import CalibrationArtifact, CalibrationContext
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.serialization import canonical_json_bytes


def _validate_compatibility(
    artifact: CalibrationArtifact,
    expected: CalibrationContext,
) -> None:
    actual = artifact.context().model_dump(mode="json")
    wanted = expected.model_dump(mode="json")
    mismatches = sorted(key for key, value in wanted.items() if actual[key] != value)
    if mismatches:
        raise SaracuraError(
            ErrorCode.CALIBRATION_INCOMPATIBLE,
            "Calibration artifact is incompatible with the request and runtime.",
            "/calibration",
            details={"mismatched_axes": cast(JsonValue, mismatches)},
        )


def load_calibration(path: Path, expected: CalibrationContext) -> CalibrationArtifact:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise SaracuraError(
            ErrorCode.CALIBRATION_MISSING,
            "Calibration artifact was not found.",
            "/calibration",
        ) from error

    try:
        artifact = CalibrationArtifact.model_validate_json(raw)
    except (ValidationError, ValueError) as error:
        details: dict[str, JsonValue] = {}
        if isinstance(error, ValidationError):
            violations = [
                {
                    "type": item["type"],
                    "path": "/" + "/".join(str(part) for part in item["loc"]),
                }
                for item in error.errors(
                    include_url=False,
                    include_context=False,
                    include_input=False,
                )
            ]
            details = {"violations": cast(JsonValue, violations)}
        raise SaracuraError(
            ErrorCode.CALIBRATION_INCOMPATIBLE,
            "Calibration artifact failed schema validation.",
            "/calibration",
            details=details,
        ) from error
    _validate_compatibility(artifact, expected)
    return artifact


def write_calibration_atomic(
    path: Path,
    artifact: CalibrationArtifact,
) -> None:
    """Create a canonical artifact atomically and never replace an existing revision."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(artifact.model_dump(mode="json")) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic create-if-absent operation. Unlike an
        # existence precheck followed by replace, it cannot overwrite a file
        # created concurrently by another writer.
        os.link(temporary, path)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
