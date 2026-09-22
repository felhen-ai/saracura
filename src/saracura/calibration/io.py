"""Fail-closed calibration loading and durable atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError

from saracura.calibration.models import (
    AnyCalibrationArtifact,
    CalibrationArtifact,
    CalibrationContext,
    IdentityCalibrationArtifact,
    IdentityParameters,
    identity_calibration_id,
)
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.serialization import canonical_json_bytes


def _validate_compatibility(
    artifact: AnyCalibrationArtifact,
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


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate calibration JSON key")
        result[key] = value
    return result


def _parse_artifact(raw: bytes) -> AnyCalibrationArtifact:
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("calibration root must be an object")
    method = payload.get("method")
    if method == "temperature-scaling":
        return CalibrationArtifact.model_validate(payload)
    if method == "identity":
        return IdentityCalibrationArtifact.model_validate(payload)
    raise ValueError("calibration method is not supported")


def load_calibration(path: Path, expected: CalibrationContext) -> AnyCalibrationArtifact:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise SaracuraError(
            ErrorCode.CALIBRATION_MISSING,
            "Calibration artifact was not found.",
            "/calibration",
        ) from error

    try:
        artifact = _parse_artifact(raw)
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
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
    artifact: AnyCalibrationArtifact,
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


def create_identity_calibration(
    destination: Path,
    context: CalibrationContext,
    created_at: str,
) -> IdentityCalibrationArtifact:
    """Create the immutable no-fit identity artifact at a new destination."""

    artifact = IdentityCalibrationArtifact(
        **context.model_dump(mode="python"),
        schema_version=1,
        calibration_id=identity_calibration_id(context),
        status="fixture_only",
        method="identity",
        parameters=IdentityParameters(temperature=1.0),
        fit_sample_count=0,
        evaluation_sample_count=0,
        fit_independent_state_count=0,
        evaluation_independent_state_count=0,
        minimum_independent_state_count=0,
        metrics_before={},
        metrics_after={},
        created_at=created_at,
    )
    write_calibration_atomic(destination, artifact)
    return artifact
