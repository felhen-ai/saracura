"""Fail-closed calibration loading and durable atomic writes."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unicodedata
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from saracura.calibration.models import (
    AnyCalibrationArtifact,
    CalibrationArtifact,
    CalibrationContext,
    IdentityCalibrationArtifact,
    IdentityParameters,
    ResearchCalibrationArtifact,
    candidate_from_research_artifact,
    identity_calibration_id,
    research_calibration_id,
)
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.research_trust import ResearchTrustError, canonical, verify_release_receipt
from saracura.serialization import canonical_json_bytes
from saracura.verified_bytes import VerifiedBytesError, read_calibration_external_file


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


def validate_calibration_compatibility(
    artifact: AnyCalibrationArtifact, expected: CalibrationContext
) -> None:
    """Compare a previously single-read artifact with an expected context."""

    _validate_compatibility(artifact, expected)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate calibration JSON key")
        result[key] = value
    return result


def _reject_nonfinite_json(_value: str) -> None:
    raise ValueError("non-finite calibration JSON")


_RESEARCH_ARTIFACT_WIRE_SCHEMA = ResearchCalibrationArtifact.model_json_schema()


def _validate_research_wire_types(value: object, schema: dict[str, Any] | None = None) -> None:
    """Reject every v2 wire-type coercion before Pydantic can normalize it.

    The artifact model intentionally keeps Python construction ergonomic (for
    example its fixed bounds are an immutable tuple), while JSON necessarily
    represents those bounds as an array.  This schema-derived walker therefore
    validates the parsed JSON's exact types recursively, and only then lets the
    model perform its safe collection representation conversion.  It tracks the
    full closed Pydantic JSON schema rather than maintaining a hand-picked list
    of security-sensitive fields.
    """

    root = _RESEARCH_ARTIFACT_WIRE_SCHEMA
    active = root if schema is None else schema
    reference = active.get("$ref")
    if isinstance(reference, str):
        prefix = "#/$defs/"
        if not reference.startswith(prefix):
            raise ValueError("schema-v2 calibration wire schema")
        definition = root.get("$defs", {}).get(reference.removeprefix(prefix))
        if not isinstance(definition, dict):
            raise ValueError("schema-v2 calibration wire schema")
        _validate_research_wire_types(value, definition)
        return
    alternatives = active.get("anyOf")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            try:
                _validate_research_wire_types(value, alternative)
            except ValueError:
                continue
            return
        raise ValueError("schema-v2 calibration wire type")
    for component in active.get("allOf", []):
        if not isinstance(component, dict):
            raise ValueError("schema-v2 calibration wire schema")
        _validate_research_wire_types(value, component)

    kind = active.get("type")
    valid = {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in {int, float} and math.isfinite(float(cast(float | int, value))),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if kind is not None and (not isinstance(kind, str) or not valid.get(kind, False)):
        raise ValueError("schema-v2 calibration wire type")

    if kind == "object":
        assert type(value) is dict
        properties = active.get("properties", {})
        required = active.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("schema-v2 calibration wire schema")
        if any(type(name) is not str or name not in value for name in required):
            raise ValueError("schema-v2 calibration wire shape")
        additional = active.get("additionalProperties", True)
        for name, child in value.items():
            property_schema = properties.get(name)
            if property_schema is None:
                if additional is False:
                    raise ValueError("schema-v2 calibration wire shape")
                property_schema = additional
            if isinstance(property_schema, dict):
                _validate_research_wire_types(child, property_schema)
    elif kind == "array":
        assert type(value) is list
        prefix_items = active.get("prefixItems")
        if isinstance(prefix_items, list):
            for index, child_schema in enumerate(prefix_items):
                if index >= len(value) or not isinstance(child_schema, dict):
                    raise ValueError("schema-v2 calibration wire shape")
                _validate_research_wire_types(value[index], child_schema)
        items = active.get("items")
        if isinstance(items, dict):
            for child in value:
                _validate_research_wire_types(child, items)


def _reject_non_nfc(value: object) -> None:
    """Reject raw schema-v2 Unicode before any signed evidence is rebuilt."""

    if isinstance(value, str):
        if value != unicodedata.normalize("NFC", value):
            raise ValueError("schema-v2 calibration contains non-NFC text")
        return
    if isinstance(value, list):
        for child in value:
            _reject_non_nfc(child)
        return
    if isinstance(value, dict):
        normalized_keys: set[str] = set()
        for key, child in value.items():
            if key != unicodedata.normalize("NFC", key):
                raise ValueError("schema-v2 calibration contains a non-NFC key")
            if key in normalized_keys:
                raise ValueError("schema-v2 calibration keys collide after NFC")
            normalized_keys.add(key)
            _reject_non_nfc(child)


def _parse_artifact(raw: bytes) -> AnyCalibrationArtifact:
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(payload, dict):
        raise ValueError("calibration root must be an object")
    # Schema-v2 is deliberately canonical before validation: it is signed
    # external evidence, unlike the historic public fixture examples.
    if payload.get("schema_version") == 2:
        _reject_non_nfc(payload)
        _validate_research_wire_types(payload)
        if canonical(payload) + b"\n" != raw:
            raise ValueError("schema-v2 calibration is not canonical")
        artifact = ResearchCalibrationArtifact.model_validate(payload)
        # Do not allow a Python-side representation conversion to turn a wire
        # value that was not actually signed into the reconstructed evidence.
        if canonical(artifact.model_dump(mode="json")) + b"\n" != raw:
            raise ValueError("schema-v2 calibration wire representation")
        return artifact
    method = payload.get("method")
    if method == "temperature-scaling":
        return CalibrationArtifact.model_validate(payload)
    if method == "identity":
        return IdentityCalibrationArtifact.model_validate(payload)
    raise ValueError("calibration method is not supported")


def load_calibration_envelope(path: Path) -> AnyCalibrationArtifact:
    """Read and validate one calibration envelope without a context comparison.

    It is the sole public envelope parser used by the CLI.  In particular, a
    malformed v2 envelope cannot be mistaken for legacy v1 after an unsafe
    preliminary probe or a replacement between reads.
    """

    try:
        raw, private_leaf, _private_parent = read_calibration_external_file(
            path, maximum=1_024 * 1_024
        )
        artifact = _parse_artifact(raw)
        if isinstance(artifact, ResearchCalibrationArtifact):
            if not private_leaf:
                raise ValueError("schema-v2 calibration must be private")
            _verify_research_artifact(artifact)
    except (
        ValidationError,
        ValueError,
        json.JSONDecodeError,
        ResearchTrustError,
        VerifiedBytesError,
    ) as error:
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
    return artifact


def load_calibration(path: Path, expected: CalibrationContext) -> AnyCalibrationArtifact:
    artifact = load_calibration_envelope(path)
    _validate_compatibility(artifact, expected)
    return artifact


def _verify_research_artifact(artifact: ResearchCalibrationArtifact) -> None:
    """Verify receipt before using a schema-v2 compatibility claim."""

    expected_id = research_calibration_id(
        artifact.context(),
        artifact.method,
        artifact.parameters,
        artifact.fit_sha256,
        artifact.blind_view_sha256,
        artifact.exposure_run_sha256,
        artifact.blind_report_sha256,
        artifact.release_receipt_sha256,
    )
    if artifact.calibration_id != expected_id:
        raise ValueError("research calibration id")
    receipt_raw = canonical(artifact.release_receipt) + b"\n"
    if sha256(receipt_raw).hexdigest() != artifact.release_receipt_sha256:
        raise ValueError("research receipt digest")
    # Rebuild the exact candidate, rather than treating the candidate digest as
    # an opaque string.  Candidate validation binds every runtime field that is
    # copied from the candidate into the final artifact.
    candidate = candidate_from_research_artifact(artifact)
    if candidate.candidate_sha256 != artifact.candidate_sha256:
        raise ValueError("research candidate digest")
    receipt = verify_release_receipt(receipt_raw)
    expected = {
        "protocol": None,
        "policy_registry": None,
        "states": None,
        "split_plan": None,
        "contributors": None,
        "annotations": None,
        "adjudications": None,
        "controls": None,
        "takedown_ledger": None,
        "packet": artifact.split_manifest_sha256,
        "training_manifest": None,
        "checkpoint": artifact.checkpoint_sha256,
        "fit": artifact.fit_sha256,
        "blind_view": artifact.blind_view_sha256,
        "exposure_run": artifact.exposure_run_sha256,
        "blind_report": artifact.blind_report_sha256,
        "calibration_candidate": artifact.candidate_sha256,
    }
    # The final artifact intentionally does not duplicate private packet
    # lineage.  Its receipt remains closed and binds all fields above; the
    # values available in the artifact must nevertheless match exactly.
    evidence = receipt.get("evidence")
    if receipt.get("scope") != "research_calibration" or not isinstance(evidence, dict):
        raise ValueError("research receipt scope")
    if set(evidence) != set(expected) or any(
        expected[name] is not None and evidence[name] != expected[name] for name in expected
    ):
        raise ValueError("research receipt evidence binding")


def load_research_calibration(path: Path) -> ResearchCalibrationArtifact:
    """Securely parse a signed v2 envelope before deriving its data profile."""

    try:
        artifact = load_calibration_envelope(path)
        if not isinstance(artifact, ResearchCalibrationArtifact):
            raise ValueError("schema-v2 research calibration required")
        return artifact
    except (OSError, ValueError, ValidationError, ResearchTrustError, VerifiedBytesError) as error:
        raise SaracuraError(
            ErrorCode.CALIBRATION_INCOMPATIBLE,
            "Calibration artifact failed schema validation.",
            "/calibration",
        ) from error


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


def write_calibration_atomic_at(
    parent_fd: int,
    name: str,
    artifact: AnyCalibrationArtifact,
) -> None:
    """Publish a final artifact only relative to one already-verified parent FD.

    This is the research-finalization path.  In contrast, the public path
    writer above preserves lightweight legacy fixture behavior.  No operation
    below resolves the parent pathname again, so replacing that pathname cannot
    redirect temporary creation, publication, cleanup, or durability fsync.
    """

    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("calibration output name is invalid")
    payload = canonical_json_bytes(artifact.model_dump(mode="json")) + b"\n"
    temporary_name = f".{name}.{os.urandom(16).hex()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    temporary_fd: int | None = None
    published = False
    failure: BaseException | None = None
    durability_failure: BaseException | None = None
    try:
        temporary_fd = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        os.fchmod(temporary_fd, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError("calibration temporary write")
            view = view[written:]
        os.fsync(temporary_fd)
        descriptor_to_close = temporary_fd
        temporary_fd = None
        os.close(descriptor_to_close)
        # Hard-link creation is atomic no-clobber publication.  Both names are
        # bound to the retained directory descriptor, not a mutable pathname.
        os.link(
            temporary_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except BaseException as exc:
            failure = exc
    finally:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except BaseException as exc:
                if failure is None:
                    failure = exc
        # Retry cleanup even after publication.  Durability is always attempted
        # for a retained destination; a failed parent fsync is never reported
        # as durable and takes precedence when no earlier operation failed.
        if published:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if failure is None:
                    failure = exc
            try:
                os.fsync(parent_fd)
            except BaseException as exc:
                durability_failure = exc
        else:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_fd)
    if durability_failure is not None:
        if failure is not None:
            raise durability_failure from failure
        raise durability_failure
    if failure is not None:
        raise failure


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
