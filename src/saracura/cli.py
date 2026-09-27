"""Local-only v1alpha1 CLI for explicit research-only backends."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Never, cast

from pydantic import JsonValue

from saracura.backends import (
    DeterministicFixtureBackend,
    MiniLMRoutingBackend,
    SaracuraUniversalBackend,
)
from saracura.backends.base import BackendCapabilities
from saracura.backends.saracura_universal import (
    SaracuraBackendError,
    VerifiedSaracuraCapsule,
    load_saracura_candidate,
    validate_saracura_model_identity,
    validate_saracura_request_structure,
    verify_training_capsule,
)
from saracura.calibration import (
    ResearchCalibrationArtifact,
    create_identity_calibration,
    load_calibration_envelope,
    validate_calibration_compatibility,
)
from saracura.contracts import ErrorCode, SaracuraError, parse_request_json
from saracura.runtime import (
    MINILM_ROUTING_WORKFLOW_ID,
    MINILM_ROUTING_WORKFLOW_REVISION,
    PHASE4A_IDENTITY_PROFILE,
    DecisionEngine,
    default_workflows,
)
from saracura.runtime.engine import MAX_STATE_PAYLOAD_BYTES, calibration_context
from saracura.runtime.workflows import (
    SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION,
    UNIVERSAL_CHOICE_WORKFLOW_ID,
    UNIVERSAL_CHOICE_WORKFLOW_REVISION,
)
from saracura.serialization import canonical_json_bytes, serialize_state
from saracura.shadow import (
    ShadowDecisionRecord,
    ShadowItem,
    ShadowPolicy,
    ShadowRunner,
    evaluate_shadow_feedback,
    parse_shadow_decision_json,
    parse_shadow_feedback_json,
    parse_shadow_item_json,
    parse_shadow_policy_json,
)
from saracura.shadow.runner import build_shadow_request
from saracura.verified_bytes import read_public_external_file

if TYPE_CHECKING:
    from saracura.backends.laya import LayaUniversalBackend

MAX_REQUEST_BYTES = 1_000_000
MAX_SHADOW_BATCH_ITEMS = 500
MAX_SHADOW_LINE_BYTES = 4096
MAX_SHADOW_INPUT_BYTES = 2_048_000
MAX_SHADOW_FILE_BYTES = 8 * 1024 * 1024
_LAYA_PREVALIDATION_CAPABILITIES = BackendCapabilities(
    execution_tier="universal",
    decision_types=frozenset({"choice"}),
    max_questions=10,
    max_criteria=20,
    execution_boundary="preconstruction-laya-capability",
    cold_warm_semantics="not-loaded",
    quality_claims=False,
    dynamic_workflows=frozenset(
        {(UNIVERSAL_CHOICE_WORKFLOW_ID, UNIVERSAL_CHOICE_WORKFLOW_REVISION)}
    ),
)
_SARACURA_PREVALIDATION_CAPABILITIES = BackendCapabilities(
    execution_tier="universal",
    decision_types=frozenset({"choice"}),
    max_questions=10,
    max_criteria=8,
    execution_boundary="preconstruction-saracura-universal",
    cold_warm_semantics="not-loaded",
    quality_claims=False,
    dynamic_workflows=frozenset(
        {(UNIVERSAL_CHOICE_WORKFLOW_ID, SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION)}
    ),
)


class _ArgumentFailure(ValueError):
    """A parser failure rendered through the normal JSON error envelope."""


def _load_laya_backend() -> type[LayaUniversalBackend]:
    """Load the opt-in Laya implementation only for a Laya command."""

    from saracura.backends.laya import LayaUniversalBackend

    return LayaUniversalBackend


def load_laya_candidate() -> Any:
    """Load Laya metadata only when the Phase 4D command is selected."""

    from saracura.laya_snapshot import load_laya_candidate as load_candidate

    return load_candidate()


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
        "--backend",
        choices=("fixture", "minilm-routing", "laya-universal", "saracura-universal"),
        default="fixture",
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
    describe.add_argument(
        "--backend",
        choices=("minilm-routing", "laya-universal", "saracura-universal"),
        required=True,
    )
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

    shadow_decide = commands.add_parser(
        "shadow-decide", help="run a bounded local shadow batch from UTF-8 JSONL stdin"
    )
    shadow_decide.add_argument("--policy", type=Path, required=True)
    shadow_decide.add_argument("--encoder-snapshot", type=Path, required=True)
    shadow_decide.add_argument("--training-capsule", type=Path, required=True)
    shadow_decide.add_argument("--device", choices=("mps", "cpu"), required=True)

    shadow_evaluate = commands.add_parser(
        "shadow-evaluate", help="evaluate local content-free shadow decisions and feedback"
    )
    shadow_evaluate.add_argument("--policy", type=Path, required=True)
    shadow_evaluate.add_argument("--decisions", type=Path, required=True)
    shadow_evaluate.add_argument("--feedback", type=Path, required=True)
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
        for name in (
            "encoder_snapshot",
            "training_manifest",
            "checkpoint",
            "training_capsule",
        )
    ):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Laya universal commands reject calibration and MiniLM/Saracura arguments.",
            "/backend",
        )
    return _load_laya_backend()(model_snapshot=values.model_snapshot, device=values.device)


def _saracura_artifact_paths(
    values: argparse.Namespace,
) -> tuple[Path, Path, Literal["cpu", "mps"]]:
    if any(
        getattr(values, name, None) is None
        for name in ("encoder_snapshot", "training_capsule", "device")
    ):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            (
                "Saracura universal commands require"
                " explicit encoder-snapshot, training-capsule, and device."
            ),
            "/",
        )
    if getattr(values, "calibration", None) is not None or any(
        getattr(values, name, None) is not None
        for name in ("model_snapshot", "training_manifest", "checkpoint")
    ):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Saracura universal commands reject calibration, Laya, and MiniLM-only arguments.",
            "/backend",
        )
    return values.encoder_snapshot, values.training_capsule, values.device


def _require_saracura_arguments(
    values: argparse.Namespace, *, verified_capsule: VerifiedSaracuraCapsule | None = None
) -> SaracuraUniversalBackend:
    encoder_snapshot, training_capsule, device = _saracura_artifact_paths(values)
    if verified_capsule is not None:
        return SaracuraUniversalBackend._from_verified_capsule(
            encoder_snapshot=encoder_snapshot,
            verified_capsule=verified_capsule,
            device=device,
        )
    return SaracuraUniversalBackend(
        encoder_snapshot=encoder_snapshot,
        training_capsule=training_capsule,
        device=device,
    )


def _backend_for_decide(
    values: argparse.Namespace,
) -> (
    DeterministicFixtureBackend
    | MiniLMRoutingBackend
    | LayaUniversalBackend
    | SaracuraUniversalBackend
):
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
    if values.backend == "saracura-universal":
        return _require_saracura_arguments(values)
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
    request_path: Path,
    *,
    execution_tier: Literal["compiled", "universal"],
    capabilities: BackendCapabilities | None = None,
) -> tuple[Any, bytes]:
    """Validate every request-only gate before optional backend construction."""

    request = parse_request_json(read_public_external_file(request_path, maximum=MAX_REQUEST_BYTES))
    if execution_tier == "compiled" and len(request.questions) != 1:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "The local CLI accepts one calibration artifact and one question.",
            "/questions",
        )
    resolved_capabilities = (
        capabilities
        if capabilities is not None
        else (_LAYA_PREVALIDATION_CAPABILITIES if execution_tier == "universal" else None)
    )
    default_workflows().validate(
        request,
        execution_tier=execution_tier,
        capabilities=resolved_capabilities,
    )
    if (
        request.model in {"latest", "main", "master"}
        and capabilities is not _SARACURA_PREVALIDATION_CAPABILITIES
    ):
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
        if capabilities is _SARACURA_PREVALIDATION_CAPABILITIES:
            # Model identity is intentionally deferred.  The next Saracura
            # step must verify the sealed capsule, then compare its candidate.
            validate_saracura_request_structure(request)
        else:
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
    if values.backend == "saracura-universal":
        saracura_backend = _require_saracura_arguments(values)
        try:
            saracura_backend.prepare()
            candidate = load_saracura_candidate()
            payload = {
                "model": saracura_backend.model.model_dump(mode="json"),
                "execution_tier": "universal",
                "execution_boundary": "explicit-verified-local-saracura-universal",
                "encoder": {
                    "id": candidate.foundation_encoder,
                    "revision": candidate.encoder_revision,
                    "snapshot_complete_sha256": candidate.encoder_snapshot_complete_sha256,
                },
                "tokenizer_revision": "minilm-verified-bytes.v1",
                "renderer_revision": "phase4e-universal-renderer.v1",
                "architecture": candidate.architecture,
                "device": values.device,
                "workflow": {
                    "id": "universal-choice",
                    "revision": "phase4e-saracura-ranker.v1",
                    "locales": ["pt-BR", "en"],
                    "question_type": "choice",
                    "questions": {"minimum": 1, "maximum": 10},
                    "criteria_per_question": {"minimum": 2, "maximum": 8},
                },
                "disposition": candidate.disposition,
                "conformance": "passed",
                "response": {
                    "status": "uncalibrated",
                    "score_semantics": "ranking_weights",
                    "abstained": True,
                    "reason": "uncalibrated_research",
                    "calibration": None,
                    "automation_allowed": False,
                },
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 0
        finally:
            saracura_backend.close()
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
        request, _state_payload = _prevalidate_request(
            values.request,
            execution_tier="universal",
            capabilities=_LAYA_PREVALIDATION_CAPABILITIES,
        )
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


def _run_saracura_decide(values: argparse.Namespace) -> int:
    """Run the explicit Saracura lane, constructing it only after request-only gates."""

    try:
        request, _state_payload = _prevalidate_request(
            values.request,
            execution_tier="universal",
            capabilities=_SARACURA_PREVALIDATION_CAPABILITIES,
        )
        _encoder_snapshot, training_capsule, _device = _saracura_artifact_paths(values)
        verified_capsule = verify_training_capsule(training_capsule)
        validate_saracura_model_identity(request, verified_capsule.candidate)
        backend = _require_saracura_arguments(values, verified_capsule=verified_capsule)
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
    except SaracuraBackendError:
        public = SaracuraError(
            ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
        )
    except OSError:
        public = SaracuraError(ErrorCode.INTERNAL_ERROR, "Local input could not be read.", "/")
    except Exception:
        public = SaracuraError(ErrorCode.INTERNAL_ERROR, "Unexpected internal error.", "/")
    _print_error(public)
    return 2


def _shadow_public_error(
    *, line_index: int | None = None, code: ErrorCode = ErrorCode.REQUEST_INVALID
) -> SaracuraError:
    details: dict[str, JsonValue] = {} if line_index is None else {"line_index": line_index}
    message = (
        "Shadow model execution is unavailable."
        if code == ErrorCode.BACKEND_UNAVAILABLE
        else "Shadow item exceeds model token capacity."
        if code == ErrorCode.CAPACITY_EXCEEDED
        else "Shadow input or execution is invalid."
    )
    return SaracuraError(
        code,
        message,
        ("/model" if code == ErrorCode.BACKEND_UNAVAILABLE else "/state")
        if line_index is not None
        else "/",
        details=details,
    )


def _load_shadow_policy(path: Path) -> ShadowPolicy:
    failed = False
    policy: ShadowPolicy | None = None
    try:
        raw = read_public_external_file(path, maximum=MAX_SHADOW_FILE_BYTES)
        policy = parse_shadow_policy_json(raw.decode("utf-8", errors="strict"))
    except BaseException:
        failed = True
    if failed or policy is None:
        raise _shadow_public_error()
    return policy


def _read_shadow_items(stream: Any, policy: ShadowPolicy) -> list[ShadowItem]:
    items: list[ShadowItem] = []
    references: set[str] = set()
    total_bytes = 0
    line_index = 0
    while True:
        failed = False
        raw_line: bytes | None = None
        try:
            raw_line = stream.readline(MAX_SHADOW_LINE_BYTES + 1)
        except BaseException:
            failed = True
        if failed or not isinstance(raw_line, bytes):
            raise _shadow_public_error(line_index=line_index)
        if raw_line == b"":
            break
        current_index = line_index
        line_index += 1
        total_bytes += len(raw_line)
        if (
            len(raw_line) > MAX_SHADOW_LINE_BYTES
            or total_bytes > MAX_SHADOW_INPUT_BYTES
            or current_index >= MAX_SHADOW_BATCH_ITEMS
        ):
            raise _shadow_public_error(line_index=current_index)
        content = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
        if content.endswith(b"\r"):
            content = content[:-1]
        failed = False
        item: ShadowItem | None = None
        try:
            if not content:
                raise ValueError("empty JSONL line")
            decoded = content.decode("utf-8", errors="strict")
            item = parse_shadow_item_json(decoded)
        except BaseException:
            failed = True
        if failed or item is None or item.item_ref in references:
            raise _shadow_public_error(line_index=current_index)
        failed = False
        try:
            build_shadow_request(policy, item)
        except BaseException:
            failed = True
        if failed:
            raise _shadow_public_error(line_index=current_index)
        references.add(item.item_ref)
        items.append(item)
    if not items:
        raise _shadow_public_error(line_index=0)
    return items


def _write_stdout_bytes(payload: bytes) -> None:
    binary = getattr(sys.stdout, "buffer", None)
    if binary is not None:
        binary.write(payload)
    else:
        sys.stdout.write(payload.decode("utf-8", errors="strict"))


def _run_shadow_decide(values: argparse.Namespace) -> int:
    try:
        policy = _load_shadow_policy(values.policy)
        input_stream = getattr(sys.stdin, "buffer", None)
        if input_stream is None:
            raise _shadow_public_error()
        items = _read_shadow_items(input_stream, policy)

        # Capsule provenance and model binding are checked before the factory can
        # construct a backend or trigger optional model loading.
        candidate = load_saracura_candidate()
        verified_capsule = verify_training_capsule(values.training_capsule, candidate)
        if policy.model != verified_capsule.candidate.model_revision:
            raise _shadow_public_error()

        backend_values = argparse.Namespace(
            encoder_snapshot=values.encoder_snapshot,
            training_capsule=values.training_capsule,
            device=values.device,
            calibration=None,
            model_snapshot=None,
            training_manifest=None,
            checkpoint=None,
        )
        records: list[ShadowDecisionRecord] = []
        failed = False
        failed_index = 0
        failure_code = ErrorCode.BACKEND_UNAVAILABLE
        try:
            with ShadowRunner(
                policy,
                lambda: _require_saracura_arguments(
                    backend_values, verified_capsule=verified_capsule
                ),
            ) as runner:
                for index, item in enumerate(items):
                    record: ShadowDecisionRecord | None = None
                    try:
                        record = runner.decide(item)
                    except SaracuraError as error:
                        failed = True
                        failed_index = index
                        failure_code = error.payload.code
                    except BaseException:
                        failed = True
                        failed_index = index
                        failure_code = ErrorCode.BACKEND_UNAVAILABLE
                    if failed or record is None:
                        failed = True
                        failed_index = index
                        break
                    records.append(record)
        except SaracuraError as error:
            failed = True
            failure_code = error.payload.code
        except BaseException:
            failed = True
            failure_code = ErrorCode.BACKEND_UNAVAILABLE
        if failed:
            raise _shadow_public_error(line_index=failed_index, code=failure_code)

        output = b"".join(
            canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
        )
    except SaracuraError as error:
        public = error
    except BaseException:
        public = _shadow_public_error()
    else:
        _write_stdout_bytes(output)
        return 0
    _print_error(public)
    return 2


def _read_shadow_records(path: Path, parser: Callable[[str], Any]) -> list[Any]:
    failed = False
    records: list[Any] = []
    try:
        raw = read_public_external_file(path, maximum=MAX_SHADOW_FILE_BYTES)
        for line in raw.splitlines():
            if not line:
                raise ValueError("empty JSONL line")
            records.append(parser(line.decode("utf-8", errors="strict")))
    except BaseException:
        failed = True
    if failed:
        raise _shadow_public_error()
    return records


def _run_shadow_evaluate(values: argparse.Namespace) -> int:
    try:
        policy = _load_shadow_policy(values.policy)
        decisions = _read_shadow_records(values.decisions, parse_shadow_decision_json)
        feedback = _read_shadow_records(values.feedback, parse_shadow_feedback_json)
        summary = evaluate_shadow_feedback(policy, decisions, feedback)
        output = canonical_json_bytes(summary.model_dump(mode="json")) + b"\n"
    except SaracuraError as error:
        public = error
    except BaseException:
        public = _shadow_public_error()
    else:
        _write_stdout_bytes(output)
        return 0
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
            if values.backend == "saracura-universal":
                return _run_saracura_decide(values)
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
        if values.command == "shadow-decide":
            return _run_shadow_decide(values)
        if values.command == "shadow-evaluate":
            return _run_shadow_evaluate(values)
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
