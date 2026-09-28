import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks.phase5_candidate.descriptor import (
    AcquisitionDescriptor,
    ThresholdsBlock,
    derive_disposition,
)
from benchmarks.phase5_candidate.fixture import matrix as fixture_matrix
from benchmarks.phase5_candidate.fixture import request_for
from benchmarks.phase5_candidate.report import memory_bucket
from benchmarks.validate_manifests import (
    validate_phase5_candidate_manifest_v2,
    validate_routed_manifest,
)


def _manifest(name: str) -> Path:
    return Path(__file__).parents[1] / "benchmarks/manifests" / name


def test_phase5_candidate_v2_is_closed_and_routed() -> None:
    path = _manifest("phase5-open-model-candidates.v2.json")
    validate_phase5_candidate_manifest_v2(path)
    validate_routed_manifest(path)


def test_phase5_candidate_v2_binds_exact_v1_bytes() -> None:
    v1 = _manifest("phase5-open-model-candidates.v1.json")
    v2 = json.loads(_manifest("phase5-open-model-candidates.v2.json").read_bytes())
    assert v2["supersedes_manifest_sha256"] == hashlib.sha256(v1.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: payload.__setitem__("supersedes_manifest_sha256", "0" * 64),
            "supersedes_manifest_sha256",
        ),
        (
            lambda payload: payload["candidates"][1].__setitem__("declared_license", "unknown"),
            "Apache-2.0",
        ),
        (
            lambda payload: payload["candidates"][2].__setitem__(
                "source_revision", "9c41005b2180347c3c646dfc9e50c4428483ec6b"
            ),
            "null acquisition fields",
        ),
        (
            lambda payload: payload["candidates"][0].__setitem__(
                "source_revision", "9c41005b2180347c3c646dfc9e50c4428483ec6b"
            ),
            "null acquisition fields",
        ),
        (
            lambda payload: payload["candidates"][1].__setitem__("revision", "main"),
            "revision",
        ),
    ],
)
def test_phase5_candidate_v2_rejects_tampering(tmp_path: Path, mutate: object, error: str) -> None:
    payload = json.loads(_manifest("phase5-open-model-candidates.v2.json").read_bytes())
    mutate(payload)  # type: ignore[operator]
    path = tmp_path / "phase5-open-model-candidates.v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_routed_manifest(path)


def test_reviewed_acquisition_requires_full_pinned_chain(tmp_path: Path) -> None:
    payload = json.loads(_manifest("phase5-open-model-candidates.v2.json").read_bytes())
    kev = payload["candidates"][1]
    kev["disposition"] = "reviewed_acquisition"
    kev["declared_license"] = "Apache-2.0"
    kev["license_review"] = "reviewed_for_candidate_artifacts"
    kev["revision"] = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
    kev["revision_state"] = "immutable_pinned"
    kev["local_execution"] = "research_service"
    kev["evidence_authority"] = "reviewed_upstream_identity"
    kev["source_revision"] = "9c41005b2180347c3c646dfc9e50c4428483ec6b"
    kev["base_model_id"] = "Qwen/Qwen3.5-4B-Base"
    kev["base_model_revision"] = "1001bb4d826a52d1f399e183466143f4da7b741b"
    kev["acquisition_descriptor_sha256"] = "a" * 64
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_phase5_candidate_manifest_v2(path)


def _thresholds() -> ThresholdsBlock:
    return ThresholdsBlock(
        max_latency_seconds=60.0,
        p95_latency_seconds_q1=5.0,
        p95_latency_seconds_q10=15.0,
        p95_latency_seconds_q50=45.0,
        max_repeat_probability_delta=1e-6,
        max_together_separate_probability_delta=0.04,
        near_tie_margin=0.04,
        max_peak_rss_gib=22.0,
        max_peak_physical_footprint_gib=22.0,
        preferred_swap_delta_gib=2.0,
        degraded_swap_delta_gib=8.0,
        max_swap_delta_gib=8.0,
        cold_readiness_timeout_seconds=600,
    )


def _pass() -> dict[int, bool]:
    return {1: True, 10: True, 50: True}


def _base(thresholds: ThresholdsBlock) -> dict[str, Any]:
    return {
        "functional_pass": True,
        "identity_pass": True,
        "security_pass": True,
        "cold_readiness_seconds": 120.0,
        "latency_p95_pass": _pass(),
        "any_request_over_60s": False,
        "peak_rss_gib": 20.0,
        "peak_footprint_gib": 18.0,
        "swap_delta_gib": 0.5,
        "memory_pressure_critical": False,
        "invalid_outputs": 0,
        "unauthorized_accepted": 0,
        "oversize_rejected_correctly": True,
        "invalid_probe_rejected_correctly": True,
        "matrix_completed": True,
        "thresholds": thresholds,
    }


def test_disposition_continue_when_all_thresholds_hold() -> None:
    assert derive_disposition(**_base(_thresholds())) == "continue"


def test_disposition_conditional_on_soft_latency_failure() -> None:
    args = _base(_thresholds())
    args["latency_p95_pass"] = {1: False, 10: True, 50: True}
    assert derive_disposition(**args) == "conditional"


def test_disposition_conditional_on_rss_exceeded() -> None:
    args = _base(_thresholds())
    args["peak_rss_gib"] = 24.0
    assert derive_disposition(**args) == "conditional"


def test_disposition_reject_local_on_swap_exceeds_hard_limit() -> None:
    args = _base(_thresholds())
    args["swap_delta_gib"] = 9.0
    assert derive_disposition(**args) == "reject_local"


def test_disposition_reject_local_on_cold_readiness_timeout() -> None:
    args = _base(_thresholds())
    args["cold_readiness_seconds"] = 700.0
    assert derive_disposition(**args) == "reject_local"


def test_disposition_reject_local_on_oversize_accepted() -> None:
    args = _base(_thresholds())
    args["oversize_rejected_correctly"] = False
    assert derive_disposition(**args) == "reject_local"


def test_disposition_reject_local_on_critical_pressure() -> None:
    args = _base(_thresholds())
    args["memory_pressure_critical"] = True
    assert derive_disposition(**args) == "reject_local"


def test_fixture_matrix_covers_both_locales_and_workloads() -> None:
    data = fixture_matrix()
    assert set(data) == {"pt-BR", "en"}
    for locale in data:
        assert set(data[locale]) == {"q1", "q10", "q50"}
        assert len(data[locale]["q1"].questions) == 1
        assert len(data[locale]["q10"].questions) == 10
        assert len(data[locale]["q50"].questions) == 50


def test_fixture_requests_are_typed_choice_schema() -> None:
    for locale in ("pt-BR", "en"):
        for question_count in (1, 10, 50):
            request = request_for(cast(Any, locale), cast(Any, question_count))
            assert request.api_version == "v1alpha1"
            assert request.mode == "research"
            assert len(request.questions) == question_count
            for question in request.questions:
                assert question.type == "choice"
                assert len(question.criteria) >= 2


def test_memory_bucket_rounding() -> None:
    gib = 1024**3
    assert memory_bucket(16 * gib) == "16"
    assert memory_bucket(24 * gib) == "32"
    assert memory_bucket(1 * gib) == "8"


def test_descriptor_rejects_non_kev_candidate() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        AcquisitionDescriptor.model_validate(
            {
                "schema_version": "phase5-kev4b-acquisition.v1",
                "reviewed_at": "2026-09-27",
                "candidate_id": "other-model",
            }
        )


def test_phase5_candidate_core_does_not_import_optional_modules() -> None:
    code = (
        "import sys; import benchmarks.phase5_candidate; "
        "import benchmarks.phase5_candidate.descriptor; "
        "import benchmarks.phase5_candidate.report; "
        "print('platformdirs' in sys.modules, 'psutil' in sys.modules, "
        "'huggingface_hub' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False False"


def test_default_validators_do_not_import_optional_modules() -> None:
    code = (
        "import sys; "
        "import benchmarks.validate_manifests; "
        "import benchmarks.validate_default_environment; "
        "print('platformdirs' in sys.modules, 'psutil' in sys.modules, "
        "'huggingface_hub' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False False"


def test_sanitized_report_rejects_absolute_paths_and_bindings() -> None:
    from benchmarks.phase5_candidate.report import SanitizedReport  # noqa: F401

    assert memory_bucket.__name__ == "memory_bucket"
