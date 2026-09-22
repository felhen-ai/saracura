"""Adversarial coverage for the frozen Phase 4B.3 numerical boundary."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from benchmarks.first_party_packet import LABELS
from benchmarks.human_calibration import (
    HumanCalibrationError,
    blind_report,
    calibration_candidate,
    fit_artifact,
    fit_temperature,
    metrics,
    probabilities,
)
from benchmarks.human_research import (
    HumanResearchError,
    begin_blind,
    blind_custody_key,
    blind_exposure_state,
    finalize_calibration,
    fit_temperature_stage,
    resume_blind,
    validate_blind_exposure_state,
)
from saracura.calibration.models import (
    CalibrationContext,
    ResearchCalibrationArtifact,
    ResearchCalibrationCandidate,
)
from saracura.research_trust import canonical
from tests.support_human_research import inject_test_trust, signed_receipt, write_private


def _context() -> CalibrationContext:
    return CalibrationContext(
        model_id="saracura-minilm-routing",
        model_revision="minilm-routing-v1.test",
        checkpoint_sha256="a" * 64,
        architecture_config_sha256="b" * 64,
        serializer_version="nfc-jcs-length-prefixed-v1",
        tokenizer_revision="minilm-tokenizer-test",
        truncation_policy_id="padding-truncation-max128-v1",
        precision="encoder-fp32-mps-pool-fp32-mps-transfer-head-fp32-cpu-v1",
        quantization="none",
        output_transform="linear-logits-softmax-v1",
        workflow_id="support-routing",
        workflow_revision="phase4a-local-minilm-routing.v1",
        question_id="department",
        dataset_id="support-routing-human-ptbr",
        dataset_revision="human-ptbr-v1.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        split_manifest_sha256="c" * 64,
        locale="pt-BR",
        domain="support",
        head="choice",
        cardinality_bucket="2-5",
        risk_policy=None,
    )


def _balanced_rows() -> tuple[list[list[float]], list[int]]:
    logits: list[list[float]] = []
    labels: list[int] = []
    for label in range(5):
        for _ in range(20):
            logits.append([3.0 if index == label else 0.0 for index in range(5)])
            labels.append(label)
    return logits, labels


def _assert_json_numbers_close(actual: object, expected: object) -> None:
    """Keep frozen metric structure exact while comparing binary64 leaves by tolerance."""

    if isinstance(expected, float):
        assert isinstance(actual, (float, int))
        assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_json_numbers_close(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for received, value in zip(actual, expected, strict=True):
            _assert_json_numbers_close(received, value)
    else:
        assert actual == expected


def test_temperature_fit_is_bounded_and_never_worse_than_identity() -> None:
    logits = [[4.0, 0.0, 0.0, 0.0, 0.0]] * 20
    labels = [0] * 20
    temperature = fit_temperature(logits, labels)
    assert 0.5 <= temperature <= 5.0
    assert metrics(logits, labels, temperature)["nll"] <= metrics(logits, labels, 1.0)["nll"]


def test_checkout_golden_vectors_cover_fixed_optimizer_branches() -> None:
    resource = Path("benchmarks/fixtures/human-calibration-golden.v1.json").read_bytes()
    value = json.loads(resource)
    assert canonical(value) + b"\n" == resource
    assert value["schema_version"] == "human-calibration-golden.v1"
    assert [item["name"] for item in value["vectors"]] == [
        "constant-branch",
        "lower-bound-branch",
        "interior-branch",
        "upper-bound-branch",
        "exact-tie-branch",
    ]
    for item in value["vectors"]:
        temperature = fit_temperature(item["logits"], item["labels"])
        assert temperature == pytest.approx(item["expected_temperature"], rel=1e-12, abs=1e-12)
        assert probabilities(item["logits"][0], temperature) == pytest.approx(
            item["expected_probabilities"], rel=1e-12, abs=1e-12
        )
        _assert_json_numbers_close(
            metrics(item["logits"], item["labels"], temperature), item["expected_metrics"]
        )


def test_metrics_uses_lowest_index_argmax_and_rejects_nonfinite_inputs() -> None:
    value = metrics([[0.0] * 5], [0], 1.0)
    assert value["confusion_matrix"][0][0] == 1
    assert value["max_confidence"] == pytest.approx(0.2)
    with pytest.raises(HumanCalibrationError):
        probabilities([math.nan, 0.0, 0.0, 0.0, 0.0], 1.0)


def test_ece_places_neighbors_of_equal_width_boundaries_in_separate_bins() -> None:
    """The lower side of 4/15 is bin 3; its upper neighbor is bin 4."""

    def logits_for_confidence(confidence: float) -> list[float]:
        return [math.log(confidence), *[math.log((1.0 - confidence) / 4.0)] * 4]

    lower = logits_for_confidence(math.nextafter(4.0 / 15.0, 0.0))
    upper = logits_for_confidence(math.nextafter(4.0 / 15.0, 1.0))
    lower_confidence = probabilities(lower, 1.0)[0]
    upper_confidence = probabilities(upper, 1.0)[0]
    assert min(14, math.floor(lower_confidence * 15)) == 3
    assert min(14, math.floor(upper_confidence * 15)) == 4

    # Lower is correct and upper is intentionally wrong.  If an implementation
    # merged adjacent bins, this weighted ECE would be materially different.
    value = metrics([lower, upper], [0, 1], 1.0)
    expected = (abs(1.0 - lower_confidence) + abs(upper_confidence)) / 2.0
    assert value["ece_15"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fit_minimums_and_candidate_metrics_are_fail_closed() -> None:
    logits, labels = _balanced_rows()
    fit = fit_artifact(
        _context(),
        packet_manifest_sha256="1" * 64,
        packet_release_receipt_sha256="2" * 64,
        calibration_view_sha256="3" * 64,
        model_manifest_sha256="4" * 64,
        checkpoint_sha256="a" * 64,
        golden_resource_sha256="5" * 64,
        logits=logits,
        labels=labels,
        independent_state_count=100,
        created_at="2026-09-22T00:00:00Z",
    )
    assert fit["fit_sample_count"] == 100
    with pytest.raises(HumanCalibrationError):
        fit_artifact(
            _context(),
            packet_manifest_sha256="1" * 64,
            packet_release_receipt_sha256="2" * 64,
            calibration_view_sha256="3" * 64,
            model_manifest_sha256="4" * 64,
            checkpoint_sha256="a" * 64,
            golden_resource_sha256="5" * 64,
            logits=logits,
            labels=labels,
            independent_state_count=99,
            created_at="2026-09-22T00:00:00Z",
        )
    aggregate = metrics(logits, labels, 1.0)
    candidate = calibration_candidate(
        _context(),
        temperature=1.0,
        fit_sample_count=100,
        evaluation_sample_count=100,
        fit_independent_state_count=100,
        evaluation_independent_state_count=100,
        fit_sha256=fit["fit_sha256"],
        blind_view_sha256="6" * 64,
        exposure_run_sha256="7" * 64,
        blind_report_sha256="8" * 64,
        metrics_before=aggregate,
        metrics_after=aggregate,
        created_at="2026-09-22T00:00:00Z",
    )
    assert (
        ResearchCalibrationCandidate.model_validate(candidate).candidate_sha256
        == candidate["candidate_sha256"]
    )
    broken = {**candidate, "metrics_after": {**aggregate, "accuracy": 0.0}}
    with pytest.raises(ValueError):
        ResearchCalibrationCandidate.model_validate(broken)


def test_blind_exposure_states_are_ordered_immutable_and_digest_bound() -> None:
    packet, descriptor = "a" * 64, "b" * 64
    custody = blind_custody_key(packet, descriptor)
    first = blind_exposure_state(
        state="precommitted",
        custody_key=custody,
        packet_sha256=packet,
        packet_manifest_bytes_sha256="9" * 64,
        packet_release_receipt_sha256="f" * 64,
        blind_capsule_sha256=descriptor,
        training_manifest_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
        fit_sha256="e" * 64,
        previous_state_sha256=None,
        created_at="2026-09-22T00:00:00Z",
    )
    raw = canonical(first) + b"\n"
    assert validate_blind_exposure_state(raw, expected_state="precommitted") == first
    tampered = dict(first)
    tampered["fit_sha256"] = "f" * 64
    with pytest.raises(HumanResearchError):
        validate_blind_exposure_state(canonical(tampered) + b"\n", expected_state="precommitted")
    with pytest.raises(HumanResearchError):
        blind_exposure_state(
            state="running",
            custody_key=custody,
            packet_sha256=packet,
            packet_manifest_bytes_sha256="9" * 64,
            packet_release_receipt_sha256="f" * 64,
            blind_capsule_sha256=descriptor,
            training_manifest_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
            fit_sha256="e" * 64,
            previous_state_sha256=None,
            created_at="2026-09-22T00:00:00Z",
        )


def test_blind_bytes_are_not_opened_until_durable_precommit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The view loader is our observable blind-byte boundary in this unit test."""

    import benchmarks.human_research as research
    import benchmarks.human_training as training

    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    registry.chmod(0o700)
    packet_raw = b'{"packet":"fixed"}\n'
    receipt_raw = b'{"receipt":"fixed"}\n'
    packet = {
        "packet_sha256": "a" * 64,
        "accepted_counts": {
            "by_split": {"blind_test": 100},
            "by_split_label": {"blind_test": {label: 20 for label in LABELS}},
        },
    }
    training_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": receipt_raw,
        "training-manifest.json": b"training",
        "checkpoint.safetensors": b"checkpoint",
    }
    fit_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": receipt_raw,
        "temperature-fit.json": b"fit",
        "capsule-descriptor.json": b"descriptor",
    }
    fit = {
        "fit_sha256": "b" * 64,
        "packet_manifest_sha256": research.digest(packet_raw),
        "packet_release_receipt_sha256": research.digest(receipt_raw),
        "training_manifest_sha256": research.digest(training_values["training-manifest.json"]),
        "checkpoint_sha256": research.digest(training_values["checkpoint.safetensors"]),
    }
    descriptor_value: dict[str, object] = {
        "schema_version": "human-capsule.v1",
        "capsule_kind": "blind",
        "packet_sha256": "a" * 64,
        "packet_release_receipt_sha256": "c" * 64,
        "files": [
            {"path": name, "bytes": 1, "sha256": "d" * 64}
            for name in ("blind-test.jsonl", "packet-manifest.json", "packet-release-receipt.json")
        ],
        "descriptor_sha256": "",
    }
    descriptor_value["descriptor_sha256"] = research.digest(
        canonical(
            {key: value for key, value in descriptor_value.items() if key != "descriptor_sha256"}
        )
    )
    descriptor = canonical(descriptor_value) + b"\n"

    class BlindRead(Exception):
        pass

    monkeypatch.setattr(
        training, "_trusted_capsule", lambda *_args, **_kwargs: (training_values, {})
    )
    monkeypatch.setattr(
        training,
        "_descriptor",
        lambda *_args, **_kwargs: {
            "packet_sha256": packet["packet_sha256"],
            "packet_release_receipt_sha256": research.digest(receipt_raw),
        },
    )
    monkeypatch.setattr(training, "_packet_only", lambda *_args, **_kwargs: packet)

    def read_fit(path: Path, _limits: dict[str, int]) -> dict[str, bytes]:
        if path.name != "fit":
            raise AssertionError("unexpected blind read")
        return fit_values

    monkeypatch.setattr(research, "_read_capsule", read_fit)
    monkeypatch.setattr(research, "_temperature_fit", lambda _raw: fit)
    monkeypatch.setattr(
        research,
        "_prepare_blind_inputs",
        lambda *_args: (
            training_values,
            fit,
            object(),
            packet,
            descriptor,
            research.digest(descriptor),
        ),
    )
    monkeypatch.setattr(
        research, "read_verified_external_file", lambda *_args, **_kwargs: descriptor
    )

    def open_blind(*_args: object, **_kwargs: object) -> tuple[dict[str, bytes], dict[str, str]]:
        active = next(registry.glob("*.active"))
        assert (active / "01-precommitted.json").is_file()
        assert (active / "01-precommitted.json").read_bytes().endswith(b"\n")
        raise BlindRead

    monkeypatch.setattr(research, "_trusted_view_capsule", open_blind)
    with pytest.raises(BlindRead):
        begin_blind(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "fit",
            tmp_path / "blind",
            registry,
        )


def test_second_begin_cannot_reach_blind_reader_after_active_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic interleaving must fail before a second blind read."""

    import benchmarks.human_research as research

    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    registry.chmod(0o700)
    packet_raw, receipt_raw, descriptor = b"packet", b"receipt", b"blind-descriptor\n"
    packet = {
        "packet_sha256": "a" * 64,
        "accepted_counts": {
            "by_split": {"blind_test": 100},
            "by_split_label": {"blind_test": {label: 20 for label in LABELS}},
        },
    }
    training_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": receipt_raw,
        "training-manifest.json": b"training",
        "checkpoint.safetensors": b"checkpoint",
    }
    fit = {
        "fit_sha256": "b" * 64,
        "packet_manifest_sha256": research.digest(packet_raw),
        "packet_release_receipt_sha256": research.digest(receipt_raw),
        "training_manifest_sha256": research.digest(b"training"),
        "checkpoint_sha256": research.digest(b"checkpoint"),
    }
    monkeypatch.setattr(
        research,
        "_prepare_blind_inputs",
        lambda *_args: (
            training_values,
            fit,
            object(),
            packet,
            descriptor,
            research.digest(descriptor),
        ),
    )
    blind_reachers = 0

    def interleaved_continue(*_args: object, **_kwargs: object) -> None:
        nonlocal blind_reachers
        blind_reachers += 1
        with pytest.raises(HumanResearchError, match="blind custody"):
            begin_blind(
                tmp_path / "snapshot",
                tmp_path / "training",
                tmp_path / "fit",
                tmp_path / "blind",
                registry,
            )
        raise RuntimeError("stop after claim")

    monkeypatch.setattr(research, "_continue_blind_run", interleaved_continue)
    with pytest.raises(RuntimeError, match="stop after claim"):
        begin_blind(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "fit",
            tmp_path / "blind",
            registry,
        )
    assert blind_reachers == 1


@pytest.mark.parametrize(
    ("packet_value", "receipt_value"),
    [(b"packet-b", b"receipt-a"), (b"packet-a", b"receipt-b")],
)
def test_fit_refuses_nonidentical_packet_or_receipt_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    packet_value: bytes,
    receipt_value: bytes,
) -> None:
    import benchmarks.human_research as research
    import benchmarks.human_training as training

    training_values = {
        "packet-manifest.json": b"packet-a",
        "packet-release-receipt.json": b"receipt-a",
    }
    calibration_values = {
        "packet-manifest.json": packet_value,
        "packet-release-receipt.json": receipt_value,
        "calibration.jsonl": b"blind text must not reach model\n",
    }
    monkeypatch.setattr(
        training, "_trusted_capsule", lambda *_args, **_kwargs: (training_values, {})
    )
    monkeypatch.setattr(
        training, "_packet_only", lambda *_args, **_kwargs: {"packet_sha256": "a" * 64}
    )
    monkeypatch.setattr(
        research,
        "_trusted_view_capsule",
        lambda *_args, **_kwargs: (calibration_values, {"packet_sha256": "b" * 64}),
    )

    class ModelMustNotConstruct:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("mixed capsule reached model construction")

    monkeypatch.setattr("saracura.backends.minilm.MiniLMRoutingBackend", ModelMustNotConstruct)
    with pytest.raises(HumanResearchError, match="fit capsule packet binding"):
        fit_temperature_stage(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "calibration",
            tmp_path,
            "fit",
        )


@pytest.mark.parametrize(
    ("axis", "replacement"),
    [("tokenizer_revision", "wrong-tokenizer"), ("model_revision", "wrong-runtime-revision")],
)
def test_begin_blind_rejects_fit_tokenizer_or_runtime_drift_before_blind_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axis: str, replacement: str
) -> None:
    import benchmarks.human_research as research
    import benchmarks.human_training as training

    packet_raw = b"packet"
    receipt_raw = b"receipt"
    training_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": receipt_raw,
        "training-manifest.json": b"training",
        "checkpoint.safetensors": b"checkpoint",
    }
    calibration_raw = b"calibration"
    packet: dict[str, Any] = {
        "packet_sha256": "a" * 64,
        "files": {
            "calibration": {
                "bytes": len(calibration_raw),
                "sha256": research.digest(calibration_raw),
            },
            "blind_test": {"bytes": 1, "sha256": "d" * 64},
        },
        "accepted_counts": {
            "by_split": {"calibration": 100},
            "by_split_label": {"calibration": {label: 20 for label in LABELS}},
        },
    }

    class Backend:
        model = type(
            "Model",
            (),
            {
                "id": "saracura-minilm-routing",
                "revision": "minilm-routing-v1.bound",
                "checkpoint_sha256": research.digest(training_values["checkpoint.safetensors"]),
            },
        )()
        calibration_metadata = type(
            "Metadata",
            (),
            {
                "architecture_config_sha256": "b" * 64,
                "tokenizer_revision": "bound-tokenizer",
                "truncation_policy_id": "padding-truncation-max128-v1",
                "precision": "encoder-fp32-mps-pool-fp32-mps-transfer-head-fp32-cpu-v1",
                "quantization": "none",
                "output_transform": "linear-logits-softmax-v1",
            },
        )()

    fit = research._blind_runtime_context(Backend(), packet["packet_sha256"])
    fit[axis] = replacement
    fit.update(
        {
            "fit_sha256": "c" * 64,
            "packet_manifest_sha256": research.digest(packet_raw),
            "packet_release_receipt_sha256": research.digest(receipt_raw),
            "training_manifest_sha256": research.digest(training_values["training-manifest.json"]),
            "checkpoint_sha256": research.digest(training_values["checkpoint.safetensors"]),
            "calibration_view_sha256": research.digest(calibration_raw),
            "fit_sample_count": 100,
            "fit_independent_state_count": 100,
            "per_label_counts": {label: 20 for label in LABELS},
            "golden_resource_sha256": research.digest(
                (Path("benchmarks/fixtures") / "human-calibration-golden.v1.json").read_bytes()
            ),
        }
    )
    fit_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": receipt_raw,
        "temperature-fit.json": b"fit",
        "calibration.jsonl": calibration_raw,
        "capsule-descriptor.json": b"descriptor",
    }
    descriptor_value: dict[str, object] = {
        "schema_version": "human-capsule.v1",
        "capsule_kind": "blind",
        "packet_sha256": packet["packet_sha256"],
        "packet_release_receipt_sha256": research.digest(receipt_raw),
        "files": [
            {
                "path": "blind-test.jsonl",
                "bytes": 1,
                "sha256": packet["files"]["blind_test"]["sha256"],
            },
            {
                "path": "packet-manifest.json",
                "bytes": len(packet_raw),
                "sha256": research.digest(packet_raw),
            },
            {
                "path": "packet-release-receipt.json",
                "bytes": len(receipt_raw),
                "sha256": research.digest(receipt_raw),
            },
        ],
        "descriptor_sha256": "",
    }
    descriptor_value["descriptor_sha256"] = research.digest(
        canonical(
            {key: value for key, value in descriptor_value.items() if key != "descriptor_sha256"}
        )
    )
    blind_descriptor = canonical(descriptor_value) + b"\n"
    monkeypatch.setattr(
        training,
        "_trusted_capsule",
        lambda *_args, **_kwargs: (training_values, {}),
    )
    monkeypatch.setattr(
        training,
        "_descriptor",
        lambda *_args, **_kwargs: {
            "packet_sha256": packet["packet_sha256"],
            "packet_release_receipt_sha256": research.digest(receipt_raw),
            "files": [
                {"path": "calibration.jsonl", **packet["files"]["calibration"]},
            ],
        },
    )
    monkeypatch.setattr(training, "_packet_only", lambda *_args, **_kwargs: packet)
    monkeypatch.setattr(research, "_read_capsule", lambda *_args, **_kwargs: fit_values)
    monkeypatch.setattr(research, "_temperature_fit", lambda _raw: fit)
    monkeypatch.setattr(
        "saracura.backends.minilm.MiniLMRoutingBackend",
        lambda **_kwargs: Backend(),
    )
    monkeypatch.setattr(
        research,
        "read_verified_external_file",
        lambda *_args, **_kwargs: blind_descriptor,
    )

    with pytest.raises(HumanResearchError, match="fit runtime context binding"):
        begin_blind(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "fit",
            tmp_path / "blind",
            tmp_path / "registry",
        )


def test_finalization_requires_exact_sealed_chain_and_research_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.human_research as research
    from benchmarks.human_training import _make_descriptor

    logits, labels = _balanced_rows()
    context = _context().model_copy(
        update={"checkpoint_sha256": "c" * 64, "split_manifest_sha256": "2" * 64}
    )
    aggregate = metrics(logits, labels, 1.0)
    custody_key = blind_custody_key("2" * 64, "b" * 64)
    precommit = blind_exposure_state(
        state="precommitted",
        custody_key=custody_key,
        packet_sha256="2" * 64,
        packet_manifest_bytes_sha256="2" * 64,
        packet_release_receipt_sha256=research.digest(b"packet-receipt"),
        blind_capsule_sha256="b" * 64,
        training_manifest_sha256="1" * 64,
        checkpoint_sha256="c" * 64,
        fit_sha256="3" * 64,
        previous_state_sha256=None,
        created_at="2026-09-22T00:00:00Z",
    )
    running = blind_exposure_state(
        state="running",
        custody_key=custody_key,
        packet_sha256="2" * 64,
        packet_manifest_bytes_sha256="2" * 64,
        packet_release_receipt_sha256=research.digest(b"packet-receipt"),
        blind_capsule_sha256="b" * 64,
        training_manifest_sha256="1" * 64,
        checkpoint_sha256="c" * 64,
        fit_sha256="3" * 64,
        previous_state_sha256=precommit["state_sha256"],
        created_at="2026-09-22T00:00:00Z",
    )
    report = blind_report(
        model_manifest_sha256="1" * 64,
        packet_manifest_sha256="2" * 64,
        fit_sha256="3" * 64,
        blind_view_sha256="4" * 64,
        exposure_run_sha256=running["state_sha256"],
        golden_resource_sha256="6" * 64,
        logits=logits,
        labels=labels,
        temperature=1.0,
        created_at="2026-09-22T00:00:00Z",
    )
    candidate = calibration_candidate(
        context,
        temperature=1.0,
        fit_sample_count=100,
        evaluation_sample_count=100,
        fit_independent_state_count=100,
        evaluation_independent_state_count=100,
        fit_sha256="3" * 64,
        blind_view_sha256="4" * 64,
        exposure_run_sha256=running["state_sha256"],
        blind_report_sha256=report["report_sha256"],
        metrics_before=aggregate,
        metrics_after=aggregate,
        created_at="2026-09-22T00:00:00Z",
    )
    sealed = blind_exposure_state(
        state="sealed",
        custody_key=custody_key,
        packet_sha256="2" * 64,
        packet_manifest_bytes_sha256="2" * 64,
        packet_release_receipt_sha256=research.digest(b"packet-receipt"),
        blind_capsule_sha256="b" * 64,
        training_manifest_sha256="1" * 64,
        checkpoint_sha256="c" * 64,
        fit_sha256="3" * 64,
        previous_state_sha256=running["state_sha256"],
        created_at="2026-09-22T00:00:00Z",
        blind_report_sha256=report["report_sha256"],
        calibration_candidate_sha256=candidate["candidate_sha256"],
    )
    values = {
        "01-precommitted.json": canonical(precommit) + b"\n",
        "02-running.json": canonical(running) + b"\n",
        "03-sealed.json": canonical(sealed) + b"\n",
        "blind-report.json": canonical(report) + b"\n",
        "calibration-candidate.json": canonical(candidate) + b"\n",
    }
    receipt: dict[str, Any] = {
        "scope": "research_calibration",
        "evidence": {
            **{
                name: "d" * 64
                for name in (
                    "protocol",
                    "policy_registry",
                    "states",
                    "split_plan",
                    "contributors",
                    "annotations",
                    "adjudications",
                    "controls",
                    "takedown_ledger",
                )
            },
            "packet": precommit["packet_sha256"],
            "training_manifest": precommit["training_manifest_sha256"],
            "checkpoint": precommit["checkpoint_sha256"],
            "fit": candidate["fit_sha256"],
            "blind_view": candidate["blind_view_sha256"],
            "exposure_run": candidate["exposure_run_sha256"],
            "blind_report": candidate["blind_report_sha256"],
            "calibration_candidate": candidate["candidate_sha256"],
        },
    }
    receipt_raw = canonical(receipt) + b"\n"
    descriptor = _make_descriptor(
        "blind-run", values, {"packet_sha256": precommit["packet_sha256"]}, b"packet-receipt"
    )
    run = tmp_path / custody_key
    run.mkdir(mode=0o700)
    run.chmod(0o700)
    for name, raw in {**values, "capsule-descriptor.json": descriptor}.items():
        target = run / name
        target.write_bytes(raw)
        target.chmod(0o600)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output_parent.chmod(0o700)
    output = output_parent / "research-calibration.json"
    monkeypatch.setattr(
        research, "read_verified_external_file", lambda *_args, **_kwargs: receipt_raw
    )
    monkeypatch.setattr(research, "verify_release_receipt", lambda _raw: receipt)

    assert finalize_calibration(run, tmp_path / "receipt.json", output) == output
    assert output.is_file()
    with pytest.raises(FileExistsError):
        finalize_calibration(run, tmp_path / "receipt.json", output)
    # A self-consistent descriptor with an unrelated packet receipt is still
    # rejected: finalization binds it to every durable exposure state.
    descriptor_tamper = _make_descriptor(
        "blind-run", values, {"packet_sha256": precommit["packet_sha256"]}, b"other-receipt"
    )
    descriptor_path = run / "capsule-descriptor.json"
    descriptor_path.write_bytes(descriptor_tamper)
    descriptor_path.chmod(0o600)
    with pytest.raises(HumanResearchError, match="blind candidate binding"):
        finalize_calibration(run, tmp_path / "receipt.json", output_parent / "descriptor.json")
    descriptor_path.write_bytes(descriptor)
    descriptor_path.chmod(0o600)
    receipt["evidence"]["fit"] = "0" * 64
    with pytest.raises(HumanResearchError):
        finalize_calibration(run, tmp_path / "receipt.json", output_parent / "tampered.json")


def _complete_recovery_fixture() -> tuple[dict[str, bytes], Any, str]:
    """Build a descriptor-complete active set from simulated aggregate-only bytes."""

    import benchmarks.human_research as research
    from benchmarks.human_training import _make_descriptor

    logits, labels = _balanced_rows()
    packet_raw, receipt_raw = b"packet-manifest", b"packet-receipt"
    training_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": receipt_raw,
        "training-manifest.json": b"training-manifest",
        "checkpoint.safetensors": b"checkpoint",
    }
    packet: dict[str, Any] = {
        "packet_sha256": "a" * 64,
        "accepted_counts": {
            "by_split": {"blind_test": 100},
            "by_split_label": {"blind_test": {label: 20 for label in LABELS}},
        },
    }
    context = _context().model_copy(
        update={
            "checkpoint_sha256": research.digest(training_values["checkpoint.safetensors"]),
            "split_manifest_sha256": packet["packet_sha256"],
        }
    )
    fit = {
        **context.model_dump(mode="json"),
        "temperature": 1.0,
        "fit_sample_count": 100,
        "fit_independent_state_count": 100,
        "per_label_counts": {label: 20 for label in LABELS},
        "fit_sha256": "b" * 64,
        "golden_resource_sha256": "c" * 64,
    }
    blind_descriptor = b"blind-capsule-governance-only"
    descriptor_digest = research.digest(blind_descriptor)
    key = blind_custody_key(research.digest(packet_raw), descriptor_digest)
    precommit = blind_exposure_state(
        state="precommitted",
        custody_key=key,
        packet_sha256=packet["packet_sha256"],
        packet_manifest_bytes_sha256=research.digest(packet_raw),
        packet_release_receipt_sha256=research.digest(receipt_raw),
        blind_capsule_sha256=descriptor_digest,
        training_manifest_sha256=research.digest(training_values["training-manifest.json"]),
        checkpoint_sha256=research.digest(training_values["checkpoint.safetensors"]),
        fit_sha256=fit["fit_sha256"],
        previous_state_sha256=None,
        created_at="2026-09-22T00:00:00Z",
    )
    running = blind_exposure_state(
        state="running",
        custody_key=key,
        packet_sha256=packet["packet_sha256"],
        packet_manifest_bytes_sha256=research.digest(packet_raw),
        packet_release_receipt_sha256=research.digest(receipt_raw),
        blind_capsule_sha256=descriptor_digest,
        training_manifest_sha256=research.digest(training_values["training-manifest.json"]),
        checkpoint_sha256=research.digest(training_values["checkpoint.safetensors"]),
        fit_sha256=fit["fit_sha256"],
        previous_state_sha256=precommit["state_sha256"],
        created_at="2026-09-22T00:00:00Z",
    )
    report = blind_report(
        model_manifest_sha256=research.digest(training_values["training-manifest.json"]),
        packet_manifest_sha256=research.digest(packet_raw),
        fit_sha256=fit["fit_sha256"],
        blind_view_sha256="d" * 64,
        exposure_run_sha256=running["state_sha256"],
        golden_resource_sha256=fit["golden_resource_sha256"],
        logits=logits,
        labels=labels,
        temperature=1.0,
        created_at="2026-09-22T00:00:00Z",
    )
    candidate = calibration_candidate(
        context,
        temperature=1.0,
        fit_sample_count=100,
        evaluation_sample_count=100,
        fit_independent_state_count=100,
        evaluation_independent_state_count=100,
        fit_sha256=fit["fit_sha256"],
        blind_view_sha256="d" * 64,
        exposure_run_sha256=running["state_sha256"],
        blind_report_sha256=report["report_sha256"],
        metrics_before=report["metrics_before"],
        metrics_after=report["metrics_after"],
        created_at="2026-09-22T00:00:00Z",
    )
    sealed = blind_exposure_state(
        state="sealed",
        custody_key=key,
        packet_sha256=packet["packet_sha256"],
        packet_manifest_bytes_sha256=research.digest(packet_raw),
        packet_release_receipt_sha256=research.digest(receipt_raw),
        blind_capsule_sha256=descriptor_digest,
        training_manifest_sha256=research.digest(training_values["training-manifest.json"]),
        checkpoint_sha256=research.digest(training_values["checkpoint.safetensors"]),
        fit_sha256=fit["fit_sha256"],
        previous_state_sha256=running["state_sha256"],
        created_at="2026-09-22T00:00:00Z",
        blind_report_sha256=report["report_sha256"],
        calibration_candidate_sha256=candidate["candidate_sha256"],
    )
    values = {
        "01-precommitted.json": canonical(precommit) + b"\n",
        "02-running.json": canonical(running) + b"\n",
        "03-sealed.json": canonical(sealed) + b"\n",
        "blind-report.json": canonical(report) + b"\n",
        "calibration-candidate.json": canonical(candidate) + b"\n",
    }
    values["capsule-descriptor.json"] = _make_descriptor("blind-run", values, packet, receipt_raw)
    static = research._BlindStaticInputs(
        training_values=training_values,
        fit_values={},
        fit=fit,
        packet=packet,
        blind_descriptor=blind_descriptor,
        descriptor_digest=descriptor_digest,
        blind_view_sha256="d" * 64,
    )
    return values, static, key


def _rebind_complete_recovery_values(values: dict[str, bytes], static: Any) -> dict[str, bytes]:
    """Recompute every self-digest and the descriptor after an adversarial edit."""

    import benchmarks.human_research as research
    from benchmarks.human_training import _make_descriptor

    result = dict(values)
    report = json.loads(result["blind-report.json"])
    report_unsigned = dict(report)
    report_unsigned.pop("report_sha256", None)
    report["report_sha256"] = research.digest(canonical(report_unsigned))
    candidate = json.loads(result["calibration-candidate.json"])
    candidate["blind_report_sha256"] = report["report_sha256"]
    candidate_unsigned = dict(candidate)
    candidate_unsigned.pop("candidate_sha256", None)
    candidate["candidate_sha256"] = research.digest(canonical(candidate_unsigned))
    running = json.loads(result["02-running.json"])
    previous_sealed = json.loads(result["03-sealed.json"])
    sealed = blind_exposure_state(
        state="sealed",
        custody_key=previous_sealed["custody_key"],
        packet_sha256=previous_sealed["packet_sha256"],
        packet_manifest_bytes_sha256=previous_sealed["packet_manifest_bytes_sha256"],
        packet_release_receipt_sha256=previous_sealed["packet_release_receipt_sha256"],
        blind_capsule_sha256=previous_sealed["blind_capsule_sha256"],
        training_manifest_sha256=previous_sealed["training_manifest_sha256"],
        checkpoint_sha256=previous_sealed["checkpoint_sha256"],
        fit_sha256=previous_sealed["fit_sha256"],
        previous_state_sha256=running["state_sha256"],
        created_at=previous_sealed["created_at"],
        blind_report_sha256=report["report_sha256"],
        calibration_candidate_sha256=candidate["candidate_sha256"],
    )
    result["blind-report.json"] = canonical(report) + b"\n"
    result["calibration-candidate.json"] = canonical(candidate) + b"\n"
    result["03-sealed.json"] = canonical(sealed) + b"\n"
    descriptor_values = {
        name: result[name]
        for name in (
            "01-precommitted.json",
            "02-running.json",
            "03-sealed.json",
            "blind-report.json",
            "calibration-candidate.json",
        )
    }
    result["capsule-descriptor.json"] = _make_descriptor(
        "blind-run",
        descriptor_values,
        static.packet,
        static.training_values["packet-release-receipt.json"],
    )
    return result


def test_complete_recovery_reconciles_every_trusted_binding_without_a_model() -> None:
    """Self-digests and a rebuilt descriptor cannot hide an anchored mismatch."""

    import benchmarks.human_research as research

    values, static, key = _complete_recovery_fixture()
    research._validate_complete_sealed_run(values, static=static, key=key)

    def mutate_temperature(candidate: dict[str, Any], _report: dict[str, Any]) -> None:
        candidate["parameters"]["temperature"] = 1.1

    def mutate_tokenizer(candidate: dict[str, Any], _report: dict[str, Any]) -> None:
        candidate["tokenizer_revision"] = "tampered-tokenizer"

    def mutate_fit_count(candidate: dict[str, Any], _report: dict[str, Any]) -> None:
        candidate["fit_sample_count"] = 101

    def mutate_report_model(_candidate: dict[str, Any], report: dict[str, Any]) -> None:
        report["model_manifest_sha256"] = "e" * 64

    def mutate_candidate_metrics(candidate: dict[str, Any], _report: dict[str, Any]) -> None:
        candidate["metrics_after"]["nll"] += 0.01

    for mutate in (
        mutate_temperature,
        mutate_tokenizer,
        mutate_fit_count,
        mutate_report_model,
        mutate_candidate_metrics,
    ):
        tampered = dict(values)
        candidate = json.loads(tampered["calibration-candidate.json"])
        report = json.loads(tampered["blind-report.json"])
        mutate(candidate, report)
        tampered["calibration-candidate.json"] = canonical(candidate) + b"\n"
        tampered["blind-report.json"] = canonical(report) + b"\n"
        tampered = _rebind_complete_recovery_values(tampered, static)
        with pytest.raises(HumanResearchError, match="blind sealed binding"):
            research._validate_complete_sealed_run(tampered, static=static, key=key)


def test_failed_precommit_staging_leaves_no_keyed_custody_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a durable precommit may become observable at the keyed active name."""

    import benchmarks.human_research as research
    from saracura.verified_bytes import open_verified_directory

    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    registry.chmod(0o700)
    key = "a" * 64
    monkeypatch.setattr(
        research,
        "_write_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated write failure")),
    )
    registry_fd = open_verified_directory(registry)
    try:
        with pytest.raises(HumanResearchError, match="blind custody claim"):
            research._publish_blind_precommit(registry_fd, key, b"precommit")
    finally:
        os.close(registry_fd)
    assert not (registry / key).exists()
    assert not (registry / f"{key}.active").exists()
    assert not list(registry.iterdir())


@pytest.mark.parametrize("failure", ("open", "chmod"))
def test_precommit_mkdir_cleans_hidden_stage_when_open_or_chmod_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A stage created before its FD is returned is still recovered safely."""

    import benchmarks.human_research as research
    from saracura.verified_bytes import open_verified_directory

    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    registry.chmod(0o700)
    key = "b" * 64
    registry_fd = open_verified_directory(registry)
    try:
        if failure == "open":
            original_open = os.open

            def fail_stage_open(
                name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if isinstance(name, str) and name.startswith(f".{key}.precommit-"):
                    raise OSError("simulated stage open failure")
                return original_open(name, flags, mode, dir_fd=dir_fd)

            monkeypatch.setattr(os, "open", fail_stage_open)
        else:
            monkeypatch.setattr(
                os,
                "fchmod",
                lambda _fd, _mode: (_ for _ in ()).throw(OSError("simulated chmod failure")),
            )
        with pytest.raises(HumanResearchError, match="immutable output conflict"):
            research._publish_blind_precommit(registry_fd, key, b"precommit")
    finally:
        os.close(registry_fd)

    assert not (registry / key).exists()
    assert not (registry / f"{key}.active").exists()
    assert not list(registry.iterdir())


def test_ephemeral_signed_fit_blind_finalization_and_verified_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the full sealed sequence with test-only bytes and Ed25519 key."""

    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private_key = crypto.Ed25519PrivateKey.generate()
    inject_test_trust(monkeypatch, private_key, scopes=["research_calibration"])
    import benchmarks.human_research as research
    import benchmarks.human_training as training
    from saracura.calibration import load_calibration

    logits, labels = _balanced_rows()
    context = _context().model_copy(
        update={
            "checkpoint_sha256": research.digest(b"checkpoint"),
            "split_manifest_sha256": "a" * 64,
        }
    )
    packet_raw, packet_receipt, blind_raw = b"packet", b"packet-receipt", b"blind"
    training_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": packet_receipt,
        "training-manifest.json": b"training",
        "checkpoint.safetensors": b"checkpoint",
    }
    golden = Path("benchmarks/fixtures/human-calibration-golden.v1.json").read_bytes()
    fit = fit_artifact(
        context,
        packet_manifest_sha256=research.digest(packet_raw),
        packet_release_receipt_sha256=research.digest(packet_receipt),
        calibration_view_sha256="2" * 64,
        model_manifest_sha256=research.digest(training_values["training-manifest.json"]),
        checkpoint_sha256=research.digest(training_values["checkpoint.safetensors"]),
        golden_resource_sha256=research.digest(golden),
        logits=logits,
        labels=labels,
        independent_state_count=100,
        created_at="2026-09-22T00:00:00Z",
    )
    packet = {
        "packet_sha256": "a" * 64,
        "accepted_counts": {
            "by_split": {"blind_test": 100},
            "by_split_label": {"blind_test": {label: 20 for label in LABELS}},
        },
    }
    blind_descriptor = b'{"blind":"descriptor"}\n'
    rows = [
        {
            "family_id": f"family-{number:03d}",
            "candidate_label": LABELS[number % len(LABELS)],
            "text": "never persisted",
        }
        for number in range(100)
    ]
    blind_values = {
        "packet-manifest.json": packet_raw,
        "packet-release-receipt.json": packet_receipt,
        "blind-test.jsonl": blind_raw,
        "capsule-descriptor.json": blind_descriptor,
    }
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    registry.chmod(0o700)
    monkeypatch.setattr(
        research,
        "_prepare_blind_inputs",
        lambda *_args: (
            training_values,
            fit,
            object(),
            packet,
            blind_descriptor,
            research.digest(blind_descriptor),
        ),
    )
    static = research._BlindStaticInputs(
        training_values=training_values,
        fit_values={},
        fit=fit,
        packet=packet,
        blind_descriptor=blind_descriptor,
        descriptor_digest=research.digest(blind_descriptor),
        blind_view_sha256=research.digest(blind_raw),
    )
    monkeypatch.setattr(research, "_prepare_blind_static", lambda *_args: static)
    monkeypatch.setattr(
        research,
        "_trusted_view_capsule",
        lambda *_args, **_kwargs: (blind_values, packet),
    )
    monkeypatch.setattr(training, "_load_rows", lambda *_args, **_kwargs: rows)

    class Interrupted(Exception):
        pass

    monkeypatch.setattr(
        research,
        "_scores_for_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Interrupted()),
    )
    with pytest.raises(Interrupted):
        begin_blind(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "fit",
            tmp_path / "blind",
            registry,
        )
    custody_key = research.blind_custody_key(
        research.digest(packet_raw), research.digest(blind_descriptor)
    )
    assert set((registry / f"{custody_key}.active").iterdir()) == {
        registry / f"{custody_key}.active" / "01-precommitted.json",
        registry / f"{custody_key}.active" / "02-running.json",
    }
    monkeypatch.setattr(research, "_scores_for_view", lambda *_args, **_kwargs: (logits, labels))
    run = resume_blind(
        tmp_path / "snapshot",
        tmp_path / "training",
        tmp_path / "fit",
        tmp_path / "blind",
        registry,
        custody_key,
    )
    values = research._read_capsule(run, research.BLIND_RUN_LIMITS)
    candidate = json.loads(values["calibration-candidate.json"])
    report = json.loads(values["blind-report.json"])
    running = json.loads(values["02-running.json"])
    evidence = {
        **{
            name: "9" * 64
            for name in (
                "protocol",
                "policy_registry",
                "states",
                "split_plan",
                "contributors",
                "annotations",
                "adjudications",
                "controls",
                "takedown_ledger",
            )
        },
        "packet": packet["packet_sha256"],
        "training_manifest": research.digest(training_values["training-manifest.json"]),
        "checkpoint": research.digest(training_values["checkpoint.safetensors"]),
        "fit": fit["fit_sha256"],
        "blind_view": research.digest(blind_raw),
        "exposure_run": running["state_sha256"],
        "blind_report": report["report_sha256"],
        "calibration_candidate": candidate["candidate_sha256"],
    }
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    external.chmod(0o700)
    receipt = external / "receipt.json"
    write_private(receipt, signed_receipt(private_key, evidence, scope="research_calibration"))
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output_parent.chmod(0o700)
    output = output_parent / "research-calibration.json"
    attacker = tmp_path / "attacker"
    attacker.mkdir(mode=0o700)
    attacker.chmod(0o700)
    anchored_parent = tmp_path / "output-anchored"
    import saracura.calibration.io as calibration_io

    original_writer = calibration_io.write_calibration_atomic_at
    original_link = os.link
    writer_fds: list[int] = []

    def swap_parent_writer(parent_fd: int, name: str, artifact: Any) -> None:
        writer_fds.append(parent_fd)
        os.rename(output_parent, anchored_parent)
        output_parent.symlink_to(attacker, target_is_directory=True)
        original_writer(parent_fd, name, artifact)

    def fd_bound_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        assert writer_fds
        assert (src_dir_fd, dst_dir_fd) == (writer_fds[0], writer_fds[0])
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(calibration_io, "write_calibration_atomic_at", swap_parent_writer)
    monkeypatch.setattr(os, "link", fd_bound_link)
    finalize_calibration(run, receipt, output)
    loaded = load_calibration(anchored_parent / output.name, context)
    assert isinstance(loaded, ResearchCalibrationArtifact)
    assert loaded.dataset_profile().split_manifest_sha256 == packet["packet_sha256"]
    assert not (attacker / output.name).exists()

    # Simulate the narrow crash window after all bytes/fsyncs but before rename:
    # resume verifies the complete descriptor-backed set and never rescores.
    active = registry / f"{custody_key}.active"
    os.rename(run, active)
    monkeypatch.setattr(
        research,
        "_prepare_blind_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("constructed sealed runtime")
        ),
    )
    monkeypatch.setattr(
        research,
        "_scores_for_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rescored sealed run")),
    )
    assert (
        resume_blind(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "fit",
            tmp_path / "blind",
            registry,
            custody_key,
        )
        == run
    )
    os.rename(run, active)
    report_path = active / "blind-report.json"
    # A partial report can be made self-digest-consistent by an attacker, but
    # resume must recompute the frozen aggregate rather than trusting it.
    (active / "03-sealed.json").unlink()
    (active / "capsule-descriptor.json").unlink()
    tampered_report = json.loads(report_path.read_bytes())
    tampered_metrics = tampered_report["metrics_after"]
    tampered_metrics["confusion_matrix"] = [
        [20 if column == (row + 1) % 5 else 0 for column in range(5)] for row in range(5)
    ]
    tampered_metrics["per_label"] = {
        label: {"support": 20, "precision": 0.0, "recall": 0.0, "f1": 0.0} for label in LABELS
    }
    tampered_metrics["accuracy"] = 0.0
    tampered_metrics["macro_f1"] = 0.0
    unsigned_report = dict(tampered_report)
    del unsigned_report["report_sha256"]
    tampered_report["report_sha256"] = research.digest(canonical(unsigned_report))
    report_path.write_bytes(canonical(tampered_report) + b"\n")
    report_path.chmod(0o600)
    monkeypatch.setattr(research, "_scores_for_view", lambda *_args, **_kwargs: (logits, labels))
    with pytest.raises(HumanResearchError, match="blind report replay binding"):
        resume_blind(
            tmp_path / "snapshot",
            tmp_path / "training",
            tmp_path / "fit",
            tmp_path / "blind",
            registry,
            custody_key,
        )
