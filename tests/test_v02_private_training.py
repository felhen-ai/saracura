from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks import v02_corpus, v02_training, v02_training_loop
from benchmarks import v02_private_training as private
from saracura import serialization as saracura_serialization
from saracura.serialization import canonical_json_bytes

_CORPUS_TEST_SPEC = importlib.util.spec_from_file_location(
    "p1_real_fixture", Path(__file__).with_name("test_v02_corpus.py")
)
if _CORPUS_TEST_SPEC is None or _CORPUS_TEST_SPEC.loader is None:
    raise RuntimeError("P1 self-authored fixture is unavailable")
_CORPUS_TEST_MODULE = importlib.util.module_from_spec(_CORPUS_TEST_SPEC)
_CORPUS_TEST_SPEC.loader.exec_module(_CORPUS_TEST_MODULE)
_sealer_fixture = _CORPUS_TEST_MODULE._sealer_fixture

_REALISTIC_RENDERER_SOURCE = _CORPUS_TEST_MODULE._RENDERER_SOURCE.replace(
    """    ids = user_tokens(tokenizer, state)
    return {"ids": ids, "labels": [OPT_NONE], "state_truncated": "TRUNCATE" in state}
""",
    """    questions = record["questions"]
    if len(questions) != 1:
        raise ValueError("one question")
    ids = user_tokens(tokenizer, state)
    end_indices = []
    for option in questions[0]["options"]:
        ids.extend(user_tokens(tokenizer, option))
        end_indices.append(len(ids) - 1)
    ids.extend(user_tokens(tokenizer, "<|fim_suffix|>"))
    return {
        "ids": ids,
        "labels": [OPT_NONE],
        "state_truncated": "TRUNCATE" in state,
        "decide_idx": [len(ids) - 1],
        "opt_idx": [end_indices],
    }
""",
)
if _REALISTIC_RENDERER_SOURCE == _CORPUS_TEST_MODULE._RENDERER_SOURCE:
    raise RuntimeError("P1 self-authored renderer fixture was not upgraded")


def _state(bindings: str) -> dict[str, object]:
    values: dict[str, object] = {key: None for key in private._RESUME_FIELDS}
    values.update(
        {
            "schema_version": "v02-private-resume-state.v1",
            "bindings_sha256": bindings,
            "private_identity": "private-c1-r8-v1",
            "epoch": 1,
            "next_batch_index": 0,
            "optimizer_steps": 0,
            "adapter_state": {},
            "head_state": {},
            "optimizer_state": {},
            "scheduler_state": {},
            "python_rng_state": [3, [1, 2], None],
            "numpy_rng_state": {
                "name": "MT19937",
                "keys": [1],
                "pos": 0,
                "has_gauss": 0,
                "cached_gaussian": 0.0,
            },
            "torch_rng_state": [],
            "cuda_rng_states": [],
            "best_adapter_state": {},
            "best_head_state": {},
            "best_epoch": 0,
            "best_score": -1.0,
            "stale_epochs": 0,
            "initial_trainable_sha256": "a" * 64,
            "base_parameter_sha256": "b" * 64,
            "duration_seconds": 0.0,
            "peak_gpu_bytes": 0,
        }
    )
    return values


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(0o600)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pinned_base(tmp_path: Path, root: Path) -> tuple[Path, Path]:
    base = tmp_path / "acquired-base"
    base.mkdir(mode=0o700)
    layers = [
        "full_attention" if index in range(3, 32, 4) else "linear_attention" for index in range(32)
    ]
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "intermediate_size": 9216,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "layer_types": layers,
        },
    }
    files = {
        "config.json": canonical_json_bytes(cast(Any, config)) + b"\n",
        "tokenizer.json": b"{}\n",
        "LICENSE": b"self-authored test license\n",
        "model-00001-of-00001.safetensors": b"self-authored no-weights fixture\n",
    }
    index = {"weight_map": {"model.test": "model-00001-of-00001.safetensors"}}
    files["model.safetensors.index.json"] = canonical_json_bytes(cast(Any, index)) + b"\n"
    for name, raw in files.items():
        target = base / name
        target.write_bytes(raw)
        target.chmod(0o400)
    inventory = {
        "role": "student",
        "model": "Qwen/Qwen3.5-4B-Base",
        "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "status": "SOURCE_BYTES_VERIFIED_NOT_LOADED",
        "inventory": [
            {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in sorted(files.items())
        ],
    }
    inventory_path = root / "base-inventory.json"
    _write(inventory_path, inventory)
    base.chmod(0o500)
    return base, inventory_path


def _admission_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    monkeypatch.setattr(_CORPUS_TEST_MODULE, "_RENDERER_SOURCE", _REALISTIC_RENDERER_SOURCE)
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    root = fixture["root"]
    source_paths = [
        Path(private.__file__).resolve(),
        Path(v02_training.__file__).resolve(),
        Path(v02_corpus.__file__).resolve(),
        Path(v02_training_loop.__file__).resolve(),
        Path(saracura_serialization.__file__).resolve(),
    ]
    runtime_code_map = {"kev-model.py": "1" * 64, "runtime-helper.py": "2" * 64}
    lock_path = root / "runtime-lock.json"
    lock = json.loads(lock_path.read_bytes())
    lock["gpu_memory_mib"] = 48
    lock["code_sha256"] = runtime_code_map
    _write(lock_path, lock)
    sealer_path = fixture["inputs"]
    sealer = json.loads(sealer_path.read_bytes())
    grant_path = root / cast(str, sealer["environment_grant_file"])
    grant = json.loads(grant_path.read_bytes())
    grant["runtime_lock_digest"] = _digest(lock)
    _write(grant_path, grant)
    licenses = {
        role: (root / path).read_bytes()
        for role, path in cast(dict[str, str], sealer["license_files"]).items()
    }
    grant_digest = v02_corpus.validate_environment_grant(grant, licenses)
    for path in (root / "private-control").glob("*.json"):
        value = json.loads(path.read_bytes())
        value["grant_digest"] = grant_digest
        _write(path, value)
    for path in (root / "role-reservations").glob("*.json"):
        value = json.loads(path.read_bytes())
        value["grant_digest"] = grant_digest
        _write(path, value)
    sealer["environment_grant_sha256"] = hashlib.sha256(grant_path.read_bytes()).hexdigest()
    sealer["runtime_lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    _write(sealer_path, sealer)
    capsule = v02_corpus.seal_training_capsule(
        root=root,
        plan_path=fixture["plan"],
        receipt_path=fixture["receipt"],
        exclusions_path=fixture["exclusions"],
        output_parent=root,
        artifact_id="training-capsule",
        sealer_inputs_path=sealer_path,
    )
    source_root = Path(private.__file__).resolve().parents[1]
    source_inventory = {
        "schema_version": "v02-trainer-source-inventory.v1",
        "source_revision": "8" * 40,
        "files": [
            {
                "path": str(path.relative_to(source_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in sorted(source_paths)
        ],
    }
    trainer_path = root / "trainer-source.json"
    _write(trainer_path, source_inventory)
    manifest_path = root / "reviewed-runtime.json"
    _write(
        manifest_path,
        {
            "schema_version": "private-reviewed-runtime-code.v1",
            "source_commit": lock["source_commit"],
            "code_sha256": runtime_code_map,
        },
    )
    preflight_path = root / "p3-preflight.json"
    _write(
        preflight_path,
        {
            "schema_version": "c7-preflight-observation.v1",
            "status": "OBSERVED_REVIEW_PENDING",
            "source_commit": lock["source_commit"],
            "source_inventory_digest": lock["public_source_integrity"]["source_inventory_digest"],
            "code_manifest_digest": _digest(runtime_code_map),
            "training_plan_digest": grant["training_plan_digest"],
            "sealed_plan_digest": grant["sealed_plan_digest"],
            "micro_selection_digest": "3" * 64,
            "grammar_receipt_digest": "4" * 64,
            "feasibility_receipt_digest": "5" * 64,
            "grant_digest": grant_digest,
            "runtime_lock_digest": grant["runtime_lock_digest"],
            "runtime_evidence_digests": {},
            "fresh_asset_probe_digests": {},
            "historical_model_metadata_sha256": "6" * 64,
            "historical_measurements": {},
            "model_calls": {},
            "reservations": {},
        },
    )
    identity = {
        "pod_id": "test-pod",
        "region": grant["region"],
        "image_sha256": lock["image_digest"],
        "started_at": "2026-10-01T00:00:00Z",
        "ssh_endpoint": "test.example:22:test",
        "kernel_boot_id": "test-boot",
        "gpu_uuid": "test-gpu",
        "gpu_memory_mib": 48,
    }
    environment_path = root / "physical-environment.json"
    _write(
        environment_path,
        {
            "schema_version": "v02-private-physical-environment.v1",
            "artifact_id": "private-c1-r8-v1",
            "status": "PHYSICALLY_ADMITTED_TRAINING_ONLY",
            "host_identity_before": identity,
            "host_identity_after": identity,
            "preflight_file": preflight_path.name,
            "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
            "environment_grant_sha256": hashlib.sha256(grant_path.read_bytes()).hexdigest(),
            "runtime_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "source_inventory_sha256": lock["public_source_integrity"]["source_inventory_digest"],
            "reviewed_code_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "weight_roles": ["training_author", "independent_reviewer"],
            "sealed_weights_loaded": False,
        },
    )
    base, base_inventory = _pinned_base(tmp_path, root)
    run_inputs = root / "private-run-inputs.json"
    _write(
        run_inputs,
        {
            "schema_version": "v02-private-run-inputs.v1",
            "artifact_id": "private-c1-r8-v1",
            "plan_file": fixture["plan"].name,
            "receipt_file": fixture["receipt"].name,
            "capsule_file": str(capsule.relative_to(root)),
            "exclusions_file": fixture["exclusions"].name,
            "sealer_inputs_file": sealer_path.name,
            "environment_observation_file": environment_path.name,
            "base_directory": str(base),
            "base_inventory_file": base_inventory.name,
            "trainer_source_inventory_file": trainer_path.name,
            "renderer_preflight_file": "renderer-preflight.json",
            "reviewed_runtime_manifest_file": manifest_path.name,
        },
    )
    return fixture, run_inputs, identity


def test_private_admission_recomputes_p1_and_prepares_lossless_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, run_inputs, identity = _admission_fixture(monkeypatch, tmp_path)
    output = fixture["root"] / "admission.json"
    result = private.create_private_admission(
        root=fixture["root"],
        run_inputs_path=run_inputs,
        output=output,
        machine_probe=lambda: {
            "kernel_boot_id": identity["kernel_boot_id"],
            "gpu_uuid": identity["gpu_uuid"],
            "gpu_memory_mib": identity["gpu_memory_mib"],
        },
    )
    assert output.stat().st_mode & 0o777 == 0o600
    assert len(result.rows) == 1600
    assert {row["stratum"] for row in result.rows} == {"pt-BR", "en"}
    assert all(row["decide_index"] > row["end_option_indices"][-1] for row in result.rows)
    _sealer, _grant, renderer, tokenizer = private._validated_runtime_components(
        fixture["root"], private.load_run_inputs(fixture["root"], run_inputs)
    )
    source = result.descriptor
    _descriptor, capsule_rows = v02_corpus.verify_training_capsule(
        fixture["root"] / "training-capsule" / "capsule.json",
        root=fixture["root"],
        plan_path=fixture["plan"],
        receipt_path=fixture["receipt"],
        exclusions_path=fixture["exclusions"],
        sealer_inputs_path=fixture["inputs"],
    )
    assert source == _descriptor
    assert [row["identity_id"] for row in result.rows] == [
        row["identity_id"] for row in capsule_rows
    ]
    preflight = json.loads(
        (
            fixture["root"]
            / private.load_run_inputs(fixture["root"], run_inputs)["renderer_preflight_file"]
        ).read_bytes()
    )
    assert set(preflight) == {
        "schema_version",
        "artifact_id",
        "capsule_sha256",
        "candidate_renderer_sha256",
        "kev_rendering_contract_sha256",
        "accepted_records",
        "candidate_all_records_pass",
        "kev_all_records_pass",
    }
    assert preflight["capsule_sha256"] == private._sha(
        fixture["root"] / "training-capsule" / "capsule.json"
    )
    assert preflight["candidate_renderer_sha256"] == v02_corpus.candidate_rendering_digest()
    assert preflight["kev_rendering_contract_sha256"] == v02_corpus.kev_rendering_function_digest()
    assert preflight["accepted_records"] == len(capsule_rows)
    first = capsule_rows[0]
    ids, decide_index, end_indices = private._render_causal_record(
        renderer,
        tokenizer,
        {"state": first["state"], "question": first["instruction"], "options": first["options"]},
    )
    assert result.rows[0]["input_ids"] == ids
    assert result.rows[0]["decide_index"] == decide_index
    assert result.rows[0]["end_option_indices"] == end_indices
    verified = private.verify_private_admission(
        root=fixture["root"],
        run_inputs_path=run_inputs,
        admission_path=output,
        machine_probe=lambda: {
            "kernel_boot_id": identity["kernel_boot_id"],
            "gpu_uuid": identity["gpu_uuid"],
            "gpu_memory_mib": identity["gpu_memory_mib"],
        },
    )
    assert verified.proof == result.proof
    with pytest.raises(ValueError, match="state"):
        private.prepare_verified_rows(
            [
                {
                    "state": "x" * 385,
                    "instruction": "self-authored overflow check",
                    "options": [{"id": "a", "description": "a"}, {"id": "b", "description": "b"}],
                    "split": "train",
                    "locale": "pt_br",
                    "gold_index": 0,
                }
            ],
            renderer=renderer,
            tokenizer=tokenizer,
        )
    malformed = _REALISTIC_RENDERER_SOURCE.replace(
        '"decide_idx": [len(ids) - 1],', '"decide_idx": [len(ids)],'
    )
    with monkeypatch.context() as local:
        raw = _CORPUS_TEST_MODULE._pin_renderer(local, malformed)
        malformed_renderer = v02_corpus.load_pinned_renderer(raw)
        with pytest.raises(ValueError, match="indices"):
            private.prepare_verified_rows([first], renderer=malformed_renderer, tokenizer=tokenizer)
    missing_marker = _REALISTIC_RENDERER_SOURCE.replace(
        '"opt_idx": [end_indices],', '"unknown_marker_idx": [end_indices],'
    )
    with monkeypatch.context() as local:
        raw = _CORPUS_TEST_MODULE._pin_renderer(local, missing_marker)
        missing_marker_renderer = v02_corpus.load_pinned_renderer(raw)
        with pytest.raises(ValueError, match="closed causal record"):
            private.prepare_verified_rows(
                [first], renderer=missing_marker_renderer, tokenizer=tokenizer
            )
    with pytest.raises(FileExistsError):
        private.create_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            output=output,
            machine_probe=lambda: identity,
        )


def test_private_admission_rejects_boot_base_and_overflow_before_ml_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, run_inputs, identity = _admission_fixture(monkeypatch, tmp_path)
    output = fixture["root"] / "admission.json"

    def probe() -> dict[str, Any]:
        return {
            "kernel_boot_id": identity["kernel_boot_id"],
            "gpu_uuid": identity["gpu_uuid"],
            "gpu_memory_mib": identity["gpu_memory_mib"],
        }

    private.create_private_admission(
        root=fixture["root"], run_inputs_path=run_inputs, output=output, machine_probe=probe
    )
    manifest = fixture["root"] / "reviewed-runtime.json"
    actual_manifest = fixture["root"] / "reviewed-runtime-actual.json"
    manifest.rename(actual_manifest)
    manifest.symlink_to(actual_manifest.name)
    with pytest.raises(ValueError, match="symlink"):
        private.load_run_inputs(fixture["root"], run_inputs)
    manifest.unlink()
    actual_manifest.rename(manifest)
    with monkeypatch.context() as local:
        uid = os.geteuid()
        local.setattr(os, "geteuid", lambda: uid + 1)
        with pytest.raises(ValueError, match="owned"):
            private.load_run_inputs(fixture["root"], run_inputs)
    with pytest.raises(ValueError, match=r"boot|GPU"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=lambda: {},
        )
    with pytest.raises(ValueError, match=r"boot|GPU"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=lambda: {**probe(), "gpu_memory_mib": 49},
        )
    preflight_path = fixture["root"] / "p3-preflight.json"
    preflight = json.loads(preflight_path.read_bytes())
    preflight["grant_digest"] = "0" * 64
    _write(preflight_path, preflight)
    with pytest.raises(ValueError, match="preflight"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=probe,
        )
    preflight["grant_digest"] = v02_corpus.validate_environment_grant(
        json.loads((fixture["root"] / "environment-grant.json").read_bytes()),
        {
            role: (fixture["root"] / path).read_bytes()
            for role, path in json.loads(fixture["inputs"].read_bytes())["license_files"].items()
        },
    )
    _write(preflight_path, preflight)
    inventory_path = fixture["root"] / "trainer-source.json"
    inventory = json.loads(inventory_path.read_bytes())
    inventory["files"].append(inventory["files"][-1])
    _write(inventory_path, inventory)
    with pytest.raises(ValueError, match="sorted"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=probe,
        )
    inventory["files"].pop()
    _write(inventory_path, inventory)
    shadow = tmp_path / "shadow-private-training.py"
    shadow.symlink_to(Path(private.__file__).resolve())
    with monkeypatch.context() as local:
        local.setattr(private, "__file__", str(shadow))
        with pytest.raises(ValueError, match="shadowed"):
            private.verify_private_admission(
                root=fixture["root"],
                run_inputs_path=run_inputs,
                admission_path=output,
                machine_probe=probe,
            )
    sealer = json.loads(fixture["inputs"].read_bytes())
    grant_path = fixture["root"] / sealer["environment_grant_file"]
    grant = json.loads(grant_path.read_bytes())
    grant["region"] = "changed-region"
    _write(grant_path, grant)
    with pytest.raises(ValueError, match=r"grant|digest|rights"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=probe,
        )
    grant["region"] = identity["region"]
    _write(grant_path, grant)
    with pytest.raises(ValueError, match=r"boot|GPU"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=lambda: {**probe(), "kernel_boot_id": "changed"},
        )
    source = Path(json.loads(run_inputs.read_bytes())["base_directory"]) / "tokenizer.json"
    source.chmod(0o600)
    source.write_bytes(b"changed")
    source.chmod(0o400)
    with pytest.raises(ValueError, match="base source bytes"):
        private.verify_private_admission(
            root=fixture["root"],
            run_inputs_path=run_inputs,
            admission_path=output,
            machine_probe=probe,
        )


def test_step_zero_claim_is_create_only_and_safe_to_reload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    inputs = root / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    inputs.chmod(0o600)
    work = tmp_path / "work"
    bindings = "c" * 64
    with private.private_identity_lock(root):
        target = private.publish_step_zero_claim(
            root=root,
            work_directory=work,
            admission_sha256="a" * 64,
            run_inputs_path=inputs,
            bindings_sha256=bindings,
            initial_state=_state(bindings),
            torch=torch,
        )
    assert (target / "claim.json").is_file()
    state = private.load_resume_state(
        target / "initial-state.pt", torch=torch, bindings_sha256=bindings
    )
    assert state["optimizer_steps"] == 0
    with private.private_identity_lock(root), pytest.raises(FileExistsError):
        private.publish_step_zero_claim(
            root=root,
            work_directory=tmp_path / "other",
            admission_sha256="a" * 64,
            run_inputs_path=inputs,
            bindings_sha256=bindings,
            initial_state=_state(bindings),
            torch=torch,
        )


def test_step_zero_claim_preserves_concurrent_empty_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    inputs = root / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    inputs.chmod(0o600)
    publish = private._publish_directory_exclusively
    reservation_inode: list[int] = []

    def reserve_before_publication(staging: Path, target: Path) -> None:
        target.mkdir(mode=0o700)
        reservation_inode.append(target.stat().st_ino)
        publish(staging, target)

    monkeypatch.setattr(private, "_publish_directory_exclusively", reserve_before_publication)
    with private.private_identity_lock(root), pytest.raises(FileExistsError):
        private.publish_step_zero_claim(
            root=root,
            work_directory=tmp_path / "work",
            admission_sha256="a" * 64,
            run_inputs_path=inputs,
            bindings_sha256="c" * 64,
            initial_state=_state("c" * 64),
            torch=torch,
        )
    target = root / "private-run-claims" / "private-c1-r8-v1"
    assert target.stat().st_ino == reservation_inode[0]
    assert list(target.iterdir()) == []
    staging = next(target.parent.glob(".private-c1-r8-v1-*"))
    assert (staging / "claim.json").is_file()
    assert (staging / "initial-state.pt").is_file()


def test_pinned_upstream_json_keeps_original_formatting(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    base, inventory_path = _pinned_base(tmp_path, root)
    inventory = json.loads(inventory_path.read_bytes())
    for name in ("config.json", "model.safetensors.index.json"):
        path = base / name
        value = json.loads(path.read_bytes())
        raw = json.dumps(value, indent=2).encode() + b"\n"
        path.chmod(0o600)
        path.write_bytes(raw)
        path.chmod(0o400)
        for entry in inventory["inventory"]:
            if entry["name"] == name:
                entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    _write(inventory_path, inventory)
    original = {
        name: (base / name).read_bytes() for name in ("config.json", "model.safetensors.index.json")
    }
    private._verify_base_inventory(base, inventory_path)
    assert len(private._module_inventory(base / "config.json")) == 64
    assert all((base / name).read_bytes() == raw for name, raw in original.items())
    path = base / "config.json"
    path.chmod(0o600)
    path.write_bytes(original["config.json"] + b" ")
    path.chmod(0o400)
    with pytest.raises(ValueError, match="base source bytes changed"):
        private._verify_base_inventory(base, inventory_path)


def test_upstream_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"text_config": {}, "text_config": {}}')
    with pytest.raises(ValueError, match="invalid JSON"):
        private._module_inventory(config)


def test_private_inventory_still_requires_canonical_json(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    base, inventory = _pinned_base(tmp_path, root)
    inventory.write_text(json.dumps(json.loads(inventory.read_bytes()), indent=2))
    with pytest.raises(ValueError, match="base inventory must contain canonical JSON"):
        private._verify_base_inventory(base, inventory)
