"""CPU proof for the P2 command composition (never a production CLI override)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks import v02_corpus, v02_training, v02_training_loop
from benchmarks import v02_private_checkpoint as checkpoint
from benchmarks import v02_private_training as private
from benchmarks import v02_private_workflow as workflow
from saracura import contracts as saracura_contracts
from saracura import serialization as saracura_serialization
from saracura.serialization import canonical_json_bytes

_P1_TEST_SPEC = importlib.util.spec_from_file_location(
    "p2_real_p1_fixture", Path(__file__).with_name("test_v02_private_training.py")
)
if _P1_TEST_SPEC is None or _P1_TEST_SPEC.loader is None:
    raise RuntimeError("P1 private admission fixture is unavailable")
_P1_TEST_MODULE = importlib.util.module_from_spec(_P1_TEST_SPEC)
_P1_TEST_SPEC.loader.exec_module(_P1_TEST_MODULE)


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "identity_id": "train-a",
            "split": "train",
            "input_ids": [1, 2, 3, 4, 5],
            "end_option_indices": [1, 2],
            "decide_index": 4,
            "target": 0,
            "stratum": "pt-BR",
        },
        {
            "identity_id": "train-b",
            "split": "train",
            "input_ids": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "end_option_indices": [1, 2, 3, 4, 5, 6, 7, 8],
            "decide_index": 10,
            "target": 1,
            "stratum": "en",
        },
        {
            "identity_id": "dev-a",
            "split": "internal_dev",
            "input_ids": [11, 12, 13, 14, 15],
            "end_option_indices": [1, 2],
            "decide_index": 4,
            "target": 0,
            "stratum": "pt-BR",
        },
        {
            "identity_id": "dev-b",
            "split": "internal_dev",
            "input_ids": [16, 17, 18, 19, 20],
            "end_option_indices": [1, 2, 3],
            "decide_index": 4,
            "target": 1,
            "stratum": "en",
        },
    ]


def _runtime() -> workflow.WorkflowRuntime:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")

    def factory(_proof: Any, _base: Path) -> tuple[Any, Any, dict[str, Any]]:
        text = transformers.Qwen3_5TextConfig(
            vocab_size=4096,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            max_position_embeddings=256,
            layer_types=["full_attention"],
            pad_token_id=0,
        )
        base = transformers.Qwen3_5ForCausalLM(text).model.to(dtype=torch.bfloat16)
        base.requires_grad_(False)
        adapter = peft.get_peft_model(
            base,
            peft.LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type=peft.TaskType.FEATURE_EXTRACTION,
            ),
        ).to(dtype=torch.bfloat16)

        class Head(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.query = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
                self.key = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)

            def forward(self, state: Any, options: Any) -> Any:
                return (self.query(state).unsqueeze(1) * self.key(options)).sum(-1) / (8**0.5)

        return (
            adapter,
            Head(),
            {
                "effective_batch_size": 32,
                "max_epochs": 2,
                "patience": 2,
                "peak_learning_rate": "0.02",
                "weight_decay": 0.0,
                "warmup_fraction": 0.0,
                "gradient_clipping": 1.0,
                "lora_rank": 2,
                "lora_alpha": 4,
                "head_dim": 8,
            },
        )

    def reload(config: dict[str, Any], base: Path) -> tuple[Any, Any]:
        return factory(None, base)[:2]

    return workflow.WorkflowRuntime(
        torch=torch,
        peft=peft,
        safetensors=safetensors,
        device=torch.device("cpu"),
        train_factory=factory,
        reload_factory=reload,
        expected_tensor_count=6,
    )


def _admission() -> private.PrivateAdmission:
    return private.PrivateAdmission(
        proof=v02_training.AdmissionProof("a" * 64),
        descriptor={"rows_sha256": "b" * 64},
        rows=_rows(),
    )


def _write_closed(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(0o600)


def _workflow_inventory(root: Path) -> None:
    """Extend the real P1 fixture with modules imported by this composition."""
    path = root / "trainer-source.json"
    source_root = Path(workflow.__file__).resolve().parents[1]
    modules = (
        workflow,
        checkpoint,
        private,
        v02_training,
        v02_training_loop,
        v02_corpus,
        saracura_contracts,
        saracura_serialization,
    )
    files = []
    for module in modules:
        assert isinstance(module.__file__, str)
        source = Path(module.__file__).resolve()
        files.append(
            {
                "path": str(source.relative_to(source_root)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": source.stat().st_size,
            }
        )
    _write_closed(
        path,
        {
            "schema_version": "v02-trainer-source-inventory.v1",
            "source_revision": "8" * 40,
            "files": sorted(files, key=lambda item: item["path"]),
        },
    )


def test_library_injected_tiny_qwen_runs_claim_train_fresh_reload_and_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir(mode=0o700)
    run_inputs = root / "run-inputs.json"
    run_inputs.write_text("{}", encoding="utf-8")
    run_inputs.chmod(0o600)
    license_file = root / "license"
    license_file.write_text("self-authored test license", encoding="utf-8")
    license_file.chmod(0o600)
    admission_path = root / "admission.json"
    admission_path.write_bytes(
        canonical_json_bytes(
            {
                "artifact_id": "private-c1-r8-v1",
                "private_identity": "private-c1-r8-v1",
                "base_id": "self-authored-test-base",
                "base_revision": "test-revision",
                "base_inventory_sha256": "c" * 64,
                "tokenizer_inventory_sha256": "d" * 64,
                "renderer_sha256": "e" * 64,
                "candidate_config_sha256": "f" * 64,
                "training_authorized": True,
                "private_only": True,
                "publication_authorized": False,
            }
        )
        + b"\n"
    )
    admission_path.chmod(0o600)
    base = tmp_path / "base"
    base.mkdir()
    calls = {"admission": 0}

    def verified(**_kwargs: Any) -> private.PrivateAdmission:
        calls["admission"] += 1
        return _admission()

    monkeypatch.setattr(workflow, "verify_admission", verified)
    monkeypatch.setattr(private, "load_run_inputs", lambda *_args: {"base_directory": str(base)})
    monkeypatch.setattr(workflow, "_licenses", lambda *_args: {"student": license_file})
    runtime = _runtime()
    work = output_parent / "work"
    publish_actual = private.publish_step_zero_claim

    def interrupted_publication(**kwargs: Any) -> None:
        publish_actual(**kwargs)
        raise RuntimeError("self-authored interruption after paired step-zero publication")

    monkeypatch.setattr(private, "publish_step_zero_claim", interrupted_publication)
    with pytest.raises(RuntimeError, match="self-authored interruption"):
        workflow.train_private(
            root=root,
            run_inputs_path=run_inputs,
            admission_path=admission_path,
            output=work,
            runtime=runtime,
        )
    assert not work.exists()
    monkeypatch.setattr(private, "publish_step_zero_claim", publish_actual)
    exported = workflow.train_private(
        root=root,
        run_inputs_path=run_inputs,
        admission_path=admission_path,
        resume=work,
        runtime=runtime,
    )
    before_export = (work / "export" / "adapter.safetensors").read_bytes()
    assert (
        workflow.train_private(
            root=root,
            run_inputs_path=run_inputs,
            admission_path=admission_path,
            resume=work,
            runtime=runtime,
        )
        == exported
    )
    assert (work / "export" / "adapter.safetensors").read_bytes() == before_export
    assert exported["private_identity"] == "private-c1-r8-v1"
    assert (root / "private-run-claims" / "private-c1-r8-v1" / "initial-state.pt").is_file()
    assert (work / "resume-state.pt").is_file()
    assert (work / "export" / "adapter.safetensors").is_file()
    final = output_parent / "final"
    result = workflow.reload_private(
        root=root,
        run_inputs_path=run_inputs,
        admission_path=admission_path,
        export=work / "export",
        output=final,
        runtime=runtime,
    )
    assert result["private_only"] is True
    assert workflow.verify_private_checkpoint(checkpoint_path=final)["automation_allowed"] is False
    smoke = workflow.smoke_private(
        root=root,
        run_inputs_path=run_inputs,
        admission_path=admission_path,
        output=output_parent / "smoke.json",
        runtime=runtime,
    )
    assert smoke["optimizer_steps"] == 1
    assert smoke["private_identity_consumed"] is False
    assert calls["admission"] == 5


def test_resume_rejects_changed_identity_and_cli_has_no_runtime_test_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    run_inputs = root / "run-inputs.json"
    run_inputs.write_text("{}", encoding="utf-8")
    run_inputs.chmod(0o600)
    admission = root / "admission.json"
    admission.write_text("{}", encoding="utf-8")
    admission.chmod(0o600)
    monkeypatch.setattr(workflow, "verify_admission", lambda **_kwargs: _admission())
    monkeypatch.setattr(
        private, "load_run_inputs", lambda *_args: {"base_directory": str(tmp_path)}
    )
    with pytest.raises(RuntimeError, match="IMPLEMENTATION_BLOCKED"):
        workflow.train_private(
            root=root,
            run_inputs_path=run_inputs,
            admission_path=admission,
            resume=tmp_path / "unclaimed",
            runtime=_runtime(),
        )
    called: dict[str, Path] = {}

    def verify(*, checkpoint_path: Path) -> dict[str, Any]:
        called["checkpoint"] = checkpoint_path
        return {}

    monkeypatch.setattr(workflow, "verify_private_checkpoint", verify)
    monkeypatch.setattr(
        sys, "argv", ["workflow", "verify-private-checkpoint", "--checkpoint", "/tmp/final"]
    )
    assert workflow.main() == 0
    assert called["checkpoint"] == Path("/tmp/final")


def test_actual_p1_admission_wires_the_shared_smoke_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, run_inputs, identity = _P1_TEST_MODULE._admission_fixture(monkeypatch, tmp_path)
    root = fixture["root"]
    _workflow_inventory(root)

    def probe() -> dict[str, Any]:
        return {
            "kernel_boot_id": identity["kernel_boot_id"],
            "gpu_uuid": identity["gpu_uuid"],
            "gpu_memory_mib": identity["gpu_memory_mib"],
        }

    admission_path = root / "admission.json"
    workflow.create_admission(
        root=root, run_inputs_path=run_inputs, output=admission_path, machine_probe=probe
    )
    monkeypatch.setattr(private, "_default_machine_probe", probe)
    verified = workflow.verify_admission(
        root=root, run_inputs_path=run_inputs, admission_path=admission_path
    )
    assert any(len(row["end_option_indices"]) == 8 for row in verified.rows)
    subset = workflow._smoke_train_rows(verified.rows)
    assert len(subset) <= 32
    assert max(len(row["input_ids"]) for row in subset) == max(
        len(row["input_ids"]) for row in verified.rows if row["split"] == "train"
    )
    assert any(len(row["end_option_indices"]) == 8 for row in subset)
    smoke = workflow.smoke_private(
        root=root,
        run_inputs_path=run_inputs,
        admission_path=admission_path,
        output=root / "smoke.json",
        runtime=_runtime(),
    )
    assert smoke["optimizer_steps"] == 1
    assert smoke["private_identity_consumed"] is False


def test_production_bf16_factory_accepts_actual_four_field_export_without_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: dict[str, Any] = {}

    class Module:
        def __init__(self) -> None:
            pass

        def to(self, *args: Any, **kwargs: Any) -> Any:
            if self is adapter:
                calls["adapter_to"] = kwargs
            return self

        def requires_grad_(self, _enabled: bool) -> None:
            pass

    adapter = Module()
    text = Module()

    def load(_base: str, **kwargs: Any) -> Any:
        calls["load"] = kwargs
        return SimpleNamespace(model=SimpleNamespace(language_model=text))

    def lora(**kwargs: Any) -> dict[str, Any]:
        calls["lora"] = kwargs
        return kwargs

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda name: name,
        bfloat16="bf16",
        nn=SimpleNamespace(Module=Module, Linear=lambda *_args, **_kwargs: Module()),
    )
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(hidden_size=2560)
        ),
        Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load),
    )
    peft = SimpleNamespace(
        get_peft_model=lambda *_args: adapter,
        LoraConfig=lora,
        TaskType=SimpleNamespace(FEATURE_EXTRACTION="features"),
    )
    monkeypatch.setattr(
        v02_training, "_training_runtime", lambda: (torch, transformers, peft, object())
    )
    monkeypatch.setattr(v02_training, "qwen35_text_config", lambda config: config)
    runtime = workflow._production_runtime()
    actual_config = {"adapter_name": "default", "lora_rank": 8, "lora_alpha": 16, "head_dim": 256}
    runtime.reload_factory(actual_config, Path("/self-authored-base"))
    assert calls["load"]["local_files_only"] is True
    assert calls["load"]["torch_dtype"] == "bf16"
    assert "quantization_config" not in calls["load"]
    assert calls["adapter_to"] == {"device": "cuda"}
    assert calls["lora"]["lora_dropout"] == 0.05
    assert len(calls["lora"]["target_modules"]) == 248
    with pytest.raises(ValueError, match="frozen c1"):
        runtime.reload_factory({**actual_config, "lora_rank": 16}, Path("/self-authored-base"))
