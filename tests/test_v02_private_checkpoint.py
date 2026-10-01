from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from benchmarks import v02_private_checkpoint as checkpoint
from benchmarks import v02_training
from benchmarks.v02_training_loop import run_optimization
from saracura.serialization import canonical_json_bytes


def _hash(letter: str) -> str:
    return letter * 64


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "identity_id": "sample-a",
            "split": "train",
            "input_ids": [1, 2, 3, 4, 5],
            "end_option_indices": [1, 2],
            "decide_index": 4,
            "target": 0,
            "stratum": "pt",
        },
        {
            "identity_id": "sample-b",
            "split": "train",
            "input_ids": [6, 7, 8, 9, 10],
            "end_option_indices": [1, 2, 3],
            "decide_index": 4,
            "target": 1,
            "stratum": "en",
        },
        {
            "identity_id": "sample-c",
            "split": "internal_dev",
            "input_ids": [11, 12, 13, 14, 15],
            "end_option_indices": [1, 2],
            "decide_index": 4,
            "target": 0,
            "stratum": "pt",
        },
        {
            "identity_id": "sample-d",
            "split": "internal_dev",
            "input_ids": [16, 17, 18, 19, 20],
            "end_option_indices": [1, 2, 3],
            "decide_index": 4,
            "target": 1,
            "stratum": "en",
        },
    ]


def _components(torch: Any, transformers: Any, peft: Any, *, alpha: int = 4) -> tuple[Any, Any]:
    v02_training.configure_determinism(torch, 20260929)
    text = transformers.Qwen3_5TextConfig(
        vocab_size=64,
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
        max_position_embeddings=32,
        layer_types=["full_attention"],
        pad_token_id=0,
    )
    base = transformers.Qwen3_5ForCausalLM(text).model.to(dtype=torch.bfloat16)
    base.requires_grad_(False)
    adapter = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=alpha,
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

    return adapter, Head()


def _train(torch: Any, peft: Any, adapter: Any, head: Any) -> dict[str, Any]:
    rows = _rows()
    return run_optimization(
        torch=torch,
        adapter=adapter,
        pointer_head=head,
        train_rows=[row for row in rows if row["split"] == "train"],
        dev_rows=[row for row in rows if row["split"] == "internal_dev"],
        config={
            "effective_batch_size": 2,
            "max_epochs": 2,
            "patience": 2,
            "peak_learning_rate": "0.02",
            "weight_decay": 0.0,
            "warmup_fraction": 0.0,
            "gradient_clipping": 1.0,
        },
        device=torch.device("cpu"),
        token_batches=v02_training._token_batches,
        to_device=v02_training._to_device,
        score_batch=v02_training._score_token_batch,
        adapter_state=lambda model: {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if "lora_" in name
        },
        load_adapter_state=peft.set_peft_model_state_dict,
    )


def _admission() -> dict[str, Any]:
    return {
        "artifact_id": "private-c1-r8-v1",
        "private_identity": "private-c1-r8-v1",
        "admission_sha256": _hash("a"),
        "base_id": "Qwen/Qwen3.5-4B-Base",
        "base_revision": "revision",
        "base_inventory_sha256": _hash("b"),
        "tokenizer_inventory_sha256": _hash("c"),
        "renderer_sha256": _hash("d"),
        "candidate_config_sha256": _hash("e"),
        "training_authorized": True,
        "private_only": True,
        "publication_authorized": False,
    }


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": "private-c1-r8-v1",
        "admission_sha256": _hash("a"),
        "private_identity": "private-c1-r8-v1",
        "seed": 20260929,
        "epochs_completed": 2,
        "optimizer_steps": result["optimizer_steps"],
        "train_rows": 2,
        "internal_dev_rows": 2,
        "best_epoch": result["best_epoch"],
        "best_internal_dev_macro_accuracy": result["internal_dev_stratified_macro_accuracy"],
        "finite_updates": True,
        "base_unchanged": True,
        "initial_trainable_sha256": _hash("f"),
        "duration_seconds": 0.0,
        "peak_gpu_bytes": 0,
    }


def _export(private: Path, torch: Any, transformers: Any, peft: Any, safetensors: Any) -> None:
    license_file = private / "license"
    license_file.write_text("self-authored test license", encoding="utf-8")
    adapter, head = _components(torch, transformers, peft)
    result = _train(torch, peft, adapter, head)
    checkpoint.export_private_checkpoint(
        work=private,
        adapter=adapter,
        pointer_head=head,
        metrics=_metrics(result),
        admission=_admission(),
        config={"adapter_name": "default", "lora_rank": 2, "lora_alpha": 4, "head_dim": 8},
        licenses={"student": license_file},
        torch=torch,
        safetensors=safetensors,
    )


def test_actual_tiny_qwen_peft_export_fresh_bf16_reload_and_final_receipts(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    license_file = private / "license"
    license_file.write_text("self-authored test license", encoding="utf-8")
    adapter, head = _components(torch, transformers, peft)
    result = _train(torch, peft, adapter, head)
    descriptor = checkpoint.export_private_checkpoint(
        work=private,
        adapter=adapter,
        pointer_head=head,
        metrics=_metrics(result),
        admission=_admission(),
        config={"adapter_name": "default", "lora_rank": 2, "lora_alpha": 4, "head_dim": 8},
        licenses={"student": license_file},
        torch=torch,
        safetensors=safetensors,
    )
    assert {item["path"] for item in descriptor["files"]} == {
        "adapter.safetensors",
        "pointer-head.safetensors",
        "config.json",
        "licenses/student",
    }
    assert all(
        not any(word in item["path"].casefold() for word in ("base", "optimizer", "corpus"))
        for item in descriptor["files"]
    )
    receipt = checkpoint.fresh_reload_proof(
        export=private / "export",
        factory=lambda: _components(torch, transformers, peft),
        samples=_rows()[:2],
        dev_rows=_rows()[2:],
        torch=torch,
        safetensors=safetensors,
        expected_tensor_count=6,
    )
    assert receipt["precision"] == "bf16"
    assert receipt["functional_effect_verified"] is True
    learned_before = (private / "export" / "adapter.safetensors").read_bytes()
    with pytest.raises(ValueError, match="reload receipt is not closed"):
        checkpoint.publish_final_checkpoint(
            export=private / "export", final=private / "failed-final", reload_receipt={}
        )
    assert (private / "export" / "adapter.safetensors").read_bytes() == learned_before
    final = private / "final"
    published = checkpoint.publish_final_checkpoint(
        export=private / "export", final=final, reload_receipt=receipt
    )
    assert checkpoint.verify_final_checkpoint(final) == published
    tampered = dict(published)
    tampered["automation_allowed"] = True
    (final / "checkpoint.json").write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(ValueError, match="checkpoint safety flags"):
        checkpoint.verify_final_checkpoint(final)
    with pytest.raises(FileExistsError):
        checkpoint.publish_final_checkpoint(
            export=private / "export", final=final, reload_receipt=receipt
        )


def test_export_and_reload_negatives_reject_tamper_and_zero_effect(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    license_file = private / "license"
    license_file.write_text("self-authored test license", encoding="utf-8")
    adapter, head = _components(torch, transformers, peft)
    result = _train(torch, peft, adapter, head)
    checkpoint.export_private_checkpoint(
        work=private,
        adapter=adapter,
        pointer_head=head,
        metrics=_metrics(result),
        admission=_admission(),
        config={"adapter_name": "default", "lora_rank": 2, "lora_alpha": 4, "head_dim": 8},
        licenses={"student": license_file},
        torch=torch,
        safetensors=safetensors,
    )
    (private / "export" / "adapter.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset bytes changed"):
        checkpoint.fresh_reload_proof(
            export=private / "export",
            factory=lambda: _components(torch, transformers, peft),
            samples=_rows()[:2],
            dev_rows=_rows()[2:],
            torch=torch,
            safetensors=safetensors,
            expected_tensor_count=6,
        )


def test_fresh_reload_rejects_live_peft_scaling_mismatch(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    license_file = private / "license"
    license_file.write_text("self-authored test license", encoding="utf-8")
    adapter, head = _components(torch, transformers, peft)
    result = _train(torch, peft, adapter, head)
    checkpoint.export_private_checkpoint(
        work=private,
        adapter=adapter,
        pointer_head=head,
        metrics=_metrics(result),
        admission=_admission(),
        config={"adapter_name": "default", "lora_rank": 2, "lora_alpha": 4, "head_dim": 8},
        licenses={"student": license_file},
        torch=torch,
        safetensors=safetensors,
    )
    with pytest.raises(ValueError, match="scaling does not match"):
        checkpoint.fresh_reload_proof(
            export=private / "export",
            factory=lambda: _components(torch, transformers, peft, alpha=2),
            samples=_rows()[:2],
            dev_rows=_rows()[2:],
            torch=torch,
            safetensors=safetensors,
            expected_tensor_count=6,
        )


def test_reload_moves_batches_to_the_live_model_device_and_rejects_fake_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    _export(private, torch, transformers, peft, safetensors)
    original_to_device = v02_training._to_device
    devices: list[Any] = []

    def recording_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
        devices.append(device)
        return original_to_device(batch, device)

    monkeypatch.setattr(v02_training, "_to_device", recording_to_device)
    with pytest.raises(ValueError, match="no nonzero functional effect"):
        checkpoint.fresh_reload_proof(
            export=private / "export",
            factory=lambda: _no_effect_components(torch, transformers, peft),
            samples=_rows()[:2],
            dev_rows=_rows()[2:],
            torch=torch,
            safetensors=safetensors,
            expected_tensor_count=6,
        )
    assert devices and all(device.type == "cpu" for device in devices)


def _no_effect_components(torch: Any, transformers: Any, peft: Any) -> tuple[Any, Any]:
    adapter, head = _components(torch, transformers, peft)
    adapter.disable_adapter = lambda: nullcontext()
    return adapter, head


def test_reload_rejects_non_bf16_head_and_atomic_safetensor_failures(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    _export(private, torch, transformers, peft, safetensors)

    def fp32_head() -> tuple[Any, Any]:
        adapter, head = _components(torch, transformers, peft)
        return adapter, head.float()

    with pytest.raises(ValueError, match="BF16 frozen base and pointer head"):
        checkpoint.fresh_reload_proof(
            export=private / "export",
            factory=fp32_head,
            samples=_rows()[:2],
            dev_rows=_rows()[2:],
            torch=torch,
            safetensors=safetensors,
            expected_tensor_count=6,
        )

    class InterruptedSave:
        @staticmethod
        def save_file(_tensors: dict[str, Any], destination: str) -> None:
            Path(destination).write_bytes(b"partial")
            raise OSError("interrupted")

    target = private / "interrupted.safetensors"
    with pytest.raises(OSError, match="interrupted"):
        checkpoint._create_safetensors(target, {"x": torch.ones(1)}, InterruptedSave)
    assert not target.exists()
    checkpoint._create_safetensors(target, {"x": torch.ones(1)}, safetensors)
    original = target.read_bytes()
    with pytest.raises(FileExistsError, match="immutable output"):
        checkpoint._create_safetensors(target, {"x": torch.zeros(1)}, safetensors)
    assert target.read_bytes() == original


def test_export_rejects_unoptimized_extra_trainable_and_nonfinite_tensors() -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    adapter, head = _components(torch, transformers, peft)
    checkpoint._learned_tensors(adapter, head)
    adapter.register_parameter("unexpected_trainable", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="trainable parameters"):
        checkpoint._learned_tensors(adapter, head)
    del adapter.unexpected_trainable
    with torch.no_grad():
        head.query.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        checkpoint._learned_tensors(adapter, head)
    metrics = _metrics(
        {"optimizer_steps": 1, "best_epoch": 1, "internal_dev_stratified_macro_accuracy": 0.0}
    )
    for key in (
        "epochs_completed",
        "optimizer_steps",
        "train_rows",
        "internal_dev_rows",
        "best_epoch",
    ):
        with pytest.raises(ValueError, match="positive optimization"):
            checkpoint._metrics_result({**metrics, key: 0}, _hash("e"))
    with pytest.raises(ValueError, match="positive optimization"):
        checkpoint._metrics_result({**metrics, "best_epoch": 3}, _hash("e"))


def test_fresh_bf16_base_preserves_actual_peft_fp32_adapter_inventory(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    safetensors: Any = pytest.importorskip("safetensors.torch")

    def factory() -> tuple[Any, Any]:
        adapter, head = _components(torch, transformers, peft)
        for name, parameter in adapter.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                parameter.data = parameter.data.to(dtype=torch.float32)
        return adapter, head

    adapter, head = factory()
    result = _train(torch, peft, adapter, head)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    license_file = private / "license"
    license_file.write_text("self-authored test license", encoding="utf-8")
    checkpoint.export_private_checkpoint(
        work=private,
        adapter=adapter,
        pointer_head=head,
        metrics=_metrics(result),
        admission=_admission(),
        config={"adapter_name": "default", "lora_rank": 2, "lora_alpha": 4, "head_dim": 8},
        licenses={"student": license_file},
        torch=torch,
        safetensors=safetensors,
    )
    receipt = checkpoint.fresh_reload_proof(
        export=private / "export",
        factory=factory,
        samples=_rows()[:2],
        dev_rows=_rows()[2:],
        torch=torch,
        safetensors=safetensors,
        expected_tensor_count=6,
    )
    assert receipt["precision"] == "bf16"
    assert receipt["functional_effect_verified"] is True
    assert receipt["matched_tensor_count"] == 6
