from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import v02_training
from benchmarks.validate_manifests import validate_routed_manifest


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=2560,
        num_hidden_layers=32,
        intermediate_size=9216,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        vocab_size=248320,
        max_position_embeddings=262144,
        layer_types=[
            "full_attention" if index in range(3, 32, 4) else "linear_attention"
            for index in range(32)
        ],
    )


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=_text_config(),
    )


def test_manifest_and_optional_extra_are_closed_and_additive() -> None:
    manifest = v02_training.load_manifest()
    assert manifest["renderer"]["candidate_renderer_config"]["lora_targets"] == "qwen35_4b_text_248"
    validate_routed_manifest(v02_training.MANIFEST_PATH)
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    assert extras["v02-training"] == [
        "bitsandbytes>=0.45,<1",
        "peft>=0.14,<1",
        "safetensors>=0.5,<1",
        "torch>=2.5,<3",
        "transformers==5.16.0",
    ]
    assert extras["local-minilm"][4:] == ["torch>=2.5,<3", "transformers>=4.47,<5"]


def test_semantic_qwen_config_and_exact_inventory() -> None:
    config = _model_config()
    copied_text_config = _text_config()
    assert v02_training.qwen35_text_config(config) is config.text_config
    assert v02_training._semantic_text_config_matches(config.text_config, copied_text_config)
    copied_text_config.hidden_size = 123
    assert not v02_training._semantic_text_config_matches(config.text_config, copied_text_config)
    config.text_config.layer_types[3] = "linear_attention"
    with pytest.raises(ValueError, match="pinned"):
        v02_training.qwen35_text_config(config)

    targets = v02_training.qwen35_lora_targets()
    shapes = v02_training._expected_adapter_shapes(8)
    assert len(targets) == 248
    assert len(shapes) == 496
    assert shapes["base_model.model.layers.0.linear_attn.out_proj.lora_A.default.weight"] == (
        8,
        4096,
    )
    assert shapes["base_model.model.layers.31.mlp.down_proj.lora_B.default.weight"] == (2560, 8)
    assert v02_training.candidate_config("c1-r8")["lora_targets"] == targets


def test_causal_row_is_single_forward_with_int64_bool_and_ragged_mask() -> None:
    torch = pytest.importorskip("torch")
    calls = 0

    class Adapter(torch.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, input_ids: object, attention_mask: object) -> object:
            nonlocal calls
            calls += 1
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(attention_mask, torch.Tensor)
            return SimpleNamespace(
                last_hidden_state=torch.nn.functional.one_hot(input_ids % 4, num_classes=4).to(
                    torch.float32
                )
            )

    class Head(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
            self.key = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)

        def forward(self, state: object, options: object) -> object:
            assert isinstance(state, torch.Tensor)
            assert isinstance(options, torch.Tensor)
            return (self.query(state).unsqueeze(1) * self.key(options)).sum(-1)

    rows = [
        {
            "split": "train",
            "input_ids": [1, 2, 3, 4, 5, 6],
            "end_option_indices": [2, 4],
            "decide_index": 5,
            "target": 1,
            "stratum": "pt_br",
        },
        {
            "split": "internal_dev",
            "input_ids": [7, 8, 9, 10, 11, 12, 13],
            "end_option_indices": [2, 3, 5],
            "decide_index": 6,
            "target": 2,
            "stratum": "english",
        },
    ]
    batch = next(iter(v02_training._token_batches(torch, rows, 2)))
    assert batch["input_ids"].dtype == torch.int64
    assert batch["end_option_indices"].dtype == torch.int64
    assert batch["targets"].dtype == torch.int64
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["option_present"].dtype == torch.bool
    assert batch["option_present"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["end_option_indices"].tolist() == [[2, 4, 0], [2, 3, 5]]
    scores = v02_training._score_token_batch(Adapter(), Head(), batch)
    assert calls == 1
    assert scores.dtype == torch.bfloat16
    assert torch.isneginf(scores[0, 2])
    assert batch["targets"].tolist() == [1, 2]


def test_prepared_causal_rows_reject_overflow_and_invalid_index(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    valid = {
        "records": [
            {
                "split": "train",
                "input_ids": [1, 2, 3, 4],
                "end_option_indices": [1, 2],
                "decide_index": 3,
                "target": 0,
                "stratum": "a",
            },
            {
                "split": "internal_dev",
                "input_ids": [1, 2, 3, 4],
                "end_option_indices": [1, 2],
                "decide_index": 3,
                "target": 1,
                "stratum": "b",
            },
        ]
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert v02_training._prepared_examples(path) == valid["records"]
    valid["records"][0]["input_ids"] = list(range(513))
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="causal row"):
        v02_training._prepared_examples(path)


def _admission() -> dict[str, object]:
    values: dict[str, object] = {key: "a" * 64 for key in v02_training._ADMISSION_FIELDS}
    values.update(
        {
            "schema_version": "v02-private-training-admission.v1",
            "artifact_id": "private-c1-r8-v1",
            "private_identity": "private-c1-r8-v1",
            "corpus_source_revision": "source-r1",
            "trainer_source_revision": "trainer-r1",
            "base_id": "Qwen/Qwen3.5-4B-Base",
            "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
            "seed": 20260929,
            "training_authorized": True,
            "private_only": True,
            "publication_authorized": False,
        }
    )
    return values


def test_admission_live_verifies_original_receipt_before_any_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission_path, receipt_path, plan_path, events_path = (
        tmp_path / "admission.json",
        tmp_path / "receipt.json",
        tmp_path / "plan.json",
        tmp_path / "events.json",
    )
    receipt_path.write_text("receipt", encoding="utf-8")
    admission = _admission()
    admission["aggregate_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    plan_path.write_text("{}", encoding="utf-8")
    events_path.write_text('{"events":[]}', encoding="utf-8")
    seen: list[Path] = []

    def verifier(path: Path, **kwargs: object) -> dict[str, object]:
        seen.append(path)
        assert kwargs["expected_lane"] == "training"
        return {"status": "READY"}

    proof = v02_training.verify_private_admission(
        admission_path, receipt_path, plan_path, events_path, verify_receipt=verifier
    )
    assert isinstance(proof, v02_training.AdmissionProof)
    assert seen == [receipt_path]
    monkeypatch.setattr(
        v02_training, "_training_runtime", lambda: (_ for _ in ()).throw(AssertionError("loaded"))
    )
    with pytest.raises(ValueError, match="live-verified"):
        v02_training.build_qlora_components(tmp_path, "c1-r8")


def test_tiny_cpu_proof_export_reload_and_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    proof = v02_training.tiny_causal_cpu_proof(torch)
    assert proof == {
        "test_only": True,
        "base_unchanged": True,
        "best_state_cpu": True,
        "finite_updates": True,
    }

    class Adapter(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 2)))

        def save_pretrained(self, destination: str, *, safe_serialization: bool) -> None:
            assert safe_serialization
            folder = Path(destination)
            folder.mkdir()
            key = "base_model.model.layers.0.mlp.up_proj.lora_A.default.weight"
            safetensors_torch.save_file(
                {key: self.weight.detach()},
                str(folder / "adapter_model.safetensors"),
            )
            (folder / "adapter_config.json").write_text("{}", encoding="utf-8")

    class Head(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Linear(2, 2, bias=False)
            self.key = torch.nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(
        v02_training, "_training_runtime", lambda: (torch, None, None, safetensors_torch)
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "artifact"
    descriptor = v02_training.export_candidate_artifact(
        Adapter(), Head(), output, v02_training.candidate_config("c1-r8")
    )
    v02_training.verify_candidate_artifact(descriptor, output)
    reloaded = safetensors_torch.load_file(str(output / "pointer_head.safetensors"))
    assert set(reloaded) == {"pointer_head.query.weight", "pointer_head.key.weight"}
    with pytest.raises(FileExistsError):
        v02_training.export_candidate_artifact(
            Adapter(), Head(), output, v02_training.candidate_config("c1-r8")
        )
