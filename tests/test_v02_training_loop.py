from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks import v02_training
from benchmarks.v02_training_loop import load_resume_state, run_optimization, write_resume_state


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "split": "train",
            "input_ids": [1, 2, 3, 4, 5],
            "end_option_indices": [1, 2, 3],
            "decide_index": 4,
            "target": index % 3,
            "stratum": "pt" if index % 2 else "en",
        }
        for index in range(5)
    ] + [
        {
            "split": "internal_dev",
            "input_ids": [6, 7, 8, 9, 10],
            "end_option_indices": [1, 2, 3],
            "decide_index": 4,
            "target": 2,
            "stratum": "pt" if index % 2 else "en",
        }
        for index in range(4)
    ]


def _config(*, max_epochs: int = 3, patience: int = 9, lr: str = "0.02") -> dict[str, Any]:
    return {
        "effective_batch_size": 4,
        "max_epochs": max_epochs,
        "patience": patience,
        "peak_learning_rate": lr,
        "weight_decay": 0.0,
        "warmup_fraction": 0.25,
        "gradient_clipping": 1.0,
    }


def _components(
    torch: Any, transformers: Any, peft: Any, *, hybrid: bool = False
) -> tuple[Any, Any, dict[str, Any]]:
    v02_training.configure_determinism(torch, 20260929)
    text_config = transformers.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2 if hybrid else 1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "full_attention"] if hybrid else ["full_attention"],
        pad_token_id=0,
    )
    base = transformers.Qwen3_5ForCausalLM(text_config).model
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
    )

    class TinyHead(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Linear(16, 8, bias=False)
            self.key = torch.nn.Linear(16, 8, bias=False)

        def forward(self, state: Any, options: Any) -> Any:
            return (self.query(state).unsqueeze(1) * self.key(options)).sum(dim=-1) / (8**0.5)

    head = TinyHead()
    frozen = {
        name: value.detach().clone()
        for name, value in adapter.named_parameters()
        if not value.requires_grad
    }
    return adapter, head, frozen


def _run(
    torch: Any, peft: Any, adapter: Any, head: Any, config: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    rows = _rows()
    return run_optimization(
        torch=torch,
        adapter=adapter,
        pointer_head=head,
        train_rows=[row for row in rows if row["split"] == "train"],
        dev_rows=[row for row in rows if row["split"] == "internal_dev"],
        config=config,
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
        **kwargs,
    )


def _learned_state(adapter: Any, head: Any) -> dict[str, Any]:
    return {
        **{
            f"adapter:{key}": value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
            if "lora_" in key
        },
        **{f"head:{key}": value.detach().cpu().clone() for key, value in head.state_dict().items()},
    }


def test_nonfinite_gradient_rejects_before_optimizer_changes_weights() -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    adapter, head, _frozen = _components(torch, transformers, peft)
    before = _learned_state(adapter, head)
    handle = head.query.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            _run(torch, peft, adapter, head, _config())
    finally:
        handle.remove()
    _assert_tensor_state_equal(before, _learned_state(adapter, head), torch)


def test_actual_hybrid_qwen_linear_attention_backward_uses_the_same_loop() -> None:
    torch: Any = pytest.importorskip("torch")
    transformers: Any = pytest.importorskip("transformers")
    peft: Any = pytest.importorskip("peft")
    adapter, head, frozen = _components(torch, transformers, peft, hybrid=True)
    result = _run(torch, peft, adapter, head, _config(max_epochs=2))
    assert result["optimizer_steps"] == 4
    assert result["interrupted"] is False
    assert all(
        bool(torch.isfinite(value).all()) for value in _learned_state(adapter, head).values()
    )
    assert all(
        torch.equal(parameter, frozen[name])
        for name, parameter in adapter.named_parameters()
        if name in frozen
    )


def _assert_tensor_state_equal(left: dict[str, Any], right: dict[str, Any], torch: Any) -> None:
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_actual_qwen_peft_cpu_loop_updates_only_lora_head_and_uses_partial_window() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    adapter, head, frozen = _components(torch, transformers, peft)
    initial = _learned_state(adapter, head)
    result = _run(torch, peft, adapter, head, _config())

    assert result["interrupted"] is False
    assert result["optimizer_steps"] == 6  # 5 rows => 4 + 1 partial accumulation windows, 3 epochs.
    assert result["best_epoch"] >= 1
    assert all(
        value.device.type == "cpu"
        for value in result["resume_state"]["best_adapter_state"].values()
    )
    assert all(
        value.device.type == "cpu" for value in result["resume_state"]["best_head_state"].values()
    )
    assert any(
        not torch.equal(value, initial[key])
        for key, value in _learned_state(adapter, head).items()
        if key.startswith("adapter:")
    )
    assert any(
        not torch.equal(value, initial[key])
        for key, value in _learned_state(adapter, head).items()
        if key.startswith("head:")
    )
    assert all(
        torch.equal(value, frozen[name])
        for name, value in adapter.named_parameters()
        if not value.requires_grad
    )


def test_tensor_only_resume_is_bitwise_at_step_zero_mid_epoch_and_epoch_boundary(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = _config(max_epochs=3)
    baseline_adapter, baseline_head, _ = _components(torch, transformers, peft)
    baseline = _run(torch, peft, baseline_adapter, baseline_head, config)
    baseline_state = _learned_state(baseline_adapter, baseline_head)

    for stop_after in (0, 1, 2):
        adapter, head, _ = _components(torch, transformers, peft)
        interrupted = _run(torch, peft, adapter, head, config, stop_after_steps=stop_after)
        assert interrupted["interrupted"] is True
        path = tmp_path / f"resume-{stop_after}.pt"
        write_resume_state(torch, path.resolve(), interrupted["resume_state"])
        restored = load_resume_state(torch, path.resolve())
        resumed_adapter, resumed_head, _ = _components(torch, transformers, peft)
        resumed = _run(torch, peft, resumed_adapter, resumed_head, config, resume_state=restored)
        assert resumed["interrupted"] is False
        _assert_tensor_state_equal(
            baseline_state, _learned_state(resumed_adapter, resumed_head), torch
        )
        assert resumed["best_epoch"] == baseline["best_epoch"]
        assert (
            resumed["internal_dev_stratified_macro_accuracy"]
            == baseline["internal_dev_stratified_macro_accuracy"]
        )


def test_actual_metric_early_stops_and_best_snapshot_is_independent() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    adapter, head, _ = _components(torch, transformers, peft)
    result = _run(torch, peft, adapter, head, _config(max_epochs=5, patience=1, lr="0.0"))
    assert result["early_stopped"] is True
    assert result["best_epoch"] == 1
    snapshot = result["resume_state"]["best_head_state"]
    current = head.state_dict()["query.weight"]
    snapshot["query.weight"].add_(1)
    assert not torch.equal(snapshot["query.weight"], current)


def test_terminal_early_stop_resume_keeps_steps_best_and_learned_tensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = _config(max_epochs=5, patience=1, lr="0.0")

    baseline_adapter, baseline_head, _ = _components(torch, transformers, peft)
    baseline = _run(torch, peft, baseline_adapter, baseline_head, config)
    baseline_state = _learned_state(baseline_adapter, baseline_head)

    interrupted_adapter, interrupted_head, _ = _components(torch, transformers, peft)
    interrupted = _run(
        torch,
        peft,
        interrupted_adapter,
        interrupted_head,
        config,
        stop_after_steps=baseline["optimizer_steps"],
    )
    assert interrupted["interrupted"] is True
    path = tmp_path / "terminal-early-stop.pt"
    write_resume_state(torch, path.resolve(), interrupted["resume_state"])

    resumed_adapter, resumed_head, _ = _components(torch, transformers, peft)
    resumed = _run(
        torch,
        peft,
        resumed_adapter,
        resumed_head,
        config,
        resume_state=load_resume_state(torch, path.resolve()),
    )
    assert resumed["interrupted"] is False
    assert resumed["optimizer_steps"] == baseline["optimizer_steps"]
    assert resumed["best_epoch"] == baseline["best_epoch"]
    assert (
        resumed["internal_dev_stratified_macro_accuracy"]
        == (baseline["internal_dev_stratified_macro_accuracy"])
    )
    _assert_tensor_state_equal(baseline_state, _learned_state(resumed_adapter, resumed_head), torch)


def test_completed_max_epoch_resume_keeps_steps_best_and_learned_tensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = _config(max_epochs=2)

    baseline_adapter, baseline_head, _ = _components(torch, transformers, peft)
    baseline = _run(torch, peft, baseline_adapter, baseline_head, config)
    baseline_state = _learned_state(baseline_adapter, baseline_head)

    interrupted_adapter, interrupted_head, _ = _components(torch, transformers, peft)
    interrupted = _run(
        torch,
        peft,
        interrupted_adapter,
        interrupted_head,
        config,
        stop_after_steps=baseline["optimizer_steps"],
    )
    path = tmp_path / "completed-max-epoch.pt"
    write_resume_state(torch, path.resolve(), interrupted["resume_state"])

    resumed_adapter, resumed_head, _ = _components(torch, transformers, peft)
    resumed = _run(
        torch,
        peft,
        resumed_adapter,
        resumed_head,
        config,
        resume_state=load_resume_state(torch, path.resolve()),
    )
    assert resumed["interrupted"] is False
    assert resumed["optimizer_steps"] == baseline["optimizer_steps"]
    assert resumed["best_epoch"] == baseline["best_epoch"]
    _assert_tensor_state_equal(baseline_state, _learned_state(resumed_adapter, resumed_head), torch)


def test_resume_rejects_changed_initial_or_frozen_base_tensor() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = _config()
    adapter, head, _ = _components(torch, transformers, peft)
    state = _run(torch, peft, adapter, head, config, stop_after_steps=0)["resume_state"]

    tampered_state = dict(state)
    tampered_state["initial_trainable_sha256"] = "0" * 64
    matching_adapter, matching_head, _ = _components(torch, transformers, peft)
    with pytest.raises(ValueError, match="initial trainable tensor binding"):
        _run(
            torch,
            peft,
            matching_adapter,
            matching_head,
            config,
            resume_state=tampered_state,
        )

    changed_initial_adapter, changed_initial_head, _ = _components(torch, transformers, peft)
    with torch.no_grad():
        changed_initial_head.query.weight.add_(1)
    with pytest.raises(ValueError, match="initial trainable tensor binding"):
        _run(
            torch,
            peft,
            changed_initial_adapter,
            changed_initial_head,
            config,
            resume_state=state,
        )

    changed_base_adapter, changed_base_head, _ = _components(torch, transformers, peft)
    with torch.no_grad():
        next(
            parameter
            for parameter in changed_base_adapter.parameters()
            if not parameter.requires_grad
        ).add_(1)
    with pytest.raises(ValueError, match="frozen base tensor binding"):
        _run(
            torch,
            peft,
            changed_base_adapter,
            changed_base_head,
            config,
            resume_state=state,
        )
