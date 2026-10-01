"""Shared, local-only optimizer and safe operational-resume primitives.

The module has no import-time ML dependency.  Callers supply the already
constructed adapter and readout head, so the production CUDA entry point and
the tiny CPU Qwen/PEFT proof exercise precisely this loop.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from saracura.serialization import canonical_json_bytes

RESUME_SCHEMA = "v02-private-resume-state.v1"
_SHA256_HEX = frozenset("0123456789abcdef")
_RESUME_KEYS = frozenset(
    {
        "schema_version",
        "bindings_sha256",
        "private_identity",
        "epoch",
        "next_batch_index",
        "optimizer_steps",
        "adapter_state",
        "head_state",
        "optimizer_state",
        "scheduler_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
        "best_adapter_state",
        "best_head_state",
        "best_epoch",
        "best_score",
        "stale_epochs",
        "initial_trainable_sha256",
        "base_parameter_sha256",
        "duration_seconds",
        "peak_gpu_bytes",
    }
)


def _cpu_clone_state(module: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _tensor_digest(torch: Any, tensors: Iterable[tuple[str, Any]]) -> str:
    """Digest a named tensor inventory without losing dtype or raw-byte identity."""
    inventory: list[dict[str, Any]] = []
    names: set[str] = set()
    for name, value in sorted(tensors, key=lambda item: item[0]):
        if not isinstance(name, str) or name in names:
            raise ValueError("tensor inventory has an ambiguous name")
        names.add(name)
        tensor = value.detach().cpu().contiguous()
        raw_value = tensor.view(torch.uint8).numpy().tobytes()
        inventory.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "value_sha256": hashlib.sha256(raw_value).hexdigest(),
            }
        )
    return hashlib.sha256(canonical_json_bytes(cast(JsonValue, inventory))).hexdigest()


def _parameter_digest(torch: Any, adapter: Any, pointer_head: Any, *, trainable: bool) -> str:
    """Bind adapter and head parameters in one unambiguous canonical inventory."""
    tensors = (
        (f"adapter.{name}", value)
        for module_name, module in (("adapter", adapter), ("pointer_head", pointer_head))
        for name, value in module.named_parameters()
        if value.requires_grad is trainable
    )
    return _tensor_digest(torch, tensors)


def _tree_to_list(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tree_to_list(item) for item in value]
    if isinstance(value, list):
        return [_tree_to_list(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_list(item) for key, item in value.items()}
    return value


def _tree_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tree_to_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _tree_to_tuple(item) for key, item in value.items()}
    return value


def _numpy_rng_state() -> dict[str, Any]:
    numpy = importlib.import_module("numpy")
    name, values, position, has_gauss, cached_gaussian = numpy.random.get_state()
    return {
        "name": str(name),
        "values": [int(item) for item in values.tolist()],
        "dtype": str(values.dtype),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(value: Any) -> None:
    numpy = importlib.import_module("numpy")
    if not isinstance(value, dict) or set(value) != {
        "name",
        "values",
        "dtype",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("resume NumPy RNG state is not closed")
    if (
        not isinstance(value["name"], str)
        or not isinstance(value["dtype"], str)
        or not isinstance(value["values"], list)
        or any(type(item) is not int for item in value["values"])
    ):
        raise ValueError("resume NumPy RNG state is invalid")
    values = numpy.array(value["values"], dtype=value["dtype"])
    numpy.random.set_state(
        (
            value["name"],
            values,
            int(value["position"]),
            int(value["has_gauss"]),
            float(value["cached_gaussian"]),
        )
    )


def _require_tensor_primitives(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        return
    if isinstance(value, list):
        for item in value:
            _require_tensor_primitives(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError("resume state contains a non-primitive key")
            _require_tensor_primitives(item)
        return
    raise ValueError("resume state contains a non-primitive value")


def write_resume_state(torch: Any, path: Path, state: dict[str, Any]) -> None:
    """Atomically replace a trusted operational state at an optimizer boundary."""
    validate_resume_state(state)
    _require_tensor_primitives(state)
    if not path.is_absolute() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("resume state path must be an absolute non-symlink file")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".resume-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_resume_state(torch: Any, path: Path) -> dict[str, Any]:
    """Load only tensor/primitive state; never unpickle an arbitrary object."""
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("resume state must be an absolute regular file")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("resume state root is invalid")
    validate_resume_state(value)
    _require_tensor_primitives(value)
    return value


def validate_resume_state(state: dict[str, Any]) -> None:
    if set(state) != _RESUME_KEYS or state.get("schema_version") != RESUME_SCHEMA:
        raise ValueError("resume state is not closed")
    integer_keys = ("epoch", "next_batch_index", "optimizer_steps", "best_epoch", "stale_epochs")
    if any(type(state[key]) is not int or state[key] < 0 for key in integer_keys):
        raise ValueError("resume cursor is invalid")
    if not isinstance(state["best_score"], float) or not math.isfinite(state["best_score"]):
        raise ValueError("resume best score is invalid")
    if not isinstance(state["duration_seconds"], float) or state["duration_seconds"] < 0:
        raise ValueError("resume duration is invalid")
    if type(state["peak_gpu_bytes"]) is not int or state["peak_gpu_bytes"] < 0:
        raise ValueError("resume peak memory is invalid")
    if not all(isinstance(state[key], str) for key in ("bindings_sha256", "private_identity")):
        raise ValueError("resume identity is invalid")
    if any(
        not isinstance(state[key], str)
        or len(state[key]) != 64
        or any(character not in _SHA256_HEX for character in state[key])
        for key in ("initial_trainable_sha256", "base_parameter_sha256")
    ):
        raise ValueError("resume tensor binding is invalid")


def _macro_accuracy(
    torch: Any,
    adapter: Any,
    head: Any,
    batches: Iterable[dict[str, Any]],
    to_device: Callable[[dict[str, Any], Any], dict[str, Any]],
    score_batch: Callable[[Any, Any, dict[str, Any]], Any],
    device: Any,
) -> float:
    correct: dict[str, list[int]] = {}
    adapter.eval()
    head.eval()
    with torch.no_grad():
        for batch in batches:
            scores = score_batch(adapter, head, to_device(batch, device))
            for predicted, target, stratum in zip(
                scores.argmax(dim=1).cpu().tolist(),
                batch["targets"].tolist(),
                batch["strata"],
                strict=True,
            ):
                bucket = correct.setdefault(stratum, [0, 0])
                bucket[0] += int(predicted == target)
                bucket[1] += 1
    return sum(hits / count for hits, count in correct.values()) / len(correct)


def run_optimization(
    *,
    torch: Any,
    adapter: Any,
    pointer_head: Any,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    config: dict[str, Any],
    device: Any,
    token_batches: Callable[[Any, list[dict[str, Any]], int], Iterable[dict[str, Any]]],
    to_device: Callable[[dict[str, Any], Any], dict[str, Any]],
    score_batch: Callable[[Any, Any, dict[str, Any]], Any],
    adapter_state: Callable[[Any], dict[str, Any]],
    load_adapter_state: Callable[[Any, dict[str, Any]], None],
    resume_state: dict[str, Any] | None = None,
    on_boundary: Callable[[dict[str, Any]], None] | None = None,
    stop_after_steps: int | None = None,
    bindings_sha256: str = "test-only-bindings",
    private_identity: str = "test-only-private-identity",
) -> dict[str, Any]:
    """Run the fixed AdamW/warmup-cosine objective from a safe boundary state."""
    if not train_rows or not dev_rows:
        raise ValueError("training and internal_dev rows are required")
    microbatch = min(4, len(train_rows))
    accumulation = max(1, int(config["effective_batch_size"]) // microbatch)
    batches_per_epoch = math.ceil(len(train_rows) / microbatch)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * int(config["max_epochs"])
    warmup_steps = int(total_steps * float(config["warmup_fraction"]))
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    trainable.extend(
        parameter for parameter in pointer_head.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise ValueError("optimizer received no trainable LoRA/head tensors")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["peak_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    initial_digest = _parameter_digest(torch, adapter, pointer_head, trainable=True)
    base_digest = _parameter_digest(torch, adapter, pointer_head, trainable=False)
    started = time.monotonic()
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if getattr(device, "type", "cpu") == "cuda"
        else 0
    )
    if resume_state is None:
        epoch, next_batch_index, optimizer_steps, best_epoch, best_score, stale_epochs = (
            1,
            0,
            0,
            0,
            -1.0,
            0,
        )
        best_adapter, best_head = adapter_state(adapter), _cpu_clone_state(pointer_head)
        previous_duration = 0.0
    else:
        validate_resume_state(resume_state)
        if resume_state["initial_trainable_sha256"] != initial_digest:
            raise ValueError("resume initial trainable tensor binding does not match")
        if resume_state["base_parameter_sha256"] != base_digest:
            raise ValueError("resume frozen base tensor binding does not match")
        load_adapter_state(adapter, resume_state["adapter_state"])
        pointer_head.load_state_dict(resume_state["head_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        random.setstate(_tree_to_tuple(resume_state["python_rng_state"]))
        _restore_numpy_rng_state(resume_state["numpy_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        if getattr(device, "type", "cpu") == "cuda":
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
        epoch, next_batch_index, optimizer_steps = (
            resume_state["epoch"],
            resume_state["next_batch_index"],
            resume_state["optimizer_steps"],
        )
        best_epoch, best_score, stale_epochs = (
            resume_state["best_epoch"],
            resume_state["best_score"],
            resume_state["stale_epochs"],
        )
        best_adapter, best_head = (
            resume_state["best_adapter_state"],
            resume_state["best_head_state"],
        )
        previous_duration = resume_state["duration_seconds"]

    def boundary(next_epoch: int, next_cursor: int) -> dict[str, Any]:
        nonlocal peak_gpu_bytes
        if getattr(device, "type", "cpu") == "cuda":
            peak_gpu_bytes = max(peak_gpu_bytes, int(torch.cuda.max_memory_allocated(device)))
            cuda_rng_states = [
                item.detach().cpu().clone() for item in torch.cuda.get_rng_state_all()
            ]
        else:
            cuda_rng_states = []
        state = {
            "schema_version": RESUME_SCHEMA,
            "bindings_sha256": bindings_sha256,
            "private_identity": private_identity,
            "epoch": next_epoch,
            "next_batch_index": next_cursor,
            "optimizer_steps": optimizer_steps,
            "adapter_state": adapter_state(adapter),
            "head_state": _cpu_clone_state(pointer_head),
            "optimizer_state": _tree_to_list(optimizer.state_dict()),
            "scheduler_state": _tree_to_list(scheduler.state_dict()),
            "python_rng_state": _tree_to_list(random.getstate()),
            "numpy_rng_state": _numpy_rng_state(),
            "torch_rng_state": torch.get_rng_state().detach().cpu().clone(),
            "cuda_rng_states": cuda_rng_states,
            "best_adapter_state": best_adapter,
            "best_head_state": best_head,
            "best_epoch": best_epoch,
            "best_score": float(best_score),
            "stale_epochs": stale_epochs,
            "initial_trainable_sha256": initial_digest,
            "base_parameter_sha256": base_digest,
            "duration_seconds": float(previous_duration + time.monotonic() - started),
            "peak_gpu_bytes": peak_gpu_bytes,
        }
        validate_resume_state(state)
        return state

    initial_state = boundary(epoch, next_batch_index)
    if on_boundary is not None:
        on_boundary(initial_state)
    if stop_after_steps == 0:
        return {"interrupted": True, "resume_state": initial_state, "initial_state": initial_state}

    def completed_result() -> dict[str, Any]:
        load_adapter_state(adapter, best_adapter)
        pointer_head.load_state_dict(best_head)
        return {
            "interrupted": False,
            "best_epoch": best_epoch,
            "internal_dev_stratified_macro_accuracy": best_score,
            "optimizer_steps": optimizer_steps,
            "early_stopped": stale_epochs >= int(config["patience"]),
            "resume_state": boundary(epoch, next_batch_index),
            "initial_state": initial_state,
        }

    # A boundary persisted just after the terminal epoch is already a completed
    # run.  Do not start a new epoch merely because it is restored in a fresh
    # process; select the immutable earliest best snapshot and return it.
    if epoch > int(config["max_epochs"]) or stale_epochs >= int(config["patience"]):
        return completed_result()

    while epoch <= int(config["max_epochs"]):
        batches = list(token_batches(torch, train_rows, microbatch))
        if not 0 <= next_batch_index < len(batches) or next_batch_index % accumulation:
            raise ValueError("resume cursor is not an optimizer boundary")
        adapter.train()
        pointer_head.train()
        optimizer.zero_grad(set_to_none=True)
        for window_start in range(next_batch_index, len(batches), accumulation):
            window = batches[window_start : window_start + accumulation]
            window_examples = sum(len(batch["strata"]) for batch in window)
            for batch in window:
                scores = score_batch(adapter, pointer_head, to_device(batch, device))
                loss = torch.nn.functional.cross_entropy(scores, batch["targets"].to(device))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("optimizer produced a non-finite causal loss")
                (loss * (len(batch["strata"]) / window_examples)).backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(config["gradient_clipping"]), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            next_batch_index = window_start + len(window)
            if next_batch_index < len(batches):
                state = boundary(epoch, next_batch_index)
                if on_boundary is not None:
                    on_boundary(state)
                if stop_after_steps is not None and optimizer_steps >= stop_after_steps:
                    return {
                        "interrupted": True,
                        "resume_state": state,
                        "initial_state": initial_state,
                    }
        metric = _macro_accuracy(
            torch,
            adapter,
            pointer_head,
            token_batches(torch, dev_rows, microbatch),
            to_device,
            score_batch,
            device,
        )
        if metric > best_score:
            best_score, best_epoch, stale_epochs = metric, epoch, 0
            best_adapter, best_head = adapter_state(adapter), _cpu_clone_state(pointer_head)
        else:
            stale_epochs += 1
        epoch += 1
        next_batch_index = 0
        state = boundary(epoch, next_batch_index)
        if on_boundary is not None:
            on_boundary(state)
        if stop_after_steps is not None and optimizer_steps >= stop_after_steps:
            return {"interrupted": True, "resume_state": state, "initial_state": initial_state}
        if stale_epochs >= int(config["patience"]):
            break
    return completed_result()
