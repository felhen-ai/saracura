"""Immutable private checkpoint export and fresh-reload proof.

This is deliberately a library, rather than a command line entry point.  The
private workflow owns admission and CLI wiring; it supplies the already
verified admission/result inputs to these create-only primitives.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from benchmarks import v02_training
from benchmarks.v02_private_training import (
    _publish_directory_exclusively as _publish_directory_exclusively,
)
from saracura.contracts import Answer, DecisionResponse, ModelReference, Usage
from saracura.serialization import canonical_json_bytes

HEX64 = frozenset("0123456789abcdef")
TRAINING_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "admission_sha256",
        "private_identity",
        "seed",
        "epochs_completed",
        "optimizer_steps",
        "train_rows",
        "internal_dev_rows",
        "best_epoch",
        "best_internal_dev_macro_accuracy",
        "finite_updates",
        "base_unchanged",
        "initial_trainable_sha256",
        "exported_trainable_sha256",
        "duration_seconds",
        "peak_gpu_bytes",
    }
)
EXPORT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "admission_sha256",
        "private_identity",
        "base_id",
        "base_revision",
        "base_inventory_sha256",
        "tokenizer_inventory_sha256",
        "renderer_sha256",
        "candidate_config_sha256",
        "tensor_inventory_sha256",
        "files",
        "training_result_sha256",
    }
)
RELOAD_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "admission_sha256",
        "exported_trainable_sha256",
        "base_inventory_sha256",
        "tensor_inventory_sha256",
        "precision",
        "lora_merged",
        "sample_identity_digest",
        "samples",
        "finite_scores",
        "typed_contract_valid",
        "matched_tensor_count",
        "loaded_tensor_inventory_sha256",
        "missing_tensor_keys",
        "unexpected_tensor_keys",
        "loaded_values_match_export",
        "active_adapter_names",
        "adapters_enabled",
        "module_scaling_inventory_sha256",
        "scaling_matches_config",
        "functional_sample_identity_digest",
        "functional_tolerance",
        "functional_max_abs_difference",
        "functional_effect_verified",
        "reload_internal_dev_macro_accuracy",
        "training_best_internal_dev_macro_accuracy",
        "automation_allowed",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "private_identity",
        "admission_sha256",
        "run_receipt_sha256",
        "base_id",
        "base_revision",
        "base_inventory_sha256",
        "tokenizer_inventory_sha256",
        "renderer_sha256",
        "candidate_config_sha256",
        "tensor_inventory_sha256",
        "files",
        "private_only",
        "evaluation_pending",
        "publication_authorized",
        "automation_allowed",
    }
)
RUN_FIELDS = TRAINING_RESULT_FIELDS | frozenset({"reload_receipt_sha256"})
CONFIG_FIELDS = frozenset({"adapter_name", "lora_alpha", "lora_rank", "head_dim"})


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return _sha_bytes(canonical_json_bytes(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise ValueError(f"{label} must be lowercase sha256")
    return value


def _private_parent(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.parent.is_dir():
        raise ValueError("output must be an absolute path below an existing directory")
    if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise ValueError("output parent must be private 0700")
    if any(parent.is_symlink() for parent in (path.parent, *path.parent.parents)):
        raise ValueError("output may not traverse a symlink")


def _create_bytes(path: Path, payload: bytes) -> None:
    _private_parent(path)
    if path.exists():
        raise FileExistsError("immutable output already exists")
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError("immutable output already exists") from exc
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _create_safetensors(path: Path, tensors: dict[str, Any], safetensors: Any) -> None:
    """Publish a completed safetensors file through the same create-only boundary.

    `save_file` must never target the immutable name directly: a killed writer
    would otherwise leave a syntactically plausible partial learned asset.
    """
    _private_parent(path)
    if path.exists():
        raise FileExistsError("immutable output already exists")
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        os.chmod(temporary, 0o600)
        safetensors.save_file(tensors, temporary)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError("immutable output already exists") from exc
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _create_json(path: Path, value: dict[str, Any]) -> None:
    _create_bytes(path, canonical_json_bytes(value) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("artifact must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("artifact is not JSON") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError("artifact JSON is not canonical")
    return value


def _asset_inventory(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative.endswith(".json") and path.name in {
            "export.json",
            "training-result.json",
            "reload-receipt.json",
            "run-receipt.json",
            "checkpoint.json",
        }:
            continue
        if path.is_symlink() or any(part in {"", ".", ".."} for part in Path(relative).parts):
            raise ValueError("asset inventory contains an unsafe path")
        result.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    if not result or result != sorted(result, key=lambda item: item["path"]):
        raise ValueError("asset inventory is empty or unsorted")
    return result


def _tensor_inventory(torch: Any, tensors: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("tensor name is invalid")
        result.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
        )
    if not result or len({item["name"] for item in result}) != len(result):
        raise ValueError("tensor inventory is ambiguous")
    return result


def _is_int(value: Any) -> bool:
    return type(value) is int


def _finite(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _require_closed_config(config: dict[str, Any]) -> dict[str, Any]:
    if set(config) != CONFIG_FIELDS:
        raise ValueError("export config is not closed")
    if (
        not isinstance(config["adapter_name"], str)
        or not config["adapter_name"]
        or not _is_int(config["lora_alpha"])
        or not _is_int(config["lora_rank"])
        or not _is_int(config["head_dim"])
        or min(config["lora_alpha"], config["lora_rank"], config["head_dim"]) <= 0
    ):
        raise ValueError("export config is invalid")
    return config


def _learned_tensors(adapter: Any, pointer_head: Any) -> dict[str, Any]:
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in adapter.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }
    tensors.update(
        {
            f"pointer_head.{name}": value.detach().cpu().contiguous()
            for name, value in pointer_head.state_dict().items()
        }
    )
    expected_adapter = {name for name in tensors if not name.startswith("pointer_head.")}
    actual_adapter = {
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    }
    expected_head = {"query.weight", "key.weight"}
    actual_head = {
        name for name, parameter in pointer_head.named_parameters() if parameter.requires_grad
    }
    if actual_adapter != expected_adapter or actual_head != expected_head:
        raise ValueError("trainable parameters do not match the closed LoRA/head inventory")
    if {
        name.removeprefix("pointer_head.") for name in tensors if name.startswith("pointer_head.")
    } != expected_head:
        raise ValueError("pointer head inventory is not closed")
    if not tensors or any(not value.is_floating_point() for value in tensors.values()):
        raise ValueError("learned export has no floating LoRA/head tensors")
    if any(not bool(value.isfinite().all()) for value in tensors.values()):
        raise ValueError("learned export has non-finite tensors")
    if any("base_model" not in name for name in tensors if not name.startswith("pointer_head.")):
        raise ValueError("adapter export contains an unexpected tensor")
    return tensors


def _metrics_result(metrics: dict[str, Any], exported: str) -> dict[str, Any]:
    result = {
        "schema_version": "v02-private-training-result.v1",
        **metrics,
        "exported_trainable_sha256": exported,
    }
    if set(result) != TRAINING_RESULT_FIELDS:
        raise ValueError("training result metrics are not closed")
    if result["finite_updates"] is not True or result["base_unchanged"] is not True:
        raise ValueError("training result lacks actual update/frozen-base proof")
    if not all(
        isinstance(result[key], str) and result[key] for key in ("artifact_id", "private_identity")
    ):
        raise ValueError("training result identity is invalid")
    for key in ("admission_sha256", "initial_trainable_sha256", "exported_trainable_sha256"):
        _digest(result[key], key)
    integer_fields = (
        "seed",
        "epochs_completed",
        "optimizer_steps",
        "train_rows",
        "internal_dev_rows",
        "best_epoch",
        "peak_gpu_bytes",
    )
    if any(not _is_int(result[key]) or result[key] < 0 for key in integer_fields):
        raise ValueError("training result integer fields are invalid")
    positive_fields = (
        "epochs_completed",
        "optimizer_steps",
        "train_rows",
        "internal_dev_rows",
        "best_epoch",
    )
    if any(result[key] <= 0 for key in positive_fields) or not (
        1 <= result["best_epoch"] <= result["epochs_completed"] <= 5
    ):
        raise ValueError("training result must describe a completed positive optimization")
    if not _finite(result["best_internal_dev_macro_accuracy"], minimum=0.0, maximum=1.0):
        raise ValueError("training result macro accuracy is invalid")
    if not _finite(result["duration_seconds"], minimum=0.0):
        raise ValueError("training result duration is invalid")
    return result


def _publish_complete_directory(staging: Path, target: Path) -> None:
    """Publish fully fsynced private assets without replacing a reservation."""
    for directory in [
        *sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True),
        staging,
    ]:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    _publish_directory_exclusively(staging, target)
    fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def export_private_checkpoint(
    *,
    work: Path,
    adapter: Any,
    pointer_head: Any,
    metrics: dict[str, Any],
    admission: dict[str, Any],
    config: dict[str, Any],
    licenses: dict[str, Path],
    torch: Any,
    safetensors: Any,
) -> dict[str, Any]:
    """Create immutable learned-only assets and their closed export descriptor."""
    _private_parent(work / "training-result.json")
    if work.exists() and not work.is_dir():
        raise ValueError("work must be a directory")
    work.mkdir(mode=0o700, exist_ok=True)
    export = work / "export"
    _private_parent(export)
    if export.exists():
        raise FileExistsError("immutable export already exists")
    required_admission = {
        "artifact_id",
        "private_identity",
        "admission_sha256",
        "base_id",
        "base_revision",
        "base_inventory_sha256",
        "tokenizer_inventory_sha256",
        "renderer_sha256",
        "candidate_config_sha256",
    }
    if (
        not required_admission <= set(admission)
        or admission.get("training_authorized") is not True
        or admission.get("private_only") is not True
        or admission.get("publication_authorized") is not False
    ):
        raise ValueError("admission is not a verified private admission")
    export_config = _require_closed_config(config)
    learned = _learned_tensors(adapter, pointer_head)
    target = export
    export = Path(tempfile.mkdtemp(prefix=".private-export-", dir=work))
    export.chmod(0o700)
    try:
        adapter_tensors = {
            name: value for name, value in learned.items() if not name.startswith("pointer_head.")
        }
        head_tensors = {
            name: value for name, value in learned.items() if name.startswith("pointer_head.")
        }
        _create_safetensors(export / "adapter.safetensors", adapter_tensors, safetensors)
        _create_safetensors(export / "pointer-head.safetensors", head_tensors, safetensors)
        _create_json(
            export / "config.json",
            export_config,
        )
        licenses_dir = export / "licenses"
        licenses_dir.mkdir(mode=0o700)
        if not licenses:
            raise ValueError("export requires verified license assets")
        for role, source in sorted(licenses.items()):
            if not role or not source.is_file() or source.is_symlink():
                raise ValueError("license asset is invalid")
            _create_bytes(licenses_dir / role, source.read_bytes())
        inventory = _asset_inventory(export)
        if any(
            any(word in item["path"].casefold() for word in ("optimizer", "corpus", "base"))
            for item in inventory
        ):
            raise ValueError("export contains a forbidden non-learned asset")
        tensor_inventory = _tensor_inventory(torch, learned)
        tensor_digest = _canonical_digest(tensor_inventory)
        trainable_digest = _canonical_digest(
            [item for item in inventory if item["path"].endswith(".safetensors")]
        )
        result = _metrics_result(metrics, trainable_digest)
        result_path = work / "training-result.json"
        if result_path.exists():
            if _load_json(result_path) != result:
                raise ValueError("preserved training result differs from completed run")
        else:
            _create_json(result_path, result)
        descriptor = {
            "schema_version": "v02-private-export.v1",
            "artifact_id": admission["artifact_id"],
            "admission_sha256": admission["admission_sha256"],
            "private_identity": admission["private_identity"],
            "base_id": admission["base_id"],
            "base_revision": admission["base_revision"],
            "base_inventory_sha256": admission["base_inventory_sha256"],
            "tokenizer_inventory_sha256": admission["tokenizer_inventory_sha256"],
            "renderer_sha256": admission["renderer_sha256"],
            "candidate_config_sha256": admission["candidate_config_sha256"],
            "tensor_inventory_sha256": tensor_digest,
            "files": inventory,
            "training_result_sha256": _sha_file(work / "training-result.json"),
        }
        _create_json(export / "export.json", descriptor)
        _publish_complete_directory(export, target)
        return descriptor
    except BaseException:
        # Failed exports never mutate the already-completed learned state.  The
        # incomplete directory is deliberately left as local failure evidence.
        raise


def _read_export(
    export: Path, safetensors: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    descriptor = _load_json(export / "export.json")
    if (
        set(descriptor) != EXPORT_FIELDS
        or descriptor.get("schema_version") != "v02-private-export.v1"
    ):
        raise ValueError("export descriptor is not closed")
    inventory = _asset_inventory(export)
    if inventory != descriptor["files"]:
        raise ValueError("export asset bytes changed")
    result = _load_json(export.parent / "training-result.json")
    if _sha_file(export.parent / "training-result.json") != descriptor["training_result_sha256"]:
        raise ValueError("training result binding changed")
    if (
        set(result) != TRAINING_RESULT_FIELDS
        or result.get("schema_version") != "v02-private-training-result.v1"
    ):
        raise ValueError("training result is not closed")
    result = _metrics_result(
        {
            key: value
            for key, value in result.items()
            if key != "schema_version" and key != "exported_trainable_sha256"
        },
        result["exported_trainable_sha256"],
    )
    adapter = safetensors.load_file(str(export / "adapter.safetensors"), device="cpu")
    head = safetensors.load_file(str(export / "pointer-head.safetensors"), device="cpu")
    if not adapter or set(head) != {"pointer_head.query.weight", "pointer_head.key.weight"}:
        raise ValueError("exported learned tensor partitions are invalid")
    tensors = {**adapter, **head}
    tensor_inventory = _tensor_inventory(None, tensors)
    if _canonical_digest(tensor_inventory) != descriptor["tensor_inventory_sha256"]:
        raise ValueError("exported tensor inventory changed")
    digest = _canonical_digest(
        [item for item in inventory if item["path"].endswith(".safetensors")]
    )
    if result["exported_trainable_sha256"] != digest:
        raise ValueError("training result exported tensor digest changed")
    config = _require_closed_config(_load_json(export / "config.json"))
    for key in (
        "admission_sha256",
        "base_inventory_sha256",
        "tokenizer_inventory_sha256",
        "renderer_sha256",
        "candidate_config_sha256",
        "tensor_inventory_sha256",
        "training_result_sha256",
    ):
        _digest(descriptor[key], key)
    if (
        result["artifact_id"] != descriptor["artifact_id"]
        or result["admission_sha256"] != descriptor["admission_sha256"]
        or result["private_identity"] != descriptor["private_identity"]
    ):
        raise ValueError("training result does not bind the export identity")
    return descriptor, result, tensors, config


def _load_learned(
    adapter: Any, head: Any, tensors: dict[str, Any], torch: Any
) -> tuple[list[str], list[str], dict[str, Any]]:
    adapter_values = {
        name: value for name, value in tensors.items() if not name.startswith("pointer_head.")
    }
    current = adapter.state_dict()
    expected = {name for name in current if ".lora_A." in name or ".lora_B." in name}
    missing = sorted(expected - set(adapter_values))
    unexpected = sorted(set(adapter_values) - expected)
    if missing or unexpected:
        raise ValueError("fresh adapter tensor names do not exactly match export")
    for name, value in adapter_values.items():
        target = current[name]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError("fresh adapter tensor shape differs from export")
        current[name] = value.to(dtype=target.dtype, device=target.device)
    adapter.load_state_dict(current, strict=True)
    expected_head = {"pointer_head.query.weight", "pointer_head.key.weight"}
    if {name for name in tensors if name.startswith("pointer_head.")} != expected_head:
        raise ValueError("fresh pointer head names differ from export")
    head_state = {
        name.removeprefix("pointer_head."): tensors[name].to(dtype=value.dtype, device=value.device)
        for name, value in (
            ("pointer_head.query.weight", head.query.weight),
            ("pointer_head.key.weight", head.key.weight),
        )
    }
    head.load_state_dict(head_state, strict=True)
    # Compare the exported values after the real runtime cast, not file bytes.
    for name, value in adapter_values.items():
        if not torch.equal(
            adapter.state_dict()[name].detach().cpu(),
            value.to(adapter.state_dict()[name].dtype).cpu(),
        ):
            raise ValueError("fresh adapter values differ from export after cast")
    for name, target in head_state.items():
        exported = tensors[f"pointer_head.{name}"]
        if not torch.equal(head.state_dict()[name].detach().cpu(), exported.to(target.dtype).cpu()):
            raise ValueError("fresh pointer head values differ from export after cast")
    loaded = _learned_tensors(adapter, head)
    if set(loaded) != set(tensors):
        raise ValueError("fresh learned tensor inventory differs from export")
    return missing, unexpected, loaded


def _active_scaling(adapter: Any, config: dict[str, Any], expected_modules: int) -> str:
    name = config["adapter_name"]
    active = getattr(adapter, "active_adapters", None)
    values = list(active if isinstance(active, (list, tuple)) else [active])
    if values != [name]:
        raise ValueError("exactly one exported PEFT adapter must be active")
    expected = float(config["lora_alpha"]) / float(config["lora_rank"])
    actual: list[dict[str, Any]] = []
    for module_name, module in adapter.named_modules():
        scaling = cast(Any, getattr(module, "scaling", None))
        lora_a = cast(Any, getattr(module, "lora_A", None))
        if (
            hasattr(scaling, "__contains__")
            and hasattr(lora_a, "__contains__")
            and name in scaling
            and name in lora_a
        ):
            if not math.isclose(float(scaling[name]), expected, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("PEFT scaling does not match alpha/r")
            actual.append({"module": module_name, "scaling": float(scaling[name])})
            merged = getattr(module, "merged", False)
            if merged is not False and merged != []:
                raise ValueError("fresh reload contains a merged PEFT layer")
    if len(actual) != expected_modules:
        raise ValueError("no live PEFT scaling modules were found")
    return _canonical_digest(actual)


def _model_device(module: Any) -> Any:
    devices = {str(parameter.device) for parameter in module.parameters()}
    if len(devices) != 1:
        raise ValueError("reload model must have one concrete device")
    return next(module.parameters()).device


def _scores(adapter: Any, head: Any, batch: dict[str, Any], torch: Any) -> Any:
    device = _model_device(adapter)
    if _model_device(head) != device:
        raise ValueError("adapter and pointer head devices differ")
    batch = v02_training._to_device(batch, device)
    with torch.no_grad():
        scores = v02_training._score_token_batch(adapter, head, batch)
    present = batch["option_present"]
    if not bool(torch.isfinite(scores[present]).all()) or not bool(
        (scores[~present] == float("-inf")).all()
    ):
        raise ValueError("reload has non-finite unmasked or unmasked padded scores")
    return scores.detach().cpu()


def fresh_reload_proof(
    *,
    export: Path,
    factory: Callable[[], tuple[Any, Any]],
    samples: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    torch: Any,
    safetensors: Any,
    expected_tensor_count: int = 498,
) -> dict[str, Any]:
    """Reconstruct a new model, load bytes, and prove active, typed ranking behavior."""
    descriptor, result, tensors, config = _read_export(export, safetensors)
    adapter, head = factory()
    device = _model_device(adapter)
    if _model_device(head) != device:
        raise ValueError("adapter and pointer head devices differ")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU reload device is malformed")
    frozen_dtypes = {
        parameter.dtype
        for name, parameter in adapter.named_parameters()
        if ".lora_A." not in name and ".lora_B." not in name
    }
    if frozen_dtypes != {torch.bfloat16} or {
        parameter.dtype for parameter in head.parameters()
    } != {torch.bfloat16}:
        raise ValueError("fresh reload must use BF16 frozen base and pointer head")
    missing, unexpected, loaded = _load_learned(adapter, head, tensors, torch)
    if len(tensors) != expected_tensor_count:
        raise ValueError("fresh reload tensor count is not the declared inventory")
    loaded_inventory = _tensor_inventory(torch, loaded)
    loaded_inventory_digest = _canonical_digest(loaded_inventory)
    if (
        loaded_inventory != _tensor_inventory(torch, tensors)
        or loaded_inventory_digest != descriptor["tensor_inventory_sha256"]
    ):
        raise ValueError("loaded tensor inventory does not match export")
    adapter.eval()
    head.eval()
    scaling_digest = _active_scaling(adapter, config, len(tensors) // 2 - 1)
    batches = list(v02_training._token_batches(torch, samples, 4))
    if not batches:
        raise ValueError("fresh reload requires fixed samples")
    disabled = getattr(adapter, "disable_adapter", None)
    if not callable(disabled):
        raise ValueError("PEFT adapter cannot be disabled for functional proof")
    maximum_difference = 0.0
    functional_effect = False
    first: Any | None = None
    for batch in batches:
        active = _scores(adapter, head, batch, torch)
        repeat = _scores(adapter, head, batch, torch)
        if not torch.equal(active, repeat):
            raise ValueError("fresh reload active scores are not stable")
        with disabled():
            baseline = _scores(adapter, head, batch, torch)
        present = batch["option_present"]
        difference = float((active[present] - baseline[present]).abs().max())
        threshold = 1e-6 + 1e-5 * float(baseline[present].abs().max())
        if not math.isfinite(difference):
            raise ValueError("active adapter has a non-finite functional effect")
        functional_effect = functional_effect or difference > threshold
        maximum_difference = max(maximum_difference, difference)
        if first is None:
            first = active
    if first is None:
        raise ValueError("fresh reload requires fixed samples")
    if not functional_effect:
        raise ValueError("active adapter has no nonzero functional effect")
    dev_batches = list(v02_training._token_batches(torch, dev_rows, 4))
    correct: dict[str, list[int]] = {}
    for batch in dev_batches:
        scores = _scores(adapter, head, batch, torch)
        for predicted, target, stratum in zip(
            scores.argmax(1).tolist(), batch["targets"].tolist(), batch["strata"], strict=True
        ):
            slot = correct.setdefault(stratum, [0, 0])
            slot[0] += int(predicted == target)
            slot[1] += 1
    if not correct:
        raise ValueError("fresh reload requires internal development rows")
    macro = sum(hits / total for hits, total in correct.values()) / len(correct)
    values = first[0, batches[0]["option_present"][0]].tolist()
    winner = max(range(len(values)), key=lambda index: values[index])
    response = DecisionResponse(
        api_version="v1alpha1",
        answers=(
            Answer(
                question_id="proof",
                type="choice",
                value=f"option-{winner}",
                raw_scores={f"option-{i}": float(value) for i, value in enumerate(values)},
                probabilities={f"option-{i}": 0.0 for i in range(len(values))},
                status="uncalibrated",
                score_semantics="ranking_weights",
                abstained=True,
                reason="uncalibrated_research",
                calibration=None,
            ),
        ),
        model=ModelReference(
            id="saracura/private",
            revision="private",
            checkpoint_sha256=result["exported_trainable_sha256"],
        ),
        usage=Usage(
            input_tokens=int(batches[0]["input_ids"].numel()), questions=1, criteria=len(values)
        ),
        automation_allowed=False,
    )
    if response.automation_allowed is not False:
        raise ValueError("typed response must fail closed for automation")
    answer = response.answers[0]
    if (
        answer.value != f"option-{winner}"
        or answer.status != "uncalibrated"
        or answer.abstained is not True
        or answer.score_semantics != "ranking_weights"
    ):
        raise ValueError("typed response does not reflect the measured ranking")
    identities = [str(row["identity_id"]) for row in samples]
    digest = _canonical_digest(identities)
    return {
        "schema_version": "v02-private-reload.v1",
        "artifact_id": descriptor["artifact_id"],
        "admission_sha256": descriptor["admission_sha256"],
        "exported_trainable_sha256": result["exported_trainable_sha256"],
        "base_inventory_sha256": descriptor["base_inventory_sha256"],
        "tensor_inventory_sha256": descriptor["tensor_inventory_sha256"],
        "precision": "bf16",
        "lora_merged": False,
        "sample_identity_digest": digest,
        "samples": len(samples),
        "finite_scores": True,
        "typed_contract_valid": True,
        "matched_tensor_count": len(tensors),
        "loaded_tensor_inventory_sha256": loaded_inventory_digest,
        "missing_tensor_keys": missing,
        "unexpected_tensor_keys": unexpected,
        "loaded_values_match_export": True,
        "active_adapter_names": [config["adapter_name"]],
        "adapters_enabled": True,
        "module_scaling_inventory_sha256": scaling_digest,
        "scaling_matches_config": True,
        "functional_sample_identity_digest": digest,
        "functional_tolerance": "atol1e-6_rtol1e-5.v1",
        "functional_max_abs_difference": maximum_difference,
        "functional_effect_verified": True,
        "reload_internal_dev_macro_accuracy": macro,
        "training_best_internal_dev_macro_accuracy": result["best_internal_dev_macro_accuracy"],
        "automation_allowed": False,
    }


def _validate_reload(
    receipt: dict[str, Any],
    descriptor: dict[str, Any],
    result: dict[str, Any],
    tensor_count: int,
    config: dict[str, Any],
) -> None:
    if set(receipt) != RELOAD_FIELDS or receipt.get("schema_version") != "v02-private-reload.v1":
        raise ValueError("reload receipt is not closed")
    bindings = {
        "artifact_id": descriptor["artifact_id"],
        "admission_sha256": descriptor["admission_sha256"],
        "exported_trainable_sha256": result["exported_trainable_sha256"],
        "base_inventory_sha256": descriptor["base_inventory_sha256"],
        "tensor_inventory_sha256": descriptor["tensor_inventory_sha256"],
        "loaded_tensor_inventory_sha256": descriptor["tensor_inventory_sha256"],
        "training_best_internal_dev_macro_accuracy": result["best_internal_dev_macro_accuracy"],
    }
    if any(receipt[key] != value for key, value in bindings.items()):
        raise ValueError("reload receipt does not bind the verified export")
    for key in (
        "admission_sha256",
        "exported_trainable_sha256",
        "base_inventory_sha256",
        "tensor_inventory_sha256",
        "loaded_tensor_inventory_sha256",
        "sample_identity_digest",
        "functional_sample_identity_digest",
        "module_scaling_inventory_sha256",
    ):
        _digest(receipt[key], key)
    if (
        receipt["precision"] != "bf16"
        or receipt["lora_merged"] is not False
        or receipt["finite_scores"] is not True
        or receipt["typed_contract_valid"] is not True
        or receipt["loaded_values_match_export"] is not True
        or receipt["adapters_enabled"] is not True
        or receipt["scaling_matches_config"] is not True
        or receipt["functional_tolerance"] != "atol1e-6_rtol1e-5.v1"
        or receipt["functional_effect_verified"] is not True
        or receipt["automation_allowed"] is not False
    ):
        raise ValueError("reload receipt proof flags are invalid")
    if (
        not _is_int(receipt["samples"])
        or receipt["samples"] <= 0
        or not _is_int(receipt["matched_tensor_count"])
        or receipt["matched_tensor_count"] <= 0
        or receipt["matched_tensor_count"] != tensor_count
        or receipt["active_adapter_names"] != [config["adapter_name"]]
        or receipt["missing_tensor_keys"] != []
        or receipt["unexpected_tensor_keys"] != []
        or not _finite(receipt["functional_max_abs_difference"], minimum=0.0)
        or receipt["functional_max_abs_difference"] <= 0
        or not _finite(receipt["reload_internal_dev_macro_accuracy"], minimum=0.0, maximum=1.0)
    ):
        raise ValueError("reload receipt values are invalid")


def publish_final_checkpoint(
    *, export: Path, final: Path, reload_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Copy verified immutable assets then emit reload -> run -> checkpoint receipts."""
    descriptor, result, tensors, config = _read_export(
        export, importlib.import_module("safetensors.torch")
    )
    _validate_reload(reload_receipt, descriptor, result, len(tensors), config)
    _private_parent(final)
    if final.exists():
        raise FileExistsError("final checkpoint is create-only")
    target = final
    final = Path(tempfile.mkdtemp(prefix=".private-final-", dir=final.parent))
    final.chmod(0o700)
    try:
        for item in descriptor["files"]:
            source, destination = export / item["path"], final / item["path"]
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _create_bytes(destination, source.read_bytes())
        _create_json(final / "reload-receipt.json", reload_receipt)
        run = {
            **result,
            "schema_version": "v02-private-training-run.v1",
            "reload_receipt_sha256": _sha_file(final / "reload-receipt.json"),
        }
        if set(run) != RUN_FIELDS:
            raise ValueError("run receipt is not closed")
        _create_json(final / "run-receipt.json", run)
        checkpoint = {
            "schema_version": "v02-private-checkpoint.v1",
            "artifact_id": descriptor["artifact_id"],
            "private_identity": descriptor["private_identity"],
            "admission_sha256": descriptor["admission_sha256"],
            "run_receipt_sha256": _sha_file(final / "run-receipt.json"),
            "base_id": descriptor["base_id"],
            "base_revision": descriptor["base_revision"],
            "base_inventory_sha256": descriptor["base_inventory_sha256"],
            "tokenizer_inventory_sha256": descriptor["tokenizer_inventory_sha256"],
            "renderer_sha256": descriptor["renderer_sha256"],
            "candidate_config_sha256": descriptor["candidate_config_sha256"],
            "tensor_inventory_sha256": descriptor["tensor_inventory_sha256"],
            "files": descriptor["files"],
            "private_only": True,
            "evaluation_pending": True,
            "publication_authorized": False,
            "automation_allowed": False,
        }
        _create_json(final / "checkpoint.json", checkpoint)
        _publish_complete_directory(final, target)
        return checkpoint
    except BaseException:
        # Export assets are never touched on a failed finalisation.
        raise


def verify_final_checkpoint(final: Path) -> dict[str, Any]:
    """Recompute every noncyclic final receipt and immutable asset binding."""
    checkpoint = _load_json(final / "checkpoint.json")
    if (
        set(checkpoint) != CHECKPOINT_FIELDS
        or checkpoint.get("schema_version") != "v02-private-checkpoint.v1"
    ):
        raise ValueError("checkpoint descriptor is not closed")
    run, reload = _load_json(final / "run-receipt.json"), _load_json(final / "reload-receipt.json")
    if set(run) != RUN_FIELDS or run.get("schema_version") != "v02-private-training-run.v1":
        raise ValueError("run receipt is not closed")
    if _sha_file(final / "run-receipt.json") != checkpoint["run_receipt_sha256"] or _sha_file(
        final / "reload-receipt.json"
    ) != run.get("reload_receipt_sha256"):
        raise ValueError("final receipt chain is not noncyclic and intact")
    files = _asset_inventory(final)
    if files != checkpoint["files"]:
        raise ValueError("final immutable assets differ from checkpoint descriptor")
    paths = {item["path"] for item in files}
    if (
        {"adapter.safetensors", "pointer-head.safetensors", "config.json"} - paths
        or not any(path.startswith("licenses/") for path in paths)
        or any(
            any(word in path.casefold() for word in ("optimizer", "corpus", "base"))
            for path in paths
        )
    ):
        raise ValueError("final checkpoint contains a non-learned asset")
    if not all(
        isinstance(checkpoint[key], str) and checkpoint[key]
        for key in (
            "artifact_id",
            "private_identity",
            "base_id",
            "base_revision",
        )
    ):
        raise ValueError("checkpoint identity fields are invalid")
    for key in (
        "admission_sha256",
        "run_receipt_sha256",
        "base_inventory_sha256",
        "tokenizer_inventory_sha256",
        "renderer_sha256",
        "candidate_config_sha256",
        "tensor_inventory_sha256",
    ):
        _digest(checkpoint[key], key)
    if (
        checkpoint["private_only"] is not True
        or checkpoint["evaluation_pending"] is not True
        or checkpoint["publication_authorized"] is not False
        or checkpoint["automation_allowed"] is not False
    ):
        raise ValueError("checkpoint safety flags drifted")
    result = _metrics_result(
        {
            key: value
            for key, value in run.items()
            if key not in {"schema_version", "exported_trainable_sha256", "reload_receipt_sha256"}
        },
        run["exported_trainable_sha256"],
    )
    descriptor = {
        "artifact_id": checkpoint["artifact_id"],
        "admission_sha256": checkpoint["admission_sha256"],
        "base_inventory_sha256": checkpoint["base_inventory_sha256"],
        "tensor_inventory_sha256": checkpoint["tensor_inventory_sha256"],
    }
    safetensors = importlib.import_module("safetensors.torch")
    adapter = safetensors.load_file(str(final / "adapter.safetensors"), device="cpu")
    head = safetensors.load_file(str(final / "pointer-head.safetensors"), device="cpu")
    tensors = {**adapter, **head}
    if _canonical_digest(_tensor_inventory(None, tensors)) != checkpoint["tensor_inventory_sha256"]:
        raise ValueError("final tensor inventory differs from checkpoint descriptor")
    config = _require_closed_config(_load_json(final / "config.json"))
    _validate_reload(reload, descriptor, result, len(tensors), config)
    if (
        run["admission_sha256"] != checkpoint["admission_sha256"]
        or run["private_identity"] != checkpoint["private_identity"]
        or run["artifact_id"] != checkpoint["artifact_id"]
        or run["exported_trainable_sha256"] != reload["exported_trainable_sha256"]
    ):
        raise ValueError("run receipt does not bind checkpoint identity")
    return checkpoint
