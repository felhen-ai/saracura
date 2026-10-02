"""Private, local-only P2 training workflow.

The public ``v02_training`` module owns the frozen model contract.  This
module owns the deliberately separate operational boundary: an admission is
verified before a runtime can be imported, a private identity is claimed once,
and an export is only made final after a fresh reload.  It is intentionally not
a downloader, a service, or a publication mechanism.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from benchmarks import v02_corpus, v02_training, v02_training_loop
from benchmarks.v02_corpus import verify_training_capsule
from saracura import serialization as saracura_serialization
from saracura.serialization import canonical_json_bytes

_RUN_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "plan_file",
        "receipt_file",
        "capsule_file",
        "exclusions_file",
        "sealer_inputs_file",
        "environment_observation_file",
        "base_directory",
        "base_inventory_file",
        "trainer_source_inventory_file",
        "renderer_preflight_file",
        "reviewed_runtime_manifest_file",
    }
)
_CLAIM_FIELDS = frozenset(
    {
        "schema_version",
        "private_identity",
        "admission_sha256",
        "run_inputs_sha256",
        "work_directory",
        "seed",
        "bindings_sha256",
    }
)
_RESUME_FIELDS = frozenset(
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
_HEX64 = v02_training.HEX64


@dataclass(frozen=True)
class PrivateAdmission:
    """The non-ML result of recomputing a private admission."""

    proof: v02_training.AdmissionProof
    descriptor: dict[str, Any]
    rows: list[dict[str, Any]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed(path: Path) -> dict[str, Any]:
    return v02_training._load_closed(path)


def _under(root: Path, value: Any, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("private input must be a non-empty root-relative path")
    path = v02_corpus._safe_relative_private_path(root, value, name="private input")
    if not path.exists() or path.is_symlink():
        raise ValueError("private input escapes training root")
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError("private input is not the required regular path")
    return path


def _private_root(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("private training root must be an absolute non-symlink directory")
    v02_corpus._safe_output_parent(root)
    return root


def _private_file(path: Path, root: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("private file must be a regular file below root")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("private file must be a regular file below root") from exc
    current = path.parent
    while current != root:
        if current.is_symlink():
            raise ValueError("private file must not traverse a symlink")
        current = current.parent
    metadata = path.stat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("private file must be owned 0600")


def load_run_inputs(root: Path, path: Path) -> dict[str, Any]:
    """Load the closed P2 input envelope without accepting path escapes."""
    root = _private_root(root)
    if path.parent != root:
        raise ValueError("private run inputs must remain at the training root")
    _private_file(path, root)
    value = _closed(path)
    if (
        set(value) != _RUN_INPUT_FIELDS
        or value.get("schema_version") != "v02-private-run-inputs.v1"
    ):
        raise ValueError("private run inputs are not closed")
    if value["artifact_id"] != "private-c1-r8-v1":
        raise ValueError("private run identity is not pinned")
    for key in _RUN_INPUT_FIELDS - {
        "schema_version",
        "artifact_id",
        "base_directory",
        "renderer_preflight_file",
    }:
        _private_file(_under(root, value[key]), root)
    _absolute_base_directory(value["base_directory"])
    preflight = v02_corpus._safe_relative_private_path(
        root, value["renderer_preflight_file"], name="renderer preflight"
    )
    if preflight.exists():
        _private_file(preflight, root)
    return value


def _absolute_base_directory(value: Any) -> Path:
    """Accept only an already acquired, read-only, non-symlink snapshot."""
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("base directory must be an absolute acquired snapshot")
    path = Path(value)
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("base directory must not traverse a symlink")
        current = current.parent
    if not path.is_dir() or path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o222:
        raise ValueError("base directory must be an owned read-only snapshot")
    return path


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    value = _closed(path)
    raw = path.read_bytes()
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{label} must contain canonical JSON")
    return value


def _safe_base_file(base: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("base inventory path is invalid")
    relative = Path(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("base inventory path is invalid")
    path = base / relative
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("base inventory traverses a symlink")
    if not path.is_file() or path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o222:
        raise ValueError("base inventory file is not read-only")
    return path


def _verify_base_inventory(base: Path, inventory_path: Path) -> tuple[str, dict[str, Any]]:
    inventory = _json_file(inventory_path, label="base inventory")
    if set(inventory) != {"role", "model", "revision", "status", "inventory"}:
        raise ValueError("base inventory is not closed")
    if (
        inventory["role"] != "student"
        or inventory["model"] != "Qwen/Qwen3.5-4B-Base"
        or inventory["revision"] != "1001bb4d826a52d1f399e183466143f4da7b741b"
        or inventory["status"] != "SOURCE_BYTES_VERIFIED_NOT_LOADED"
        or not isinstance(inventory["inventory"], list)
        or not inventory["inventory"]
    ):
        raise ValueError("base inventory binding is invalid")
    entries = inventory["inventory"]
    names: list[str] = []
    shard_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "bytes", "sha256"}:
            raise ValueError("base inventory entry is invalid")
        name, length, digest = entry["name"], entry["bytes"], entry["sha256"]
        if (
            not isinstance(name, str)
            or type(length) is not int
            or length < 0
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
        ):
            raise ValueError("base inventory entry is invalid")
        path = _safe_base_file(base, name)
        if path.stat().st_size != length or _sha(path) != digest:
            raise ValueError("base source bytes changed")
        names.append(name)
        if name.endswith(".safetensors"):
            shard_names.add(name)
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("base inventory entries are not sorted")
    if "config.json" not in names or not any(name.startswith("tokenizer") for name in names):
        raise ValueError("base inventory lacks config or tokenizer")
    if not any("license" in name.casefold() for name in names):
        raise ValueError("base inventory lacks license")
    for name in names:
        if name.endswith(".safetensors.index.json"):
            # Upstream bytes are pinned above, not rewritten into our receipt
            # serialization. HF snapshots use ordinary JSON formatting.
            index = _closed(base / name)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("safetensor index is invalid")
            referenced = set(weight_map.values())
            if not all(isinstance(item, str) and item in shard_names for item in referenced):
                raise ValueError("safetensor shards are incomplete")
    return _sha(inventory_path), inventory


def _module_inventory(config_path: Path) -> str:
    config = _closed(config_path)
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("base config has no pinned text configuration")
    model = SimpleNamespace(**config)
    model.text_config = SimpleNamespace(**text)
    v02_training.qwen35_text_config(model)
    tensors = [
        {"name": name, "shape": list(shape), "dtype": "bfloat16"}
        for name, shape in v02_training._expected_adapter_shapes(8).items()
    ]
    tensors.extend(
        (
            {"name": "pointer_head.query.weight", "shape": [256, 2560], "dtype": "bfloat16"},
            {"name": "pointer_head.key.weight", "shape": [256, 2560], "dtype": "bfloat16"},
        )
    )
    return _canonical_sha(sorted(tensors, key=lambda item: cast(str, item["name"])))


def _render_causal_record(
    renderer: v02_corpus.VerifiedRenderer,
    tokenizer: v02_corpus.VerifiedTokenizer,
    case: dict[str, Any],
) -> tuple[list[int], int, list[int]]:
    """Read the one pinned causal encoding without reconstructing its boundaries."""
    encoded = renderer._encode(
        tokenizer,
        v02_corpus._renderer_record(case),
        max_state=384,
        max_branch=1024,
        strict=True,
        option_isolation=False,
    )
    if not isinstance(encoded, dict):
        raise ValueError("pinned renderer did not produce a closed causal record")
    ids = encoded.get("ids")
    decide_indices = encoded.get("decide_idx")
    option_indices = encoded.get("opt_idx")
    if (
        not isinstance(ids, list)
        or any(type(item) is not int or item < 0 for item in ids)
        or encoded.get("state_truncated") is not False
        or encoded.get("labels") != [0]
        or not isinstance(decide_indices, list)
        or len(decide_indices) != 1
        or type(decide_indices[0]) is not int
        or not isinstance(option_indices, list)
        or len(option_indices) != 1
        or not isinstance(option_indices[0], list)
        or len(option_indices[0]) != len(case["options"])
        or any(type(index) is not int for index in option_indices[0])
    ):
        raise ValueError("pinned renderer did not produce a closed causal record")
    decide_index = decide_indices[0]
    end_indices = list(option_indices[0])
    if (
        not 0 <= decide_index < len(ids)
        or any(not 0 <= index < len(ids) for index in end_indices)
        or end_indices != sorted(end_indices)
        or len(set(end_indices)) != len(end_indices)
        or decide_index <= end_indices[-1]
    ):
        raise ValueError("pinned renderer causal indices are invalid")
    return list(ids), decide_index, end_indices


def prepare_verified_rows(
    capsule_rows: list[dict[str, Any]],
    *,
    renderer: v02_corpus.VerifiedRenderer,
    tokenizer: v02_corpus.VerifiedTokenizer,
) -> list[dict[str, Any]]:
    """Losslessly prepare verified P1 rows for the one-forward causal contract."""
    renderer.revalidate()
    tokenizer.revalidate()
    prepared: list[dict[str, Any]] = []
    for source in capsule_rows:
        options = source["options"]
        if not isinstance(options, list) or not 2 <= len(options) <= 8:
            raise ValueError("verified row option cardinality drifted")
        case = {"state": source["state"], "question": source["instruction"], "options": options}
        causal_ids, decide_index, end_indices = _render_causal_record(renderer, tokenizer, case)
        if len(causal_ids) > 512:
            raise ValueError("verified row exceeds the frozen causal context")
        locale = source["locale"]
        if (
            source.get("split") not in {"train", "internal_dev"}
            or locale not in {"pt_br", "english"}
            or type(source.get("gold_index")) is not int
            or not 0 <= source["gold_index"] < len(options)
        ):
            raise ValueError("verified row locale drifted")
        prepared.append(
            {
                "split": source["split"],
                "input_ids": causal_ids,
                "decide_index": decide_index,
                "end_option_indices": end_indices,
                "target": source["gold_index"],
                "stratum": "pt-BR" if locale == "pt_br" else "en",
            }
        )
    renderer.revalidate()
    tokenizer.revalidate()
    # The established parser supplies the final closed-index and lane checks.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "prepared.json"
        path.write_bytes(canonical_json_bytes(cast(Any, {"records": prepared})))
        validated = v02_training._prepared_examples(path)
        identities = [source.get("identity_id") for source in capsule_rows]
        if any(not isinstance(identity, str) or not identity for identity in identities) or len(
            set(identities)
        ) != len(identities):
            raise ValueError("verified row identities are invalid")
        return [
            row | {"identity_id": identity}
            for row, identity in zip(validated, identities, strict=True)
        ]


def _validated_runtime_components(
    root: Path, inputs: dict[str, Any]
) -> tuple[
    dict[str, Any], dict[str, Any], v02_corpus.VerifiedRenderer, v02_corpus.VerifiedTokenizer
]:
    sealer_path = _under(root, inputs["sealer_inputs_file"])
    sealer = _json_file(sealer_path, label="sealer inputs")
    grant_path = _under(root, sealer["environment_grant_file"])
    grant = _json_file(grant_path, label="environment grant")
    licenses = {
        role: _under(root, relative).read_bytes()
        for role, relative in cast(dict[str, Any], sealer["license_files"]).items()
    }
    v02_corpus.validate_environment_grant(grant, licenses)
    renderer_path = _under(root, sealer["renderer_source_file"])
    renderer_raw = renderer_path.read_bytes()
    if _sha(renderer_path) != sealer["renderer_source_sha256"]:
        raise ValueError("renderer source bytes changed")
    renderer = v02_corpus.load_pinned_renderer(renderer_raw)
    inventory_path = _under(root, sealer["tokenizer_inventory_file"])
    inventory = v02_corpus._read_canonical_json_value(
        inventory_path.read_bytes(), name="tokenizer inventory"
    )
    if not isinstance(inventory, list):
        raise ValueError("tokenizer inventory is not a list")
    tokenizer = v02_corpus.load_verified_tokenizer(
        _under(root, sealer["tokenizer_directory"], directory=True), inventory
    )
    return sealer, grant, renderer, tokenizer


def _module_path(module: Any, repository: Path) -> Path:
    value = getattr(module, "__file__", None)
    if type(value) is not str or not Path(value).is_absolute():
        raise ValueError("imported module has no absolute source path")
    path = Path(value)
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError("imported module is shadowed")
    try:
        path.relative_to(repository)
    except ValueError as exc:
        raise ValueError("imported module is shadowed") from exc
    return path


def _verify_trainer_sources(root: Path, path: Path) -> tuple[str, str]:
    del root
    value = _json_file(path, label="trainer source inventory")
    if (
        set(value) != {"schema_version", "source_revision", "files"}
        or value.get("schema_version") != "v02-trainer-source-inventory.v1"
    ):
        raise ValueError("trainer source inventory is not closed")
    revision = value.get("source_revision")
    files = value.get("files")
    if not isinstance(revision, str) or v02_training.HEX40.fullmatch(revision) is None:
        raise ValueError("trainer source revision is invalid")
    if not isinstance(files, list) or not files:
        raise ValueError("trainer source provenance drifted")
    own_path = Path(__file__)
    if (
        not own_path.is_absolute()
        or own_path.is_symlink()
        or own_path.resolve(strict=True) != own_path
    ):
        raise ValueError("trainer source is shadowed")
    repository = own_path.parents[1]
    actual: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("trainer source inventory entry is invalid")
        relative = item["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or _HEX64.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("trainer source inventory path is invalid")
        target = repository / relative
        if (
            target.is_symlink()
            or not target.is_file()
            or target.resolve(strict=True) != target
            or target.stat().st_size != item["bytes"]
        ):
            raise ValueError("trainer source file changed")
        if _sha(target) != item["sha256"]:
            raise ValueError("trainer source file changed")
        actual.append(item)
    names = [cast(str, item["path"]) for item in actual]
    if actual != sorted(actual, key=lambda item: cast(str, item["path"])) or len(set(names)) != len(
        names
    ):
        raise ValueError("trainer source inventory is not sorted")
    imported = (
        _module_path(SimpleNamespace(__file__=__file__), repository),
        _module_path(v02_training, repository),
        _module_path(v02_corpus, repository),
        _module_path(v02_training_loop, repository),
        _module_path(saracura_serialization, repository),
    )
    listed = {repository / cast(str, item["path"]) for item in actual}
    if any(module not in listed for module in imported):
        raise ValueError("imported module is shadowed or absent from source inventory")
    return revision, _canonical_sha(actual)


def _default_machine_probe() -> dict[str, Any]:
    """Read live host identity without involving any ML runtime."""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        uuid, memory = output.splitlines()[0].split(",", maxsplit=1)
        return {"kernel_boot_id": boot_id, "gpu_uuid": uuid.strip(), "gpu_memory_mib": int(memory)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise ValueError("live private host observation is unavailable") from exc


def _verify_environment(
    root: Path,
    inputs: dict[str, Any],
    *,
    grant_raw_sha: str,
    runtime_lock_raw_sha: str,
    runtime_lock: dict[str, Any],
    grant: dict[str, Any],
    grant_digest: str,
    training_plan_sha: str,
    sealed_plan_sha: str,
    probe: Callable[[], dict[str, Any]],
) -> str:
    path = _under(root, inputs["environment_observation_file"])
    value = _json_file(path, label="physical environment receipt")
    expected = {
        "schema_version",
        "artifact_id",
        "status",
        "host_identity_before",
        "host_identity_after",
        "preflight_file",
        "preflight_sha256",
        "environment_grant_sha256",
        "runtime_lock_sha256",
        "source_inventory_sha256",
        "reviewed_code_manifest_sha256",
        "weight_roles",
        "sealed_weights_loaded",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != "v02-private-physical-environment.v1"
    ):
        raise ValueError("physical environment receipt is not closed")
    if (
        value["status"] != "PHYSICALLY_ADMITTED_TRAINING_ONLY"
        or value["environment_grant_sha256"] != grant_raw_sha
        or value["runtime_lock_sha256"] != runtime_lock_raw_sha
        or value["weight_roles"] != ["training_author", "independent_reviewer"]
        or value["sealed_weights_loaded"] is not False
    ):
        raise ValueError("physical environment receipt binding drifted")
    identity_keys = {
        "pod_id",
        "region",
        "image_sha256",
        "started_at",
        "ssh_endpoint",
        "kernel_boot_id",
        "gpu_uuid",
        "gpu_memory_mib",
    }
    before, after = value["host_identity_before"], value["host_identity_after"]
    if not isinstance(before, dict) or before != after or set(before) != identity_keys:
        raise ValueError("physical environment identity is not closed")
    endpoint = before["ssh_endpoint"]
    if (
        any(
            type(before[field]) is not str or not before[field].strip()
            for field in identity_keys - {"gpu_memory_mib"}
        )
        or not isinstance(endpoint, str)
        or len(endpoint.split(":")) != 3
        or not endpoint.split(":")[0].strip()
        or not endpoint.split(":")[2].strip()
        or not endpoint.split(":")[1].isdigit()
        or not 1 <= int(endpoint.split(":")[1]) <= 65535
    ):
        raise ValueError("physical environment identity is not closed")
    lock_memory = runtime_lock.get("gpu_memory_mib")
    if (
        before["image_sha256"] != runtime_lock.get("image_digest")
        or before["region"] != grant.get("region")
        or type(before["gpu_memory_mib"]) is not int
        or before["gpu_memory_mib"] <= 0
        or type(lock_memory) is not int
        or lock_memory <= 0
        or before["gpu_memory_mib"] != lock_memory
    ):
        raise ValueError("physical environment identity binding drifted")
    observed = probe()
    machine_keys = {"kernel_boot_id", "gpu_uuid", "gpu_memory_mib"}
    if (
        not isinstance(observed, dict)
        or set(observed) != machine_keys
        or any(observed[key] != before[key] for key in machine_keys)
    ):
        raise ValueError("live host boot or GPU identity drifted")
    preflight_path = _under(root, value["preflight_file"])
    if _sha(preflight_path) != value["preflight_sha256"]:
        raise ValueError("physical preflight bytes changed")
    preflight = _json_file(preflight_path, label="P3 preflight")
    preflight_keys = {
        "schema_version",
        "status",
        "source_commit",
        "source_inventory_digest",
        "code_manifest_digest",
        "training_plan_digest",
        "sealed_plan_digest",
        "micro_selection_digest",
        "grammar_receipt_digest",
        "feasibility_receipt_digest",
        "grant_digest",
        "runtime_lock_digest",
        "runtime_evidence_digests",
        "fresh_asset_probe_digests",
        "historical_model_metadata_sha256",
        "historical_measurements",
        "model_calls",
        "reservations",
    }
    if (
        set(preflight) != preflight_keys
        or preflight.get("schema_version") != "c7-preflight-observation.v1"
    ):
        raise ValueError("P3 preflight is not closed")
    if (
        preflight.get("status") != "OBSERVED_REVIEW_PENDING"
        or preflight.get("source_commit") != runtime_lock.get("source_commit")
        or preflight.get("source_inventory_digest")
        != runtime_lock.get("public_source_integrity", {}).get("source_inventory_digest")
        or preflight.get("training_plan_digest") != training_plan_sha
        or preflight.get("sealed_plan_digest") != sealed_plan_sha
        or preflight.get("grant_digest") != grant_digest
        or preflight.get("runtime_lock_digest") != _canonical_sha(runtime_lock)
        or grant.get("training_plan_digest") != training_plan_sha
        or grant.get("sealed_plan_digest") != sealed_plan_sha
        or grant.get("runtime_lock_digest") != _canonical_sha(runtime_lock)
        or value["source_inventory_sha256"] != preflight.get("source_inventory_digest")
    ):
        raise ValueError("P3 preflight source binding drifted")
    manifest_path = _under(root, inputs["reviewed_runtime_manifest_file"])
    manifest = _json_file(manifest_path, label="reviewed runtime manifest")
    if (
        set(manifest) != {"schema_version", "source_commit", "code_sha256"}
        or manifest.get("schema_version") != "private-reviewed-runtime-code.v1"
        or manifest.get("source_commit") != runtime_lock.get("source_commit")
        or manifest.get("code_sha256") != runtime_lock.get("code_sha256")
        or _sha(manifest_path) != value["reviewed_code_manifest_sha256"]
        or preflight.get("code_manifest_digest") != _canonical_sha(manifest["code_sha256"])
    ):
        raise ValueError("reviewed runtime manifest binding drifted")
    return _sha(path)


def _expected_admission(
    *,
    root: Path,
    inputs: dict[str, Any],
    descriptor: dict[str, Any],
    renderer: v02_corpus.VerifiedRenderer,
    tokenizer: v02_corpus.VerifiedTokenizer,
    environment_sha: str,
    trainer_revision: str,
    trainer_inventory_sha: str,
    base_inventory_sha: str,
    base_inventory: dict[str, Any],
    module_inventory_sha: str,
) -> dict[str, Any]:
    sealer, _grant, _renderer, _tokenizer = _validated_runtime_components(root, inputs)
    licenses = cast(dict[str, Any], sealer["license_files"])
    license_inventory = {role: _sha(_under(root, path)) for role, path in licenses.items()}
    sealed_models = {
        role: {key: v02_corpus.MODEL_ROLES[role][key] for key in ("model", "revision", "license")}
        for role in sorted(v02_corpus.MODEL_ROLES)
        if role.startswith("sealed_")
    }
    candidate = v02_training.candidate_config("c1-r8")
    return {
        "schema_version": "v02-private-training-admission.v1",
        "artifact_id": "private-c1-r8-v1",
        "private_identity": "private-c1-r8-v1",
        "capsule_sha256": _sha(_under(root, inputs["capsule_file"])),
        "aggregate_receipt_sha256": descriptor["aggregate_receipt_sha256"],
        "rights_receipt_sha256": descriptor["rights_receipt_sha256"],
        "license_inventory_sha256": _canonical_sha(license_inventory),
        "environment_receipt_sha256": environment_sha,
        "corpus_source_revision": descriptor["corpus_source_revision"],
        "trainer_source_revision": trainer_revision,
        "trainer_code_inventory_sha256": trainer_inventory_sha,
        "base_id": base_inventory["model"],
        "base_revision": base_inventory["revision"],
        "base_inventory_sha256": base_inventory_sha,
        "tokenizer_inventory_sha256": tokenizer.inventory_digest,
        "renderer_sha256": renderer.source_sha256,
        "module_inventory_sha256": module_inventory_sha,
        "candidate_config_sha256": _canonical_sha(candidate),
        "renderer_preflight_sha256": "",
        "sealed_plan_sha256": hashlib.sha256(v02_corpus._frozen_plan_bytes("sealed")).hexdigest(),
        "sealed_prompt_inventory_sha256": v02_corpus.prompt_contract()["contract_digest"],
        "sealed_model_inventory_sha256": _canonical_sha(sealed_models),
        "seed": 20260929,
        "training_authorized": True,
        "private_only": True,
        "publication_authorized": False,
    }


def _renderer_preflight(
    *,
    descriptor: dict[str, Any],
    capsule_sha256: str,
    renderer: v02_corpus.VerifiedRenderer,
    tokenizer: v02_corpus.VerifiedTokenizer,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # ``prepare_verified_rows`` runs the candidate bound.  This independently
    # runs the broader Kev bound so a claimed boolean is never trusted.
    for row in rows:
        case = {"state": row["state"], "question": row["instruction"], "options": row["options"]}
        measured = renderer.measure(case, tokenizer)
        if measured["token_count"] > 2048 or measured["state_truncated"] is not False:
            raise ValueError("Kev renderer preflight failed")
    return {
        "schema_version": "v02-renderer-preflight.v1",
        "artifact_id": descriptor["artifact_id"] + "-renderer-preflight",
        "capsule_sha256": capsule_sha256,
        "candidate_renderer_sha256": v02_corpus.candidate_rendering_digest(),
        "kev_rendering_contract_sha256": v02_corpus.kev_rendering_function_digest(),
        "accepted_records": len(rows),
        "candidate_all_records_pass": True,
        "kev_all_records_pass": True,
    }


def _create_private_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("private output already exists")
    fd, temporary_name = tempfile.mkstemp(prefix=".private-json-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _admission_context(
    *, root: Path, run_inputs_path: Path, probe: Callable[[], dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Recompute every dependency used by admission, with no ML import."""
    root = _private_root(root)
    inputs = load_run_inputs(root, run_inputs_path)
    paths = {
        key: _under(root, inputs[key])
        for key in _RUN_INPUT_FIELDS
        - {"schema_version", "artifact_id", "base_directory", "renderer_preflight_file"}
    }
    descriptor, capsule_rows = verify_training_capsule(
        paths["capsule_file"],
        root=root,
        plan_path=paths["plan_file"],
        receipt_path=paths["receipt_file"],
        exclusions_path=paths["exclusions_file"],
        sealer_inputs_path=paths["sealer_inputs_file"],
    )
    sealer, grant, renderer, tokenizer = _validated_runtime_components(root, inputs)
    if (
        descriptor["renderer_sha256"] != renderer.source_sha256
        or descriptor["tokenizer_inventory_sha256"] != tokenizer.inventory_digest
        or descriptor["rights_receipt_sha256"]
        != _sha(_under(root, sealer["environment_grant_file"]))
    ):
        raise ValueError("P1 descriptor runtime binding drifted")
    runtime_lock_path = _under(root, sealer["runtime_lock_file"])
    runtime_lock = _json_file(runtime_lock_path, label="runtime lock")
    trainer_revision, trainer_inventory_sha = _verify_trainer_sources(
        root, paths["trainer_source_inventory_file"]
    )
    base_directory = _absolute_base_directory(inputs["base_directory"])
    base_inventory_sha, base_inventory = _verify_base_inventory(
        base_directory, paths["base_inventory_file"]
    )
    module_inventory_sha = _module_inventory(base_directory / "config.json")
    licenses = {
        role: _under(root, relative).read_bytes()
        for role, relative in cast(dict[str, Any], sealer["license_files"]).items()
    }
    environment_sha = _verify_environment(
        root,
        inputs,
        grant_raw_sha=_sha(_under(root, sealer["environment_grant_file"])),
        runtime_lock_raw_sha=_sha(runtime_lock_path),
        runtime_lock=runtime_lock,
        grant=grant,
        grant_digest=v02_corpus.validate_environment_grant(grant, licenses),
        training_plan_sha=hashlib.sha256(v02_corpus._frozen_plan_bytes("training")).hexdigest(),
        sealed_plan_sha=hashlib.sha256(v02_corpus._frozen_plan_bytes("sealed")).hexdigest(),
        probe=probe,
    )
    expected = _expected_admission(
        root=root,
        inputs=inputs,
        descriptor=descriptor,
        renderer=renderer,
        tokenizer=tokenizer,
        environment_sha=environment_sha,
        trainer_revision=trainer_revision,
        trainer_inventory_sha=trainer_inventory_sha,
        base_inventory_sha=base_inventory_sha,
        base_inventory=base_inventory,
        module_inventory_sha=module_inventory_sha,
    )
    prepared = prepare_verified_rows(capsule_rows, renderer=renderer, tokenizer=tokenizer)
    preflight = _renderer_preflight(
        descriptor=descriptor,
        capsule_sha256=_sha(paths["capsule_file"]),
        renderer=renderer,
        tokenizer=tokenizer,
        rows=capsule_rows,
    )
    return expected, descriptor, prepared, preflight


def create_private_admission(
    *,
    root: Path,
    run_inputs_path: Path,
    output: Path,
    machine_probe: Callable[[], dict[str, Any]] | None = None,
) -> PrivateAdmission:
    """Create a single immutable admission after actual P1 and physical checks."""
    root = _private_root(root)
    if output.parent != root:
        raise ValueError("admission output must be created at the training root")
    if output.exists() or output.is_symlink():
        raise FileExistsError("admission output already exists")
    expected, descriptor, prepared, preflight = _admission_context(
        root=root, run_inputs_path=run_inputs_path, probe=machine_probe or _default_machine_probe
    )
    preflight_path = v02_corpus._safe_relative_private_path(
        root,
        load_run_inputs(root, run_inputs_path)["renderer_preflight_file"],
        name="renderer preflight",
    )
    if preflight_path.exists():
        if _json_file(preflight_path, label="renderer preflight") != preflight:
            raise ValueError("renderer preflight binding drifted")
    else:
        _create_private_json(preflight_path, preflight)
    expected["renderer_preflight_sha256"] = _sha(preflight_path)
    _create_private_json(output, expected)
    proof = v02_training.AdmissionProof(_sha(output))
    return PrivateAdmission(proof, descriptor, prepared)


def verify_private_admission(
    *,
    root: Path,
    run_inputs_path: Path,
    admission_path: Path,
    machine_probe: Callable[[], dict[str, Any]] | None = None,
) -> PrivateAdmission:
    """Recompute a create-only admission; malformed or stale bytes never pass."""
    root = _private_root(root)
    _private_file(admission_path, root)
    expected, descriptor, prepared, preflight = _admission_context(
        root=root, run_inputs_path=run_inputs_path, probe=machine_probe or _default_machine_probe
    )
    preflight_path = v02_corpus._safe_relative_private_path(
        root,
        load_run_inputs(root, run_inputs_path)["renderer_preflight_file"],
        name="renderer preflight",
    )
    if _json_file(preflight_path, label="renderer preflight") != preflight:
        raise ValueError("renderer preflight binding drifted")
    expected["renderer_preflight_sha256"] = _sha(preflight_path)
    actual = _json_file(admission_path, label="private admission")
    if set(actual) != v02_training._ADMISSION_FIELDS or actual != expected:
        raise ValueError("private admission binding drifted")
    return PrivateAdmission(v02_training.AdmissionProof(_sha(admission_path)), descriptor, prepared)


@contextmanager
def private_identity_lock(root: Path) -> Iterator[None]:
    claims = root / "private-run-claims"
    claims.mkdir(mode=0o700, exist_ok=True)
    os.chmod(claims, 0o700)
    lock_path = claims / "private-c1-r8-v1.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".p2-", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish_directory_exclusively(staging: Path, target: Path) -> None:
    """Publish a complete sibling directory without replacing any reservation."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1)
    elif sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staging), os.fsencode(target), 4)
    else:
        raise OSError("atomic exclusive directory publication is unsupported on this host")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def publish_step_zero_claim(
    *,
    root: Path,
    work_directory: Path,
    admission_sha256: str,
    run_inputs_path: Path,
    bindings_sha256: str,
    initial_state: dict[str, Any],
    torch: Any,
) -> Path:
    """Atomically publish the paired claim and safe step-zero resume state."""
    if not (_HEX64.fullmatch(admission_sha256) and _HEX64.fullmatch(bindings_sha256)):
        raise ValueError("claim digest is invalid")
    claims = root / "private-run-claims"
    target = claims / "private-c1-r8-v1"
    if target.exists() or target.is_symlink():
        raise FileExistsError("private identity is already claimed")
    if work_directory.exists() or not work_directory.is_absolute():
        raise ValueError("work directory must be a new absolute location")
    staging = claims / f".private-c1-r8-v1-{os.getpid()}"
    if staging.exists():
        raise FileExistsError("claim staging collision")
    staging.mkdir(mode=0o700)
    claim = {
        "schema_version": "v02-private-run-claim.v1",
        "private_identity": "private-c1-r8-v1",
        "admission_sha256": admission_sha256,
        "run_inputs_sha256": _sha(run_inputs_path),
        "work_directory": str(work_directory),
        "seed": 20260929,
        "bindings_sha256": bindings_sha256,
    }
    v02_training_loop.validate_resume_state(initial_state)
    if (
        initial_state["optimizer_steps"] != 0
        or initial_state["epoch"] != 1
        or initial_state["next_batch_index"] != 0
        or initial_state["bindings_sha256"] != bindings_sha256
        or initial_state["private_identity"] != "private-c1-r8-v1"
    ):
        raise ValueError("step-zero state is not closed")
    _atomic_bytes(staging / "claim.json", canonical_json_bytes(cast(Any, claim)) + b"\n")
    v02_training_loop.write_resume_state(torch, staging / "initial-state.pt", initial_state)
    _publish_directory_exclusively(staging, target)
    directory = os.open(claims, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return target


def load_resume_state(path: Path, *, torch: Any, bindings_sha256: str) -> dict[str, Any]:
    """Load only a state published by this run, without pickle globals."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != _RESUME_FIELDS:
        raise ValueError("resume state is not closed")
    if (
        state["bindings_sha256"] != bindings_sha256
        or state["private_identity"] != "private-c1-r8-v1"
    ):
        raise ValueError("resume state binding mismatch")
    return state


def main() -> int:
    from benchmarks.v02_private_workflow import main as workflow_main

    return workflow_main()


if __name__ == "__main__":
    raise SystemExit(main())
