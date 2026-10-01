"""Offline-only machinery for the first Saracura-owned checkpoint.

This module intentionally has no model import.  It seals aggregate evidence,
defines the immutable candidate grid, and validates selection evidence.  Model
loading is deferred to the separately authorised live phases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from benchmarks.v02_corpus import RENDERER_CONTRACT
from benchmarks.v02_evaluation import (
    _normalise,
    _option_description,
    combined_content_fingerprint,
    state_question_fingerprint,
)
from saracura.serialization import canonical_json_bytes

MANIFEST_PATH = Path(__file__).parent / "manifests/v02-first-checkpoint.v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_GRID = (
    {
        "id": "c1-r8",
        "lora_rank": 8,
        "lora_alpha": 16,
        "dropout": 0.05,
        "peak_learning_rate": "2e-4",
    },
    {
        "id": "c2-r16",
        "lora_rank": 16,
        "lora_alpha": 32,
        "dropout": 0.05,
        "peak_learning_rate": "1e-4",
    },
    {
        "id": "c3-r32",
        "lora_rank": 32,
        "lora_alpha": 64,
        "dropout": 0.05,
        "peak_learning_rate": "5e-5",
    },
)
TRAINING_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "record_count",
        "split_counts",
        "identity_set_digest",
        "state_question_fingerprint_set_digest",
        "combined_content_fingerprint_set_digest",
        "option_multiset_fingerprint_set_digest",
        "accepted_packet_digest",
        "successor_grant_digest",
        "candidate_rendering_digest",
        "source_policy_identifiers",
    }
)
READINESS_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "training_descriptor_digest",
        "sealed_descriptor_digest",
        "disjointness_receipt_digest",
        "candidate_rendering_digest",
        "kev_rendering_digest",
        "all_records_candidate_preflight_passed",
        "all_records_kev_preflight_passed",
    }
)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_closed(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest_set(values: Iterable[str]) -> str:
    return _digest(sorted(set(values)))


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be sha256")
    return value


def load_manifest() -> dict[str, Any]:
    manifest = _load_closed(MANIFEST_PATH)
    expected = {
        "schema_version",
        "architecture",
        "candidate_grid",
        "early_stopping",
        "evaluation",
        "renderer",
        "selection",
        "training",
    }
    if set(manifest) != expected or manifest["schema_version"] != "v02-first-checkpoint.v1":
        raise ValueError("first checkpoint manifest is not closed")
    if manifest["candidate_grid"] != list(CANDIDATE_GRID):
        raise ValueError("candidate grid drifted")
    renderer = manifest["renderer"]
    if not isinstance(renderer, dict) or renderer != {
        "candidate_rendering_digest": RENDERER_CONTRACT["candidate_rendering_digest"],
        "source_path": "kev/model.py",
        "source_revision": RENDERER_CONTRACT["candidate_renderer_source_revision"],
        "source_sha256": RENDERER_CONTRACT["candidate_renderer_source_sha256"],
        "token_limit": 512,
        "truncation_disabled": True,
    }:
        raise ValueError("candidate renderer ledger drifted")
    if manifest["evaluation"] != {
        "batch": 1,
        "concurrency": 1,
        "determinism_flags": [
            "CUBLAS_WORKSPACE_CONFIG",
            "torch.use_deterministic_algorithms",
            "cudnn_benchmark_false",
        ],
        "diagnostic_subset_records": 50,
        "loopback_host": "127.0.0.1",
        "runtime_boundary": "loopback_http_python_worker_v1",
        "timing_procedure": "saracura-v02-loopback-single-request-v1",
        "warmups": 10,
        "warmup_measurements": 100,
    }:
        raise ValueError("loopback timing boundary drifted")
    return manifest


def _safe_output(path: Path) -> None:
    if not path.is_absolute() or path.name in {"", ".", ".."} or path.is_symlink():
        raise ValueError("output path must be an absolute non-symlink file")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink() or stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise ValueError("output parent must be a private existing directory")
    if path.exists():
        raise FileExistsError("create-only output already exists")


def create_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create a 0600 canonical JSON artifact, never replacing one."""
    _safe_output(path)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".v02-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError("create-only output already exists") from exc
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {
        "opaque_record_id",
        "split",
        "state",
        "question",
        "options",
    }:
        raise ValueError("training record is not closed")
    if record["split"] not in {"train", "internal_dev"} or not isinstance(
        record["opaque_record_id"], str
    ):
        raise ValueError("training record identity or split invalid")
    if not isinstance(record["options"], list):
        raise ValueError("training options invalid")
    return record


def _option_multiset_fingerprint(options: list[Any]) -> str:
    """Use the same #44 NFC/casefold/RFC-8785 option-description normalization."""
    normalized = [_normalise(_option_description(option)) for option in options]
    return hashlib.sha256(
        canonical_json_bytes(
            [json.loads(item) for item in sorted(canonical_json_bytes(x) for x in normalized)]
        )
    ).hexdigest()


def seal_training(
    records: list[Any],
    *,
    accepted_packet_digest: str,
    successor_grant_digest: str,
    source_policy_identifiers: list[str],
) -> dict[str, Any]:
    """Produce only aggregate descriptor material; callers retain raw rows privately."""
    load_manifest()
    if (
        not records
        or not source_policy_identifiers
        or not all(isinstance(item, str) and item for item in source_policy_identifiers)
    ):
        raise ValueError("training lanes and source policies are required")
    parsed = [_record(item) for item in records]
    if {item["split"] for item in parsed} != {"train", "internal_dev"}:
        raise ValueError("Phase 4E train and internal dev must both be in training lane")
    identities = [item["opaque_record_id"] for item in parsed]
    if len(identities) != len(set(identities)):
        raise ValueError("training identity collision")
    return {
        "schema_version": "v02-training-descriptor.v1",
        "record_count": len(parsed),
        "split_counts": {
            split: sum(item["split"] == split for item in parsed)
            for split in ("train", "internal_dev")
        },
        "identity_set_digest": _digest_set(identities),
        "state_question_fingerprint_set_digest": _digest_set(
            state_question_fingerprint(item["state"], item["question"]) for item in parsed
        ),
        "combined_content_fingerprint_set_digest": _digest_set(
            combined_content_fingerprint(item["state"], item["question"], item["options"])
            for item in parsed
        ),
        "option_multiset_fingerprint_set_digest": _digest_set(
            _option_multiset_fingerprint(item["options"]) for item in parsed
        ),
        "accepted_packet_digest": _require_digest(accepted_packet_digest, "accepted packet"),
        "successor_grant_digest": _require_digest(successor_grant_digest, "successor grant"),
        "candidate_rendering_digest": RENDERER_CONTRACT["candidate_rendering_digest"],
        "source_policy_identifiers": sorted(set(source_policy_identifiers)),
    }


def validate_training_descriptor(value: dict[str, Any]) -> None:
    if (
        set(value) != TRAINING_DESCRIPTOR_FIELDS
        or value.get("schema_version") != "v02-training-descriptor.v1"
    ):
        raise ValueError("training descriptor is not closed")
    if (
        not isinstance(value["record_count"], int)
        or value["record_count"] <= 0
        or value["split_counts"]
        != {
            "train": value["split_counts"].get("train"),
            "internal_dev": value["split_counts"].get("internal_dev"),
        }
        or sum(value["split_counts"].values()) != value["record_count"]
    ):
        raise ValueError("training descriptor counts invalid")
    for key in TRAINING_DESCRIPTOR_FIELDS - {
        "schema_version",
        "record_count",
        "split_counts",
        "source_policy_identifiers",
    }:
        _require_digest(value[key], key)
    if value["candidate_rendering_digest"] != RENDERER_CONTRACT["candidate_rendering_digest"]:
        raise ValueError("candidate renderer digest mismatch")


def diagnostic_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) < 50:
        raise ValueError("public development lane must contain at least 50 records")
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            b"saracura-v02-public-dev-order-v1"
            + len(row["opaque_record_id"].encode()).to_bytes(8, "big")
            + row["opaque_record_id"].encode()
        ).digest(),
    )[:50]


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, str]:
    """Apply the frozen development-only rule without receiving a held-out path."""
    if len(candidates) != 3 or {row.get("id") for row in candidates} != {
        item["id"] for item in CANDIDATE_GRID
    }:
        raise ValueError("selection requires exactly the frozen three candidates")
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        if set(row) != {
            "id",
            "coverage",
            "invalid_output_rate",
            "repeat_stability",
            "option_order_stability",
            "planned_top1_accuracy",
            "warm_p95_ms",
            "valid",
            "truncation_detected",
            "identity_match",
        }:
            raise ValueError("candidate report is not closed")
        if not all(
            isinstance(row[key], (int, float)) and not isinstance(row[key], bool)
            for key in (
                "coverage",
                "invalid_output_rate",
                "repeat_stability",
                "option_order_stability",
                "planned_top1_accuracy",
                "warm_p95_ms",
            )
        ):
            raise ValueError("candidate metric invalid")
        if (
            row["valid"] is True
            and row["truncation_detected"] is False
            and row["identity_match"] is True
            and row["coverage"] >= 0.98
            and row["invalid_output_rate"] == 0
            and row["repeat_stability"] == 1
            and row["option_order_stability"] >= 0.95
            and row["planned_top1_accuracy"] >= 0.4
        ):
            eligible.append(row)
    if not eligible:
        return {"status": "NO_RELEASE_CANDIDATE"}
    winner = sorted(
        eligible,
        key=lambda row: (
            -row["planned_top1_accuracy"],
            -row["option_order_stability"],
            row["warm_p95_ms"],
            row["id"],
        ),
    )[0]
    return {"status": "SELECTED", "candidate_id": str(winner["id"])}


def loopback_worker_plan() -> dict[str, Any]:
    """Closed configuration shared by candidate and Kev loopback workers.

    The actual model process is intentionally deferred to the authorised GPU
    phase; this plan is the offline proof that neither side can choose a
    different endpoint, batching, or measurement procedure.
    """
    evaluation = load_manifest()["evaluation"]
    assert isinstance(evaluation, dict)
    return dict(evaluation)


def validate_code_revisions(revisions: dict[str, str], replacement_revision: str) -> None:
    """Fail closed unless every pre-held-out artifact uses one replacement revision."""
    if not revisions or not HEX40.fullmatch(replacement_revision):
        raise ValueError("code revision is invalid")
    terminal = {"selection_commit", "NO_RELEASE_CANDIDATE", "held_out_access"}
    if terminal.intersection(revisions):
        raise ValueError("terminal evidence requires a successor protocol")
    if any(not HEX40.fullmatch(value) for value in revisions.values()):
        raise ValueError("code revision is invalid")
    if any(value != replacement_revision for value in revisions.values()):
        raise ValueError("code correction invalidates phases C, D, and E globally")


def validate_readiness(training: Path, sealed: Path, readiness: Path, receipt: Path) -> int:
    training_value = _load_closed(training)
    validate_training_descriptor(training_value)
    value = _load_closed(readiness)
    if (
        set(value) != READINESS_DESCRIPTOR_FIELDS
        or value.get("schema_version") != "v02-readiness-descriptor.v1"
    ):
        raise ValueError("readiness descriptor is not closed")
    for key in READINESS_DESCRIPTOR_FIELDS - {
        "schema_version",
        "all_records_candidate_preflight_passed",
        "all_records_kev_preflight_passed",
    }:
        _require_digest(value[key], key)
    if (
        value["training_descriptor_digest"] != hashlib.sha256(training.read_bytes()).hexdigest()
        or value["sealed_descriptor_digest"] != hashlib.sha256(sealed.read_bytes()).hexdigest()
        or value["disjointness_receipt_digest"] != hashlib.sha256(receipt.read_bytes()).hexdigest()
        or value["candidate_rendering_digest"] != RENDERER_CONTRACT["candidate_rendering_digest"]
        or value["all_records_candidate_preflight_passed"] is not True
        or value["all_records_kev_preflight_passed"] is not True
    ):
        raise ValueError("readiness bindings invalid")
    print("readiness valid")
    return 0


def _training_runtime() -> tuple[Any, Any, Any, Any]:
    """Load the opt-in ML stack only for an explicitly invoked local run."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        safetensors = importlib.import_module("safetensors.torch")
    except ImportError as exc:
        raise RuntimeError("install the v02-training extra before a local training run") from exc
    return torch, transformers, peft, safetensors


def candidate_config(candidate_id: str) -> dict[str, Any]:
    """Return the immutable QLoRA and pointer-head contract for one candidate."""
    manifest = load_manifest()
    candidate = next((item for item in CANDIDATE_GRID if item["id"] == candidate_id), None)
    if candidate is None:
        raise ValueError("candidate is outside the frozen grid")
    return {
        "base_model": manifest["architecture"]["base_model"],
        "base_revision": manifest["architecture"]["base_revision"],
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "lora_targets": "all-linear",
        "head_dim": 256,
        "trainable": "lora_and_pointer_readout_head",
        **candidate,
        **manifest["training"],
        **manifest["early_stopping"],
    }


def qwen35_text_config(model_config: Any) -> Any:
    """Reject a generic/vision fallback and return only the pinned text subconfig."""
    if getattr(model_config, "model_type", None) != "qwen3_5" or getattr(
        model_config, "architectures", None
    ) != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("base config is not the pinned Qwen3.5 conditional-generation class")
    text_config = getattr(model_config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if (
        getattr(text_config, "model_type", None) != "qwen3_5_text"
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise ValueError("base config has no valid Qwen3.5 text subconfiguration")
    return text_config


def build_qlora_components(
    base_directory: Path, candidate_id: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Build frozen-base QLoRA and a trainable 256-wide pointer head from local bytes.

    `local_files_only=True` is intentional: this source never downloads a base
    checkpoint.  Callers must supply a pre-acquired immutable local snapshot.
    """
    if (
        not base_directory.is_absolute()
        or not base_directory.is_dir()
        or base_directory.is_symlink()
    ):
        raise ValueError("base directory must be an absolute non-symlink local directory")
    torch, transformers, peft, _safetensors = _training_runtime()
    config = candidate_config(candidate_id)
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    config_loader = getattr(transformers, "AutoConfig", None)
    conditional_model = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if config_loader is None or conditional_model is None:
        raise RuntimeError("v0.2 requires Transformers 5.16.0 with Qwen3.5 support")
    model_config = config_loader.from_pretrained(
        str(base_directory), local_files_only=True, revision=config["base_revision"]
    )
    qwen35_text_config(model_config)
    base = conditional_model.from_pretrained(
        str(base_directory),
        local_files_only=True,
        revision=config["base_revision"],
        config=model_config,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    base.requires_grad_(False)
    text_model = getattr(getattr(base, "model", None), "language_model", None)
    if text_model is None or getattr(text_model, "config", None) is not qwen35_text_config(
        model_config
    ):
        raise ValueError("Qwen3.5 conditional model did not expose its pinned text path")
    lora = peft.get_peft_model(
        text_model,
        peft.LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["dropout"],
            bias="none",
            target_modules="all-linear",
            task_type=peft.TaskType.FEATURE_EXTRACTION,
        ),
    )

    class PointerReadoutHead(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.query = torch.nn.Linear(hidden_size, config["head_dim"], bias=False)
            self.key = torch.nn.Linear(hidden_size, config["head_dim"], bias=False)
            self.temperature = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        def forward(self, state: Any, options: Any) -> Any:
            scale = self.temperature.clamp_min(1e-6) * (config["head_dim"] ** -0.5)
            return (self.query(state).unsqueeze(1) * self.key(options)).sum(dim=-1) * scale

    hidden_size = qwen35_text_config(model_config).hidden_size
    return lora, PointerReadoutHead(hidden_size), config


def _last_token(hidden: Any, mask: Any) -> Any:
    """Select a representation without treating padded tokens as content."""
    positions = mask.to(dtype=hidden.dtype).sum(dim=1).to(dtype=hidden.dtype).long() - 1
    if bool((positions < 0).any()):
        raise ValueError("prepared token batch has an empty sequence")
    return hidden[range(hidden.shape[0]), positions]


def _score_token_batch(adapter: Any, pointer_head: Any, batch: dict[str, Any]) -> Any:
    """Run the actual frozen-base/LoRA/pointer forward pass for one token batch."""
    state = adapter(
        input_ids=batch["state_input_ids"], attention_mask=batch["state_attention_mask"]
    )
    state_vector = _last_token(state.last_hidden_state, batch["state_attention_mask"])
    option_ids = batch["option_input_ids"]
    option_mask = batch["option_attention_mask"]
    batch_size, option_count, sequence_length = option_ids.shape
    options = adapter(
        input_ids=option_ids.reshape(batch_size * option_count, sequence_length),
        attention_mask=option_mask.reshape(batch_size * option_count, sequence_length),
    )
    option_vectors = _last_token(
        options.last_hidden_state,
        option_mask.reshape(batch_size * option_count, sequence_length),
    ).reshape(batch_size, option_count, -1)
    scores = pointer_head(state_vector, option_vectors)
    return scores.masked_fill(~batch["option_present"], float("-inf"))


def _prepared_examples(path: Path) -> list[dict[str, Any]]:
    """Read a private, local-only token representation; never tokenize/download here."""
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("prepared examples must be an absolute non-symlink local file")
    value = _load_closed(path).get("records")
    if not isinstance(value, list) or not value:
        raise ValueError("prepared examples require a non-empty records list")
    expected = {"split", "state_input_ids", "options_input_ids", "target", "stratum"}
    parsed: list[dict[str, Any]] = []
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != expected
            or row["split"] not in {"train", "internal_dev"}
        ):
            raise ValueError("prepared example is not closed")
        if not isinstance(row["stratum"], str) or not row["stratum"]:
            raise ValueError("prepared example stratum invalid")
        if not isinstance(row["state_input_ids"], list) or not row["state_input_ids"]:
            raise ValueError("prepared state tokens invalid")
        if (
            not isinstance(row["options_input_ids"], list)
            or not 2 <= len(row["options_input_ids"]) <= 8
        ):
            raise ValueError("prepared option tokens invalid")
        sequences = [row["state_input_ids"], *row["options_input_ids"]]
        if any(
            not isinstance(sequence, list)
            or not sequence
            or any(not isinstance(token, int) or token < 0 for token in sequence)
            for sequence in sequences
        ):
            raise ValueError("prepared token IDs invalid")
        if not isinstance(row["target"], int) or not 0 <= row["target"] < len(
            row["options_input_ids"]
        ):
            raise ValueError("prepared target invalid")
        parsed.append(row)
    if {row["split"] for row in parsed} != {"train", "internal_dev"}:
        raise ValueError("prepared examples require train and internal_dev lanes")
    return parsed


def _token_batches(
    torch: Any, records: list[dict[str, Any]], batch_size: int
) -> Iterable[dict[str, Any]]:
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        max_state = max(len(row["state_input_ids"]) for row in rows)
        max_options = max(len(row["options_input_ids"]) for row in rows)
        max_option_tokens = max(len(option) for row in rows for option in row["options_input_ids"])
        state_ids = torch.zeros((len(rows), max_state), dtype=torch.long)
        state_mask = torch.zeros_like(state_ids)
        option_ids = torch.zeros((len(rows), max_options, max_option_tokens), dtype=torch.long)
        option_mask = torch.zeros_like(option_ids)
        option_present = torch.zeros((len(rows), max_options), dtype=torch.bool)
        targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
        for index, row in enumerate(rows):
            state = torch.tensor(row["state_input_ids"], dtype=torch.long)
            state_ids[index, : len(state)] = state
            state_mask[index, : len(state)] = 1
            for option_index, option in enumerate(row["options_input_ids"]):
                tokens = torch.tensor(option, dtype=torch.long)
                option_ids[index, option_index, : len(tokens)] = tokens
                option_mask[index, option_index, : len(tokens)] = 1
                option_present[index, option_index] = True
        yield {
            "state_input_ids": state_ids,
            "state_attention_mask": state_mask,
            "option_input_ids": option_ids,
            "option_attention_mask": option_mask,
            "option_present": option_present,
            "targets": targets,
            "strata": [str(row["stratum"]) for row in rows],
        }


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def train_pointer_candidate(
    adapter: Any,
    pointer_head: Any,
    prepared_examples: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Optimize only LoRA and pointer tensors, selecting the earliest best dev epoch.

    Prepared examples contain already-rendered token IDs.  This keeps corpus
    text, tokenizer acquisition, and model download outside this offline source
    module while still providing the real QLoRA/pointer optimization loop.
    """
    torch, _transformers, _peft, _safetensors = _training_runtime()
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("candidate training requires an authorised CUDA worker")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    adapter.to(device)
    pointer_head.to(device)
    train_rows = [row for row in prepared_examples if row["split"] == "train"]
    dev_rows = [row for row in prepared_examples if row["split"] == "internal_dev"]
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    trainable.extend(
        parameter for parameter in pointer_head.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise ValueError("QLoRA/pointer construction yielded no trainable tensors")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["peak_learning_rate"]),
        weight_decay=config["weight_decay"],
    )
    microbatch = min(4, len(train_rows))
    accumulation = max(1, config["effective_batch_size"] // microbatch)
    total_steps = max(
        1, ((len(train_rows) + microbatch - 1) // microbatch) * config["max_epochs"] // accumulation
    )
    warmup_steps = int(total_steps * config["warmup_fraction"])

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_adapter: dict[str, Any] | None = None
    best_head: dict[str, Any] | None = None
    for epoch in range(1, config["max_epochs"] + 1):
        adapter.train()
        pointer_head.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_number, batch in enumerate(
            _token_batches(torch, train_rows, microbatch), start=1
        ):
            score = _score_token_batch(adapter, pointer_head, _to_device(batch, device))
            loss = (
                torch.nn.functional.cross_entropy(score, batch["targets"].to(device)) / accumulation
            )
            loss.backward()
            if (
                batch_number % accumulation == 0
                or batch_number == (len(train_rows) + microbatch - 1) // microbatch
            ):
                torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clipping"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        adapter.eval()
        pointer_head.eval()
        correct: dict[str, list[int]] = {}
        with torch.no_grad():
            for batch in _token_batches(torch, dev_rows, microbatch):
                score = _score_token_batch(adapter, pointer_head, _to_device(batch, device))
                predicted = score.argmax(dim=1).cpu().tolist()
                for prediction, target, stratum in zip(
                    predicted, batch["targets"].tolist(), batch["strata"], strict=True
                ):
                    bucket = correct.setdefault(stratum, [0, 0])
                    bucket[0] += int(prediction == target)
                    bucket[1] += 1
        metric = sum(hit / count for hit, count in correct.values()) / len(correct)
        if metric > best_score:
            best_score, best_epoch, stale_epochs = metric, epoch, 0
            best_adapter = copy.deepcopy(adapter.state_dict())
            best_head = copy.deepcopy(pointer_head.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= load_manifest()["early_stopping"]["patience"]:
                break
    if best_adapter is None or best_head is None:
        raise RuntimeError("training produced no development checkpoint")
    adapter.load_state_dict(best_adapter)
    pointer_head.load_state_dict(best_head)
    return {"best_epoch": best_epoch, "internal_dev_stratified_macro_accuracy": best_score}


def _private_output_directory(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("artifact output must be a new absolute non-symlink directory")
    if path.exists():
        raise FileExistsError("create-only artifact output already exists")
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("artifact output parent must be a private existing directory")


def _file_digests(root: Path) -> dict[str, str]:
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if any(item.is_symlink() for item in files):
        raise ValueError("artifact contains symlink")
    return {
        str(item.relative_to(root)): hashlib.sha256(item.read_bytes()).hexdigest() for item in files
    }


def export_candidate_artifact(
    adapter: Any, pointer_head: Any, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Atomically create a LoRA-plus-pointer artifact and never export base tensors."""
    _private_output_directory(output)
    _torch, _transformers, _peft, safetensors = _training_runtime()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        adapter_path = temporary / "adapter"
        if not callable(getattr(adapter, "save_pretrained", None)):
            raise ValueError("QLoRA adapter cannot be safely exported")
        adapter.save_pretrained(str(adapter_path), safe_serialization=True)
        tensors = {
            f"pointer_head.{name}": value.detach().cpu().contiguous()
            for name, value in pointer_head.state_dict().items()
        }
        if not tensors or any(not name.startswith("pointer_head.") for name in tensors):
            raise ValueError("pointer head export is invalid")
        safetensors.save_file(tensors, str(temporary / "pointer_head.safetensors"))
        files = _file_digests(temporary)
        if not any(name.endswith(".safetensors") for name in files) or any(
            name.startswith(("model.", "pytorch_model", "base_model.")) for name in files
        ):
            raise ValueError("artifact has missing adapter tensors or base tensors")
        descriptor = {
            "schema_version": "v02-candidate-artifact.v2",
            "candidate_id": config["id"],
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "adapter_files_sha256": files,
            "base_tensors_exported": False,
        }
        create_json(temporary / "artifact.json", descriptor)
        os.rename(temporary, output)
    except BaseException:
        with suppress(FileNotFoundError):
            for item in sorted(temporary.rglob("*"), reverse=True):
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            temporary.rmdir()
        raise
    return descriptor


def verify_candidate_artifact(descriptor: dict[str, Any], output: Path) -> None:
    expected = {
        "schema_version",
        "candidate_id",
        "base_model",
        "base_revision",
        "adapter_files_sha256",
        "base_tensors_exported",
    }
    manifest = load_manifest()["architecture"]
    if (
        set(descriptor) != expected
        or descriptor["schema_version"] != "v02-candidate-artifact.v2"
        or descriptor["candidate_id"] not in {item["id"] for item in CANDIDATE_GRID}
        or descriptor["base_model"] != manifest["base_model"]
        or descriptor["base_revision"] != manifest["base_revision"]
        or descriptor["base_tensors_exported"] is not False
        or not isinstance(descriptor["adapter_files_sha256"], dict)
    ):
        raise ValueError("candidate artifact descriptor drifted")
    expected_files = descriptor["adapter_files_sha256"]
    if not expected_files or any(
        not isinstance(name, str) or not HEX64.fullmatch(digest)
        for name, digest in expected_files.items()
    ):
        raise ValueError("candidate artifact file digest invalid")
    if not output.is_absolute() or not output.is_dir() or output.is_symlink():
        raise ValueError("candidate artifact directory invalid")
    actual = _file_digests(output)
    actual.pop("artifact.json", None)
    if actual != expected_files or "pointer_head.safetensors" not in actual:
        raise ValueError("candidate artifact digest mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    check = sub.add_parser("validate-readiness")
    for name in ("training", "sealed", "readiness", "receipt"):
        check.add_argument(f"--{name}", type=Path, required=True)
    train = sub.add_parser("train-candidate")
    train.add_argument("--base-directory", type=Path, required=True)
    train.add_argument("--prepared-examples", type=Path, required=True)
    train.add_argument(
        "--candidate", choices=[item["id"] for item in CANDIDATE_GRID], required=True
    )
    train.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select")
    select.add_argument("--reports", type=Path, required=True)
    verify = sub.add_parser("verify-artifact")
    verify.add_argument("--artifact-directory", type=Path, required=True)
    for name in ("seal-readiness", "smoke", "evaluate"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "preflight":
        load_manifest()
        print("preflight valid")
        return 0
    if args.command == "validate-readiness":
        return validate_readiness(args.training, args.sealed, args.readiness, args.receipt)
    if args.command == "select":
        reports = _load_closed(args.reports).get("candidates")
        if not isinstance(reports, list):
            raise ValueError("candidate reports must be a closed candidates list")
        print(canonical_json_bytes(cast(JsonValue, select_candidate(reports))).decode("utf-8"))
        return 0
    if args.command == "verify-artifact":
        descriptor = _load_closed(args.artifact_directory / "artifact.json")
        verify_candidate_artifact(descriptor, args.artifact_directory)
        print("candidate artifact valid")
        return 0
    if args.command == "train-candidate":
        adapter, pointer_head, config = build_qlora_components(args.base_directory, args.candidate)
        outcome = train_pointer_candidate(
            adapter, pointer_head, _prepared_examples(args.prepared_examples), config
        )
        descriptor = export_candidate_artifact(adapter, pointer_head, args.output, config)
        print(canonical_json_bytes({"artifact": descriptor, "training": outcome}).decode("utf-8"))
        return 0
    raise ValueError(
        f"{args.command} requires the separately authorised live-phase command contract"
    )


if __name__ == "__main__":
    raise SystemExit(main())
