"""Closed local wiring for the P2 private training commands.

This module deliberately owns the operational composition only.  The corpus
verifier, optimizer and immutable checkpoint primitives remain independently
testable modules.  In particular, no command downloads a model, talks to a
host, or turns a private checkpoint into a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchmarks import v02_private_checkpoint as checkpoint
from benchmarks import v02_private_training as private
from benchmarks import v02_training, v02_training_loop
from saracura import contracts as saracura_contracts
from saracura.serialization import canonical_json_bytes


@dataclass(frozen=True)
class WorkflowRuntime:
    """The sole test seam; production callers do not expose it through CLI."""

    torch: Any
    peft: Any
    safetensors: Any
    device: Any
    train_factory: Callable[[private.PrivateAdmission, Path], tuple[Any, Any, dict[str, Any]]]
    reload_factory: Callable[[dict[str, Any], Path], tuple[Any, Any]]
    expected_tensor_count: int = 498


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _private_directory(path: Path) -> None:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.parent.is_dir()
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("private work directory must be below an existing 0700 parent")
    if path.exists():
        raise FileExistsError("private work directory already exists")


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    checkpoint._create_json(path, value)


def _source_wiring(root: Path, inputs: dict[str, Any]) -> None:
    """Close every source module used by this command composition.

    ``v02_private_training`` verifies its own direct dependencies.  The
    workflow imports checkpoint/export and the public response contracts too,
    so they must be part of the same reviewed inventory before an ML runtime is
    created.  This deliberately checks the live module paths rather than just
    trusting names in an operator-provided JSON file.
    """
    inventory_path = private._under(root, inputs["trainer_source_inventory_file"])
    inventory = private._json_file(inventory_path, label="trainer source inventory")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("trainer source inventory is invalid")
    repository = Path(__file__).resolve().parents[1]
    listed: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("trainer source inventory is invalid")
        path = item["path"]
        if not isinstance(path, str) or path in listed:
            raise ValueError("trainer source inventory is ambiguous")
        listed[path] = item
    modules = (
        checkpoint,
        private,
        v02_training,
        v02_training_loop,
        saracura_contracts,
        __import__("saracura.serialization", fromlist=["serialization"]),
    )
    paths = [Path(__file__).resolve()]
    for module in modules:
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            raise ValueError("imported workflow source has no file")
        paths.append(Path(source).resolve())
    for source in paths:
        if source.is_symlink() or not source.is_file() or source.parent == source:
            raise ValueError("imported workflow source is shadowed")
        try:
            relative = str(source.relative_to(repository))
        except ValueError as exc:
            raise ValueError("imported workflow source is outside repository") from exc
        entry = listed.get(relative)
        if (
            entry is None
            or entry.get("sha256") != _sha(source)
            or entry.get("bytes") != source.stat().st_size
        ):
            raise ValueError("imported workflow source is absent from reviewed inventory")


def verify_admission(
    *,
    root: Path,
    run_inputs_path: Path,
    admission_path: Path,
    machine_probe: Callable[[], dict[str, Any]] | None = None,
) -> private.PrivateAdmission:
    """Verify every non-ML gate before a runtime factory can be called."""
    inputs = private.load_run_inputs(root, run_inputs_path)
    _source_wiring(root, inputs)
    return private.verify_private_admission(
        root=root,
        run_inputs_path=run_inputs_path,
        admission_path=admission_path,
        machine_probe=machine_probe,
    )


def create_admission(
    *,
    root: Path,
    run_inputs_path: Path,
    output: Path,
    machine_probe: Callable[[], dict[str, Any]] | None = None,
) -> private.PrivateAdmission:
    inputs = private.load_run_inputs(root, run_inputs_path)
    _source_wiring(root, inputs)
    return private.create_private_admission(
        root=root, run_inputs_path=run_inputs_path, output=output, machine_probe=machine_probe
    )


def _production_runtime() -> WorkflowRuntime:
    torch, transformers, peft, safetensors = v02_training._training_runtime()
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("private production training requires an authorised CUDA runtime")

    def train_factory(
        proof: private.PrivateAdmission, base: Path
    ) -> tuple[Any, Any, dict[str, Any]]:
        adapter, head, config = v02_training.build_qlora_components(
            base, "c1-r8", admission=proof.proof
        )
        adapter.to(torch.device("cuda"))
        head.to(torch.device("cuda"))
        return adapter, head, config

    def reload_factory(config: dict[str, Any], base: Path) -> tuple[Any, Any]:
        candidate = v02_training.candidate_config("c1-r8")
        for key in ("lora_rank", "lora_alpha", "head_dim"):
            if config.get(key) != candidate[key]:
                raise ValueError("export config differs from the frozen c1 contract")
        model_config = transformers.AutoConfig.from_pretrained(str(base), local_files_only=True)
        text_config = v02_training.qwen35_text_config(model_config)
        model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            str(base), local_files_only=True, config=model_config, torch_dtype=torch.bfloat16
        )
        text = model.model.language_model
        text.requires_grad_(False)
        adapter = peft.get_peft_model(
            text,
            peft.LoraConfig(
                r=candidate["lora_rank"],
                lora_alpha=candidate["lora_alpha"],
                lora_dropout=candidate["dropout"],
                bias="none",
                target_modules=list(candidate["lora_targets"]),
                task_type=peft.TaskType.FEATURE_EXTRACTION,
            ),
        ).to(device=torch.device("cuda"))

        class PointerHead(torch.nn.Module):  # type: ignore[name-defined, misc]
            def __init__(self) -> None:
                super().__init__()
                self.query = torch.nn.Linear(
                    text_config.hidden_size, config["head_dim"], bias=False, dtype=torch.bfloat16
                )
                self.key = torch.nn.Linear(
                    text_config.hidden_size, config["head_dim"], bias=False, dtype=torch.bfloat16
                )

            def forward(self, state: Any, options: Any) -> Any:
                return (self.query(state).unsqueeze(1) * self.key(options)).sum(-1) * (
                    config["head_dim"] ** -0.5
                )

        return adapter, PointerHead().to(torch.device("cuda"))

    return WorkflowRuntime(
        torch, peft, safetensors, torch.device("cuda"), train_factory, reload_factory
    )


def _bindings(admission: private.PrivateAdmission, run_inputs: Path) -> str:
    return _canonical_sha(
        {
            "admission_sha256": admission.proof.admission_sha256,
            "capsule_sha256": admission.descriptor["rows_sha256"],
            "run_inputs_sha256": _sha(run_inputs),
            "seed": 20260929,
        }
    )


def _group_digests(runtime: WorkflowRuntime, adapter: Any, head: Any) -> tuple[str, str]:
    """Keep the LoRA and pointer-head update claims independently observable."""
    adapter_digest = v02_training_loop._tensor_digest(
        runtime.torch,
        (
            (name, parameter)
            for name, parameter in adapter.named_parameters()
            if parameter.requires_grad
        ),
    )
    head_digest = v02_training_loop._tensor_digest(
        runtime.torch,
        (
            (name, parameter)
            for name, parameter in head.named_parameters()
            if parameter.requires_grad
        ),
    )
    return adapter_digest, head_digest


def _claim(root: Path) -> dict[str, Any]:
    path = root / "private-run-claims" / "private-c1-r8-v1" / "claim.json"
    try:
        return private._json_file(path, label="private run claim")
    except ValueError as exc:
        raise RuntimeError("IMPLEMENTATION_BLOCKED: no valid step-zero claim") from exc


def _verified_export_admission(
    admission: private.PrivateAdmission, admission_path: Path
) -> dict[str, Any]:
    """Use the actual verified admission bytes plus its separately derived proof."""
    value = private._json_file(admission_path, label="private admission")
    if value.get("private_identity") != "private-c1-r8-v1":
        raise ValueError("private admission identity is invalid")
    return {**value, "admission_sha256": admission.proof.admission_sha256}


def _finite_trainables(runtime: WorkflowRuntime, adapter: Any, head: Any) -> bool:
    return all(
        bool(runtime.torch.isfinite(parameter.detach()).all())
        for module in (adapter, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _smoke_train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("smoke requires admitted train rows")
    longest = max(train, key=lambda row: (len(row["input_ids"]), str(row["identity_id"])))
    card8 = [row for row in train if len(row["end_option_indices"]) == 8]
    if not card8:
        raise ValueError("smoke requires an admitted cardinality-eight row")
    chosen = {str(longest["identity_id"]): longest, str(card8[0]["identity_id"]): card8[0]}
    for row in sorted(train, key=lambda item: str(item["identity_id"])):
        if len(chosen) >= 32:
            break
        chosen.setdefault(str(row["identity_id"]), row)
    return [chosen[key] for key in sorted(chosen)]


def _reload_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [row for row in rows if row["split"] in {"train", "internal_dev"}]
    train = [row for row in accepted if row["split"] == "train"]
    dev = [row for row in accepted if row["split"] == "internal_dev"]
    if not train or not dev:
        raise ValueError("reload samples must include train and internal_dev rows")
    chosen = {
        str(train[0]["identity_id"]): train[0],
        str(dev[0]["identity_id"]): dev[0],
    }
    for row in sorted(accepted, key=lambda item: str(item["identity_id"])):
        if len(chosen) >= 32:
            break
        chosen.setdefault(str(row["identity_id"]), row)
    return [chosen[key] for key in sorted(chosen)]


def _licenses(root: Path, inputs: dict[str, Any]) -> dict[str, Path]:
    sealer, _grant, _renderer, _tokenizer = private._validated_runtime_components(root, inputs)
    return {
        role: private._under(root, path)
        for role, path in cast(dict[str, Any], sealer["license_files"]).items()
    }


@contextmanager
def _identity(root: Path) -> Iterator[None]:
    """Acquire the identity lease before model construction or seeding."""
    with private.private_identity_lock(root):
        yield


def train_private(
    *,
    root: Path,
    run_inputs_path: Path,
    admission_path: Path,
    output: Path | None = None,
    resume: Path | None = None,
    runtime: WorkflowRuntime | None = None,
) -> dict[str, Any]:
    """Train or resume exactly the one admitted private identity.

    Runtime injection is intentionally a library-only seam for the CPU tiny
    Qwen/PEFT proof.  The command-line route always selects CUDA production.
    """
    if (output is None) == (resume is None):
        raise ValueError("train requires exactly one of output or resume")
    admission = verify_admission(
        root=root, run_inputs_path=run_inputs_path, admission_path=admission_path
    )
    inputs = private.load_run_inputs(root, run_inputs_path)
    binding = _bindings(admission, run_inputs_path)
    runtime = runtime or _production_runtime()
    with _identity(root):
        if output is not None:
            _private_directory(output)
            work, prior = output, None
        else:
            assert resume is not None
            claim = _claim(root)
            if (
                set(claim) != private._CLAIM_FIELDS
                or claim.get("private_identity") != "private-c1-r8-v1"
                or claim.get("admission_sha256") != admission.proof.admission_sha256
                or claim.get("run_inputs_sha256") != _sha(run_inputs_path)
                or claim.get("seed") != 20260929
                or claim.get("bindings_sha256") != binding
            ):
                raise RuntimeError("IMPLEMENTATION_BLOCKED: claim binding does not match")
            if (
                claim.get("work_directory") != str(resume)
                or not resume.is_absolute()
                or resume.is_symlink()
                or not resume.parent.is_dir()
                or stat.S_IMODE(resume.parent.stat().st_mode) != 0o700
            ):
                raise RuntimeError(
                    "IMPLEMENTATION_BLOCKED: resume work directory does not match claim"
                )
            candidate = resume / "resume-state.pt"
            if resume.exists() and (not resume.is_dir() or candidate.is_symlink()):
                raise RuntimeError("IMPLEMENTATION_BLOCKED: resume work directory is unsafe")
            prior = private.load_resume_state(
                candidate
                if candidate.is_file()
                else root / "private-run-claims" / "private-c1-r8-v1" / "initial-state.pt",
                torch=runtime.torch,
                bindings_sha256=binding,
            )
            v02_training_loop.validate_resume_state(prior)
            work = resume
            if not work.exists():
                work.mkdir(mode=0o700)
            if work.stat().st_uid != os.geteuid() or stat.S_IMODE(work.stat().st_mode) != 0o700:
                raise RuntimeError(
                    "IMPLEMENTATION_BLOCKED: resume work ownership or mode is unsafe"
                )
        # The lock and all admission bytes are now checked; only now can a model initialise.
        v02_training.configure_determinism(runtime.torch, 20260929)
        adapter, head, config = runtime.train_factory(admission, Path(inputs["base_directory"]))
        initial_adapter, initial_head = _group_digests(runtime, adapter, head)
        train_rows = [row for row in admission.rows if row["split"] == "train"]
        dev_rows = [row for row in admission.rows if row["split"] == "internal_dev"]
        published = False

        def boundary(state: dict[str, Any]) -> None:
            nonlocal published
            if not _finite_trainables(runtime, adapter, head):
                raise RuntimeError("optimization boundary contains non-finite trainable values")
            if not published and prior is None:
                private.publish_step_zero_claim(
                    root=root,
                    work_directory=work,
                    admission_sha256=admission.proof.admission_sha256,
                    run_inputs_path=run_inputs_path,
                    bindings_sha256=binding,
                    initial_state=state,
                    torch=runtime.torch,
                )
                work.mkdir(mode=0o700)
                published = True
            if prior is None and not published:
                raise RuntimeError("step-zero claim was not published")
            v02_training_loop.write_resume_state(runtime.torch, work / "resume-state.pt", state)

        result = v02_training_loop.run_optimization(
            torch=runtime.torch,
            adapter=adapter,
            pointer_head=head,
            train_rows=train_rows,
            dev_rows=dev_rows,
            config=config,
            device=runtime.device,
            token_batches=v02_training._token_batches,
            to_device=v02_training._to_device,
            score_batch=v02_training._score_token_batch,
            adapter_state=lambda model: v02_training._adapter_state_on_cpu(runtime.peft, model),
            load_adapter_state=runtime.peft.set_peft_model_state_dict,
            resume_state=prior,
            on_boundary=boundary,
            bindings_sha256=binding,
            private_identity="private-c1-r8-v1",
        )
        if result["interrupted"]:
            return result
        current_trainable = v02_training_loop._parameter_digest(
            runtime.torch, adapter, head, trainable=True
        )
        current_adapter, current_head = _group_digests(runtime, adapter, head)
        state = result["resume_state"]
        if (
            result["optimizer_steps"] <= 0
            or current_trainable == state["initial_trainable_sha256"]
            or current_adapter == initial_adapter
            or current_head == initial_head
            or state["base_parameter_sha256"]
            != v02_training_loop._parameter_digest(runtime.torch, adapter, head, trainable=False)
            or not _finite_trainables(runtime, adapter, head)
        ):
            raise RuntimeError("optimizer did not produce verified learned-only updates")
        metrics = {
            "artifact_id": "private-c1-r8-v1",
            "admission_sha256": admission.proof.admission_sha256,
            "private_identity": "private-c1-r8-v1",
            "seed": 20260929,
            "epochs_completed": state["epoch"] - 1,
            "optimizer_steps": result["optimizer_steps"],
            "train_rows": len(train_rows),
            "internal_dev_rows": len(dev_rows),
            "best_epoch": result["best_epoch"],
            "best_internal_dev_macro_accuracy": result["internal_dev_stratified_macro_accuracy"],
            "finite_updates": True,
            "base_unchanged": True,
            "initial_trainable_sha256": state["initial_trainable_sha256"],
            "duration_seconds": state["duration_seconds"],
            "peak_gpu_bytes": state["peak_gpu_bytes"],
        }
        export = work / "export"
        if export.exists():
            descriptor, result_receipt, _tensors, _config = checkpoint._read_export(
                export, runtime.safetensors
            )
            if (
                descriptor["admission_sha256"] != admission.proof.admission_sha256
                or result_receipt["initial_trainable_sha256"] != state["initial_trainable_sha256"]
            ):
                raise RuntimeError("existing export does not match completed private run")
            return descriptor
        return checkpoint.export_private_checkpoint(
            work=work,
            adapter=adapter,
            pointer_head=head,
            metrics=metrics,
            admission=_verified_export_admission(admission, admission_path),
            config={
                "adapter_name": "default",
                "lora_rank": config["lora_rank"],
                "lora_alpha": config["lora_alpha"],
                "head_dim": config["head_dim"],
            },
            licenses=_licenses(root, inputs),
            torch=runtime.torch,
            safetensors=runtime.safetensors,
        )


def reload_private(
    *,
    root: Path,
    run_inputs_path: Path,
    admission_path: Path,
    export: Path,
    output: Path,
    runtime: WorkflowRuntime | None = None,
) -> dict[str, Any]:
    admission = verify_admission(
        root=root, run_inputs_path=run_inputs_path, admission_path=admission_path
    )
    inputs = private.load_run_inputs(root, run_inputs_path)
    runtime = runtime or _production_runtime()
    descriptor, _result, _tensors, export_config = checkpoint._read_export(
        export, runtime.safetensors
    )
    if descriptor["admission_sha256"] != admission.proof.admission_sha256:
        raise ValueError("export admission differs from verified admission")
    receipt = checkpoint.fresh_reload_proof(
        export=export,
        factory=lambda: runtime.reload_factory(export_config, Path(inputs["base_directory"])),
        samples=_reload_rows(admission.rows),
        dev_rows=[row for row in admission.rows if row["split"] == "internal_dev"],
        torch=runtime.torch,
        safetensors=runtime.safetensors,
        expected_tensor_count=runtime.expected_tensor_count,
    )
    return checkpoint.publish_final_checkpoint(export=export, final=output, reload_receipt=receipt)


def verify_private_checkpoint(*, checkpoint_path: Path) -> dict[str, Any]:
    return checkpoint.verify_final_checkpoint(checkpoint_path)


def smoke_private(
    *,
    root: Path,
    run_inputs_path: Path,
    admission_path: Path,
    output: Path,
    runtime: WorkflowRuntime | None = None,
) -> dict[str, Any]:
    """Perform an admitted one-step local smoke without taking the c1 identity."""
    admission = verify_admission(
        root=root, run_inputs_path=run_inputs_path, admission_path=admission_path
    )
    runtime = runtime or _production_runtime()
    _private_directory(output)
    started = time.monotonic()
    v02_training.configure_determinism(runtime.torch, 20260929)
    inputs = private.load_run_inputs(root, run_inputs_path)
    adapter, head, config = runtime.train_factory(admission, Path(inputs["base_directory"]))
    load_seconds = time.monotonic() - started
    train = _smoke_train_rows(admission.rows)
    dev = [row for row in admission.rows if row["split"] == "internal_dev"]
    before_trainable = v02_training_loop._parameter_digest(
        runtime.torch, adapter, head, trainable=True
    )
    before_adapter, before_head = _group_digests(runtime, adapter, head)
    before_base = v02_training_loop._parameter_digest(runtime.torch, adapter, head, trainable=False)
    optimized = time.monotonic()
    result = v02_training_loop.run_optimization(
        torch=runtime.torch,
        adapter=adapter,
        pointer_head=head,
        train_rows=train,
        dev_rows=dev,
        config={**config, "max_epochs": 1},
        device=runtime.device,
        token_batches=v02_training._token_batches,
        to_device=v02_training._to_device,
        score_batch=v02_training._score_token_batch,
        adapter_state=lambda model: v02_training._adapter_state_on_cpu(runtime.peft, model),
        load_adapter_state=runtime.peft.set_peft_model_state_dict,
    )
    optimizer_step_seconds = time.monotonic() - optimized
    if result["interrupted"] or result["optimizer_steps"] != 1:
        raise RuntimeError("smoke did not execute exactly one optimizer step")
    after_trainable = v02_training_loop._parameter_digest(
        runtime.torch, adapter, head, trainable=True
    )
    after_adapter, after_head = _group_digests(runtime, adapter, head)
    after_base = v02_training_loop._parameter_digest(runtime.torch, adapter, head, trainable=False)
    if (
        after_trainable == before_trainable
        or after_adapter == before_adapter
        or after_head == before_head
        or after_base != before_base
        or not _finite_trainables(runtime, adapter, head)
    ):
        raise RuntimeError("smoke did not prove learned-only updates")
    work = output.parent / f".{output.name}.smoke-work"
    _private_directory(work)
    work.mkdir(mode=0o700)
    state = result["resume_state"]
    metrics = {
        "artifact_id": "private-c1-r8-v1",
        "admission_sha256": admission.proof.admission_sha256,
        "private_identity": "private-smoke-c1-r8-v1",
        "seed": 20260929,
        "epochs_completed": 1,
        "optimizer_steps": 1,
        "train_rows": len(train),
        "internal_dev_rows": len(dev),
        "best_epoch": result["best_epoch"],
        "best_internal_dev_macro_accuracy": result["internal_dev_stratified_macro_accuracy"],
        "finite_updates": True,
        "base_unchanged": True,
        "initial_trainable_sha256": state["initial_trainable_sha256"],
        "duration_seconds": result["resume_state"]["duration_seconds"],
        "peak_gpu_bytes": state["peak_gpu_bytes"],
    }
    checkpoint.export_private_checkpoint(
        work=work,
        adapter=adapter,
        pointer_head=head,
        metrics=metrics,
        admission={
            **_verified_export_admission(admission, admission_path),
            "private_identity": "private-smoke-c1-r8-v1",
        },
        config={
            "adapter_name": "default",
            "lora_rank": config["lora_rank"],
            "lora_alpha": config["lora_alpha"],
            "head_dim": config["head_dim"],
        },
        licenses=_licenses(root, inputs),
        torch=runtime.torch,
        safetensors=runtime.safetensors,
    )
    del adapter, head
    reloaded = time.monotonic()
    receipt = checkpoint.fresh_reload_proof(
        export=work / "export",
        factory=lambda: runtime.reload_factory(config, Path(inputs["base_directory"])),
        samples=_reload_rows(admission.rows),
        dev_rows=dev,
        torch=runtime.torch,
        safetensors=runtime.safetensors,
        expected_tensor_count=runtime.expected_tensor_count,
    )
    reload_seconds = time.monotonic() - reloaded
    receipt_path = work / "reload-receipt.json"
    _write_private_json(receipt_path, receipt)
    if state["peak_gpu_bytes"] <= 0 and getattr(runtime.device, "type", "cpu") == "cuda":
        raise RuntimeError("smoke did not measure CUDA memory")
    payload = {
        "schema_version": "v02-private-smoke.v1",
        "artifact_id": "private-c1-r8-v1",
        "admission_sha256": admission.proof.admission_sha256,
        "precision": "nf4-bf16",
        "train_rows": len(train),
        "internal_dev_rows": len(dev),
        "optimizer_steps": 1,
        "load_seconds": load_seconds,
        "optimizer_step_seconds": optimizer_step_seconds,
        "reload_seconds": reload_seconds,
        "duration_seconds": float(time.monotonic() - started),
        "peak_gpu_bytes": result["resume_state"]["peak_gpu_bytes"],
        "finite_updates": True,
        "base_unchanged": True,
        "lora_updated": True,
        "head_updated": True,
        "reload_receipt_sha256": _sha(receipt_path),
        "private_identity_consumed": False,
    }
    _write_private_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    admission = commands.add_parser("validate-private-admission")
    admission.add_argument("--root", type=Path, required=True)
    admission.add_argument("--run-inputs", type=Path, required=True)
    admission.add_argument("--output", type=Path, required=True)
    for name in ("smoke-private", "train-private", "reload-private"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--run-inputs", type=Path, required=True)
        command.add_argument("--admission", type=Path, required=True)
        if name == "train-private":
            group = command.add_mutually_exclusive_group(required=True)
            group.add_argument("--output", type=Path)
            group.add_argument("--resume", type=Path)
        else:
            command.add_argument("--output", type=Path, required=True)
        if name == "reload-private":
            command.add_argument("--export", type=Path, required=True)
    verify = commands.add_parser("verify-private-checkpoint")
    verify.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result: Any
    if args.command == "validate-private-admission":
        result = create_admission(
            root=args.root, run_inputs_path=args.run_inputs, output=args.output
        )
    elif args.command == "smoke-private":
        result = smoke_private(
            root=args.root,
            run_inputs_path=args.run_inputs,
            admission_path=args.admission,
            output=args.output,
        )
    elif args.command == "train-private":
        result = train_private(
            root=args.root,
            run_inputs_path=args.run_inputs,
            admission_path=args.admission,
            output=args.output,
            resume=args.resume,
        )
    elif args.command == "reload-private":
        result = reload_private(
            root=args.root,
            run_inputs_path=args.run_inputs,
            admission_path=args.admission,
            export=args.export,
            output=args.output,
        )
    else:
        result = verify_private_checkpoint(checkpoint_path=args.checkpoint)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
