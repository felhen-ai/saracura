"""Offline, deterministic corpus and sealed-plan planner for Phase B1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import stat
import subprocess
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from saracura.serialization import canonical_json_bytes

MANIFEST_PATH = Path(__file__).parent / "manifests" / "v02-distillation-safe-corpus.v1.json"

SEED_TRAINING = 20260929
SEED_SEALED = 20260930
NAMESPACE = "saracura-v02-cleanroom-v1"

DOMAINS = [
    "email_triage",
    "customer_support",
    "finance",
    "accounting",
    "commerce",
    "operations",
    "scheduling",
    "document_routing",
    "browser_action",
    "security_triage",
    "content_moderation",
    "personal_productivity",
]

OPTION_COUNTS = [2, 3, 4, 5, 6, 7, 8]

LEDGER_SCHEMA = "v02-offline-ledger.v1"
RECEIPT_SCHEMA = "v02-aggregate-receipt.v1"
LOCAL_GATES = frozenset(
    {
        "schema_valid",
        "privacy_valid",
        "fictionality_valid",
        "length_valid",
        "duplicate_valid",
        "renderer_valid",
        "exclusive_options_valid",
        "ambiguity_free",
    }
)
_MODEL_ROLE_NAMES = frozenset(
    {
        "training_author",
        "independent_reviewer",
        "sealed_author",
        "sealed_annotator_a",
        "sealed_annotator_b",
        "sealed_adjudicator",
    }
)

MODEL_ROLES = {
    "training_author": {
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "family": "Qwen",
        "license": "Apache-2.0",
    },
    "independent_reviewer": {
        "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "family": "Mistral",
        "license": "Apache-2.0",
    },
    "sealed_author": {
        "model": "microsoft/Phi-4-mini-instruct",
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "family": "Microsoft",
        "license": "MIT",
    },
    "sealed_annotator_a": {
        "model": "ibm-granite/granite-3.3-8b-instruct",
        "revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "family": "IBM Granite",
        "license": "Apache-2.0",
    },
    "sealed_annotator_b": {
        "model": "allenai/OLMo-2-1124-7B-Instruct",
        "revision": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
        "family": "Allenai OLMo",
        "license": "Apache-2.0",
    },
    "sealed_adjudicator": {
        "model": "HuggingFaceTB/SmolLM3-3B",
        "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "family": "HuggingFace TB",
        "license": "Apache-2.0",
    },
}

RENDERER_CONTRACT: dict[str, Any] = {
    "kev_rendering_function_digest": (
        "9f42035579e68f6c0e535df2e107b189314b9c93f3899b442855a9dd4e6a9c66"
    ),
    "candidate_renderer_source_revision": "9c41005b2180347c3c646dfc9e50c4428483ec6b",
    "candidate_renderer_source_sha256": (
        "d78fab645f29513a62816e594d10296b02bf166ba77835c55db5eda80c7f1978"
    ),
    "candidate_rendering_digest": (
        "0f5592b54f096ac0328b45d69e5a74b9cb579a804f2349fc4ceea9b1f8a40e40"
    ),
    "candidate_renderer_config": {
        "max_state": 384,
        "max_branch": 1024,
        "max_packed": 2048,
        "strict": True,
        "option_isolation": False,
        "special_embeddings": False,
        "lora_targets": "all",
        "head_dim": 256,
    },
    "max_rendered_input_tokens": 512,
    "truncation_disabled": True,
    "preflight_before_every_call": True,
}


def _load_manifest() -> dict[str, Any]:
    raw = MANIFEST_PATH.read_bytes()
    result: dict[str, Any] = json.loads(raw)
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def validate_protocol() -> int:
    from benchmarks.validate_manifests import validate_v02_distillation_safe_corpus

    raw = MANIFEST_PATH.read_bytes()
    try:
        json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest has invalid JSON: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"manifest has duplicate keys: {exc}") from exc

    manifest = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    if raw != canonical_json_bytes(cast(Any, manifest)) + b"\n":
        raise ValueError("manifest bytes are not canonical")
    validate_v02_distillation_safe_corpus(MANIFEST_PATH)
    _validate_closed_schema(manifest)
    _validate_frozen_values(manifest)
    _validate_models(manifest)
    _validate_runtime(manifest)
    _validate_renderer(manifest)
    _validate_counts(manifest)
    _validate_terminal_values(manifest)
    _validate_pilot(manifest)
    print("protocol valid")
    return 0


def _validate_closed_schema(manifest: dict[str, Any]) -> None:
    required = frozenset(
        {
            "schema_version",
            "protocol_digest",
            "model_roles",
            "runtime_policy",
            "renderer_contracts",
            "corpus_plan",
            "sealed_plan",
            "pilot",
            "file_schemas",
            "terminal_values",
            "privacy_boundary",
        }
    )
    if set(manifest) != required:
        raise ValueError(f"manifest schema is not closed: got {sorted(set(manifest))}")
    if manifest.get("schema_version") != "v02-distillation-safe-corpus.v1":
        raise ValueError("manifest schema_version mismatch")


def _validate_frozen_values(manifest: dict[str, Any]) -> None:
    if manifest.get("protocol_digest") != "b1-offline-plan-r4":
        raise ValueError("protocol_digest mismatch")
    runtime = manifest.get("runtime_policy", {})
    if runtime.get("serving_runtime") != "vLLM":
        raise ValueError("runtime_policy serving_runtime mismatch")
    if runtime.get("quantization") != "bitsandbytes-nf4":
        raise ValueError("runtime_policy quantization mismatch")
    if runtime.get("compute") != "bf16":
        raise ValueError("runtime_policy compute mismatch")
    if runtime.get("no_hosted_api") is not True:
        raise ValueError("runtime_policy no_hosted_api must be true")
    renderer = manifest.get("renderer_contracts", {})
    if (
        renderer.get("kev_rendering_function_digest")
        != RENDERER_CONTRACT["kev_rendering_function_digest"]
    ):
        raise ValueError("renderer kev digest mismatch")
    if (
        renderer.get("candidate_renderer_source_revision")
        != RENDERER_CONTRACT["candidate_renderer_source_revision"]
    ):
        raise ValueError("renderer source revision mismatch")
    if (
        renderer.get("candidate_renderer_source_sha256")
        != RENDERER_CONTRACT["candidate_renderer_source_sha256"]
    ):
        raise ValueError("renderer source sha256 mismatch")
    if (
        renderer.get("candidate_rendering_digest")
        != RENDERER_CONTRACT["candidate_rendering_digest"]
    ):
        raise ValueError("candidate rendering digest mismatch")
    config: dict[str, Any] = renderer.get("candidate_renderer_config", {})
    for key, value in RENDERER_CONTRACT["candidate_renderer_config"].items():
        if config.get(key) != value:
            raise ValueError(f"renderer config {key} mismatch")
    if renderer.get("max_rendered_input_tokens") != 512:
        raise ValueError("renderer max_rendered_input_tokens mismatch")
    if renderer.get("truncation_disabled") is not True:
        raise ValueError("renderer truncation_disabled must be true")
    corpus = manifest.get("corpus_plan", {})
    if corpus.get("namespace") != NAMESPACE:
        raise ValueError("corpus namespace mismatch")
    if corpus.get("total_slots") != 1600:
        raise ValueError("corpus total_slots mismatch")
    if corpus.get("seed") != SEED_TRAINING:
        raise ValueError("corpus seed mismatch")
    sealed = manifest.get("sealed_plan", {})
    if sealed.get("seed") != SEED_SEALED:
        raise ValueError("sealed seed mismatch")
    if sealed.get("total_identities") != 140:
        raise ValueError("sealed total_identities mismatch")
    if sealed.get("permutation_selection_seed") != "saracura-v02-sealed-permutation-v1":
        raise ValueError("sealed permutation seed mismatch")
    if sealed.get("permutation_subset_size") != 50:
        raise ValueError("sealed permutation subset size mismatch")


def _validate_models(manifest: dict[str, Any]) -> None:
    roles = manifest.get("model_roles", {})
    if set(roles) != set(MODEL_ROLES.keys()):
        raise ValueError("model_roles schema mismatch")
    for key, expected in MODEL_ROLES.items():
        actual = roles[key]
        for field, value in expected.items():
            if actual.get(field) != value:
                raise ValueError(f"model {key} {field} mismatch")


def _validate_runtime(manifest: dict[str, Any]) -> None:
    runtime = manifest.get("runtime_policy", {})
    expected_keys = frozenset(
        {
            "serving_runtime",
            "container",
            "quantization",
            "compute",
            "gpu_class",
            "gpu_fallback",
            "gpu_forbidden",
            "sequence",
            "no_hosted_api",
        }
    )
    if set(runtime) != expected_keys:
        raise ValueError("runtime_policy schema mismatch")


def _validate_renderer(manifest: dict[str, Any]) -> None:
    renderer = manifest.get("renderer_contracts", {})
    expected_keys = frozenset(
        {
            "kev_rendering_function_digest",
            "candidate_renderer_source_revision",
            "candidate_renderer_source_sha256",
            "candidate_rendering_digest",
            "candidate_renderer_config",
            "max_rendered_input_tokens",
            "truncation_disabled",
            "preflight_before_every_call",
        }
    )
    if set(renderer) != expected_keys:
        raise ValueError("renderer_contracts schema mismatch")
    config = renderer.get("candidate_renderer_config", {})
    expected_config_keys = frozenset(
        {
            "max_state",
            "max_branch",
            "max_packed",
            "strict",
            "option_isolation",
            "special_embeddings",
            "lora_targets",
            "head_dim",
        }
    )
    if set(config) != expected_config_keys:
        raise ValueError("renderer config schema mismatch")


def _validate_counts(manifest: dict[str, Any]) -> None:
    corpus = manifest.get("corpus_plan", {})
    splits = corpus.get("splits", {})
    if set(splits) != {"train", "internal_dev"}:
        raise ValueError("corpus splits mismatch")
    train = splits["train"]
    internal_dev = splits["internal_dev"]
    if train.get("total") != 1360 or internal_dev.get("total") != 240:
        raise ValueError("corpus split totals mismatch")
    if train.get("locale_pt_br") != 816 or train.get("locale_english") != 544:
        raise ValueError("train locale totals mismatch")
    if internal_dev.get("locale_pt_br") != 144 or internal_dev.get("locale_english") != 96:
        raise ValueError("internal_dev locale totals mismatch")
    if train.get("bilingual_pairs") != 255 or internal_dev.get("bilingual_pairs") != 45:
        raise ValueError("bilingual pair totals mismatch")
    if corpus.get("total_bilingual_pairs") != 300:
        raise ValueError("total bilingual pairs mismatch")
    if set(corpus.get("domains", [])) != set(DOMAINS):
        raise ValueError("domains mismatch")
    if corpus.get("option_counts") != OPTION_COUNTS:
        raise ValueError("option_counts mismatch")
    locales = corpus.get("locale_totals", {})
    if locales.get("pt_br") != 960 or locales.get("english") != 640:
        raise ValueError("locale totals mismatch")
    floor = corpus.get("acceptance_floor", {})
    expected_floor = {
        "min_accepted_rows": 1200,
        "min_train_accepted": 1020,
        "min_internal_dev_accepted": 180,
        "min_ptbr_pct": 0.6,
        "min_english_pct": 0.2,
        "min_train_cell": 20,
        "min_internal_dev_cell": 10,
        "min_train_domain": 60,
        "min_internal_dev_domain": 10,
        "min_complete_bilingual_pairs": 120,
    }
    for key, value in expected_floor.items():
        if floor.get(key) != value:
            raise ValueError(f"acceptance_floor {key} mismatch")


def _validate_terminal_values(manifest: dict[str, Any]) -> None:
    terminals = manifest.get("terminal_values", {})
    expected = {
        "READY": "all-artifacts-sealed-validated-review-passed",
        "NO_GO": "feasibility-stop-proves-acceptance-floor-unreachable",
        "BLOCKED_DATA_RIGHTS": "rights-license-readiness-receipt-negative",
        "BLOCKED_SEALED_TEST": "frozen-sealed-plan-cannot-produce-compliant-capsule",
        "BLOCKED_REVIEW": "independent-review-returns-negative",
        "BLOCKED_RUNTIME": "pinned-models-cannot-pass-frozen-self-hosted-runtime-preflight",
    }
    if set(terminals) != set(expected):
        raise ValueError("terminal_values schema mismatch")
    for key, value in expected.items():
        if terminals.get(key) != value:
            raise ValueError(f"terminal {key} mismatch")


def _validate_pilot(manifest: dict[str, Any]) -> None:
    pilot = manifest.get("pilot", {})
    if pilot.get("training_slots") != 140:
        raise ValueError("pilot training_slots mismatch")
    if pilot.get("train_slots") != 119:
        raise ValueError("pilot train_slots mismatch")
    if pilot.get("internal_dev_slots") != 21:
        raise ValueError("pilot internal_dev_slots mismatch")
    if pilot.get("complete_bilingual_pairs") != 21:
        raise ValueError("pilot bilingual_pairs mismatch")
    if pilot.get("slots_per_locale_cardinality_cell") != 10:
        raise ValueError("pilot slots_per_locale_cardinality_cell mismatch")


def _safe_output_parent(output_parent: Path) -> None:
    if not output_parent.is_absolute():
        raise ValueError("output-parent must be an absolute path")
    if not output_parent.exists():
        raise ValueError("output-parent must be an existing directory")
    if not output_parent.is_dir():
        raise ValueError("output-parent must be an existing directory")
    current = output_parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("output-parent must not contain a symlink")
        current = current.parent
    if os.geteuid() != output_parent.stat().st_uid:
        raise ValueError("output-parent must be owned by the current user")
    if stat.S_IMODE(output_parent.stat().st_mode) != 0o700:
        raise ValueError("output-parent must have exact permission mode 0700")
    resolved_parent = output_parent.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if resolved_parent == repo_root or repo_root in resolved_parent.parents:
        raise ValueError("output-parent must be outside the repository")
    git_dir = _git_common_dir(repo_root)
    if resolved_parent == git_dir or git_dir in resolved_parent.parents:
        raise ValueError("output-parent must be outside Git")
    _check_worktrees(output_parent)


def _git_common_dir(repo_root: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("unable to resolve Git common directory")
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = repo_root / common
    return common.resolve()


def _is_within_worktree(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == "true":
            return True
    except OSError:
        pass
    return False


def _check_worktrees(output_parent: Path) -> None:
    current = output_parent
    seen: set[str] = set()
    while current != current.parent:
        key = str(current.resolve())
        if key in seen:
            break
        seen.add(key)
        if _is_within_worktree(current):
            raise ValueError("output-parent must be outside all worktrees")
        current = current.parent


def _atomic_write(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("temporary output path already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("output path already exists") from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _opaque_slot_id(namespace: str, index: int, rng: random.Random | None = None) -> str:
    token = f"{rng.randint(0, 2**64 - 1):016x}" if rng is not None else secrets.token_hex(8)
    raw = f"{namespace}:{index}:{token}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _balanced_allocation(total: int, num_buckets: int) -> list[int]:
    base = total // num_buckets
    remainder = total % num_buckets
    return [base + (1 if i < remainder else 0) for i in range(num_buckets)]


def _validated_corpus_taxonomy() -> tuple[list[str], list[int], list[str], dict[str, list[str]]]:
    manifest = _load_manifest()
    _validate_closed_schema(manifest)
    _validate_frozen_values(manifest)
    _validate_counts(manifest)
    corpus = cast(dict[str, Any], manifest["corpus_plan"])
    mapping = cast(dict[str, list[str]], corpus["domain_scenario_map"])
    return (
        list(cast(list[str], corpus["domains"])),
        list(cast(list[int], corpus["option_counts"])),
        list(cast(list[str], corpus["criterion_roles"])),
        {domain: list(scenarios) for domain, scenarios in mapping.items()},
    )


def _generate_training_plan() -> dict[str, Any]:
    rng = random.Random(SEED_TRAINING)
    domains, option_counts, criterion_roles, domain_scenario_map = _validated_corpus_taxonomy()

    slot_rows: list[dict[str, Any]] = []
    split_config: dict[str, dict[str, int]] = {
        "train": {"pt_br": 816, "english": 544},
        "internal_dev": {"pt_br": 144, "english": 96},
    }
    pair_config: dict[str, int] = {"train": 255, "internal_dev": 45}

    for split in ["train", "internal_dev"]:
        for locale in ["pt_br", "english"]:
            locale_total = split_config[split][locale]
            per_oc = _balanced_allocation(locale_total, len(option_counts))
            domain_sequence: list[str] = []
            for domain, count in zip(
                domains,
                _balanced_allocation(locale_total, len(domains)),
                strict=True,
            ):
                domain_sequence.extend([domain] * count)
            rng.shuffle(domain_sequence)
            domain_cursor = 0

            for oc_idx, oc in enumerate(option_counts):
                cell_total = per_oc[oc_idx]
                gold_dist = _balanced_allocation(cell_total, oc)
                gold_sequence: list[int] = []
                for pos, count in enumerate(gold_dist):
                    gold_sequence.extend([pos] * count)
                rng.shuffle(gold_sequence)
                for slot_in_cell in range(cell_total):
                    slot_rows.append(
                        {
                            "split": split,
                            "locale": locale,
                            "domain": domain_sequence[domain_cursor + slot_in_cell],
                            "option_count": oc,
                            "gold_position": gold_sequence[slot_in_cell],
                            "bilingual_pair_id": None,
                            "in_pilot": False,
                        }
                    )
                domain_cursor += cell_total

    pilot_indices = _select_training_pilot(slot_rows)
    for index in pilot_indices:
        slot_rows[index]["in_pilot"] = True
    _assign_bilingual_pairs(slot_rows, pair_config)

    for i, row in enumerate(slot_rows):
        row["slot_id"] = _opaque_slot_id(NAMESPACE, i, rng)
    _assign_semantic_targets(slot_rows, criterion_roles, domain_scenario_map)

    slot_ids = [r["slot_id"] for r in slot_rows]
    split_assignments: list[str] = [r["split"] for r in slot_rows]
    locale_assignments: list[str] = [r["locale"] for r in slot_rows]
    domain_assignments: list[str] = [r["domain"] for r in slot_rows]
    option_count_assignments: list[int] = [r["option_count"] for r in slot_rows]
    gold_positions: list[int] = [r["gold_position"] for r in slot_rows]
    bilingual_pair_ids = [r["bilingual_pair_id"] for r in slot_rows]
    pilot_slots = [r["slot_id"] for r in slot_rows if r["in_pilot"]]

    train_pilot_count = sum(1 for r in slot_rows if r["in_pilot"] and r["split"] == "train")
    internal_dev_pilot_count = sum(
        1 for r in slot_rows if r["in_pilot"] and r["split"] == "internal_dev"
    )
    pilot_pair_count = len(
        set(
            pair_id
            for pair_id, r in zip(bilingual_pair_ids, slot_rows, strict=True)
            if r["in_pilot"] and pair_id is not None
        )
    )

    domain_totals: dict[str, int] = {}
    for d in domains:
        domain_totals[d] = sum(1 for d2 in domain_assignments if d2 == d)

    oc_totals: dict[int, int] = {}
    for oc in option_count_assignments:
        oc_totals[oc] = oc_totals.get(oc, 0) + 1

    split_locale_oc: dict[tuple[str, str], dict[int, int]] = {}
    for i in range(len(slot_rows)):
        key = (split_assignments[i], locale_assignments[i])
        oc = option_count_assignments[i]
        cell = split_locale_oc.setdefault(key, {})
        cell[oc] = cell.get(oc, 0) + 1

    gold_cell_balance: dict[tuple[str, str, int], dict[int, int]] = {}
    for i in range(len(slot_rows)):
        gold_key = (split_assignments[i], locale_assignments[i], option_count_assignments[i])
        gp = gold_positions[i]
        cell = gold_cell_balance.setdefault(gold_key, {})
        cell[gp] = cell.get(gp, 0) + 1

    return {
        "schema_version": "v02-plan.v1",
        "lane": "training",
        "namespace": NAMESPACE,
        "seed": SEED_TRAINING,
        "slot_ids": slot_ids,
        "slots": [
            {
                "slot_id": row["slot_id"],
                "split": row["split"],
                "locale": row["locale"],
                "domain": row["domain"],
                "option_count": row["option_count"],
                "gold_position": row["gold_position"],
                "bilingual_pair_id": row["bilingual_pair_id"],
                "family_id": row["family_id"],
                "scenario_code": row["scenario_code"],
                "criterion_roles": row["criterion_roles"],
                "in_pilot": row["in_pilot"],
            }
            for row in slot_rows
        ],
        "split_totals": {
            "train": sum(1 for s in split_assignments if s == "train"),
            "internal_dev": sum(1 for s in split_assignments if s == "internal_dev"),
        },
        "locale_totals": {
            "pt_br": sum(1 for s in locale_assignments if s == "pt_br"),
            "english": sum(1 for s in locale_assignments if s == "english"),
        },
        "pair_totals": {
            "train": len(set(p for p in bilingual_pair_ids if p and p.startswith("bp-train-"))),
            "internal_dev": len(
                set(p for p in bilingual_pair_ids if p and p.startswith("bp-internal_dev-"))
            ),
        },
        "domain_allocation": domain_totals,
        "option_count_allocation": {str(oc): oc_totals[oc] for oc in option_counts},
        "split_locale_option_count_allocation": {
            f"{sp}:{lo}": {str(oc): counts[oc] for oc in option_counts}
            for (sp, lo), counts in split_locale_oc.items()
        },
        "gold_position_allocation": {
            f"{sp}:{lo}:{oc}": {str(position): count for position, count in sorted(golds.items())}
            for (sp, lo, oc), golds in gold_cell_balance.items()
        },
        "pilot_prefix": {
            "slot_ids": pilot_slots,
            "count": len(pilot_slots),
            "train_count": train_pilot_count,
            "internal_dev_count": internal_dev_pilot_count,
            "complete_bilingual_pairs": pilot_pair_count,
            "slots_per_locale_cardinality_cell": 10,
        },
    }


def _matching_pairs(
    slot_rows: list[dict[str, Any]],
    *,
    split: str,
    option_count: int | None,
    pilot: bool | None,
) -> list[tuple[int, int]]:
    buckets: dict[tuple[str, int], dict[str, list[int]]] = {}
    for index, row in enumerate(slot_rows):
        if row["split"] != split or (
            option_count is not None and row["option_count"] != option_count
        ):
            continue
        if pilot is not None and row["in_pilot"] is not pilot:
            continue
        if row["bilingual_pair_id"] is not None:
            continue
        key = (cast(str, row["domain"]), cast(int, row["option_count"]))
        buckets.setdefault(key, {"pt_br": [], "english": []})[cast(str, row["locale"])].append(
            index
        )
    pairs: list[tuple[int, int]] = []
    for key in sorted(buckets):
        members = buckets[key]
        pairs.extend(zip(members["pt_br"], members["english"], strict=False))
    return pairs


def _select_training_pilot(slot_rows: list[dict[str, Any]]) -> set[int]:
    """Choose exact per-cell quotas while reserving structurally matched pairs."""
    selected: set[int] = set()
    quotas = {
        "train": {"pt_br": 8, "english": 9, "pairs": 2},
        "internal_dev": {"pt_br": 2, "english": 1, "pairs": 1},
    }
    for split, quota in quotas.items():
        for option_count in OPTION_COUNTS:
            matches = _matching_pairs(slot_rows, split=split, option_count=option_count, pilot=None)
            chosen = matches[: quota["pairs"]]
            if len(chosen) != quota["pairs"]:
                raise ValueError("insufficient structurally matched pilot pairs")
            for left, right in chosen:
                selected.update((left, right))
            for locale in ("pt_br", "english"):
                already = sum(
                    slot_rows[index]["locale"] == locale
                    for index in selected
                    if slot_rows[index]["split"] == split
                    and slot_rows[index]["option_count"] == option_count
                )
                needed = quota[locale] - already
                candidates = [
                    index
                    for index, row in enumerate(slot_rows)
                    if row["split"] == split
                    and row["locale"] == locale
                    and row["option_count"] == option_count
                    and index not in selected
                ]
                selected.update(candidates[:needed])
    return selected


def _assign_bilingual_pairs(slot_rows: list[dict[str, Any]], pair_config: dict[str, int]) -> None:
    """Pair one row per locale with a shared split, domain and cardinality."""
    forced_per_option_count = {"train": 2, "internal_dev": 1}
    for split, pair_count in pair_config.items():
        pair_number = 0
        for option_count in OPTION_COUNTS:
            matches = _matching_pairs(slot_rows, split=split, option_count=option_count, pilot=True)
            for left, right in matches[: forced_per_option_count[split]]:
                pair_id = f"bp-{split}-{pair_number:04d}"
                slot_rows[left]["bilingual_pair_id"] = pair_id
                slot_rows[right]["bilingual_pair_id"] = pair_id
                pair_number += 1

        remaining = pair_count - pair_number
        candidates = _matching_pairs(slot_rows, split=split, option_count=None, pilot=False)
        random.Random(f"{SEED_TRAINING}:{split}:pairs").shuffle(candidates)
        if len(candidates) < remaining:
            raise ValueError("insufficient structurally matched bilingual pairs")
        for left, right in candidates[:remaining]:
            pair_id = f"bp-{split}-{pair_number:04d}"
            slot_rows[left]["bilingual_pair_id"] = pair_id
            slot_rows[right]["bilingual_pair_id"] = pair_id
            pair_number += 1


def _assign_semantic_targets(
    slot_rows: list[dict[str, Any]],
    criterion_roles: list[str],
    domain_scenario_map: dict[str, list[str]],
) -> None:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in slot_rows:
        pair_id = cast(str | None, row["bilingual_pair_id"])
        family_id = pair_id or f"single-{row['slot_id']}"
        row["family_id"] = family_id
        families.setdefault(family_id, []).append(row)
    distractor_pool = [role for role in criterion_roles if role != "matches_rule"]
    for family_id, members in families.items():
        exemplar = members[0]
        domain = cast(str, exemplar["domain"])
        option_count = cast(int, exemplar["option_count"])
        family_rng = random.Random(f"{SEED_TRAINING}:{family_id}:semantic")
        scenario = family_rng.choice(domain_scenario_map[domain])
        distractors = family_rng.sample(distractor_pool, option_count - 1)
        for row in members:
            roles = list(distractors)
            roles.insert(cast(int, row["gold_position"]), "matches_rule")
            row["scenario_code"] = scenario
            row["criterion_roles"] = roles


def _generate_sealed_plan() -> dict[str, Any]:
    rng = random.Random(SEED_SEALED)
    identities: list[dict[str, Any]] = []

    for oc in OPTION_COUNTS:
        for _i in range(3):
            identities.append(
                {
                    "identity_id": _opaque_slot_id("sealed", len(identities), rng),
                    "locale": "pt_br",
                    "option_count": oc,
                    "permutation_rank": len(identities),
                }
            )
    for oc in OPTION_COUNTS:
        for _i in range(17):
            identities.append(
                {
                    "identity_id": _opaque_slot_id("sealed", len(identities), rng),
                    "locale": "pt_br",
                    "option_count": oc,
                    "permutation_rank": len(identities),
                }
            )

    pilot_identities = [dict(identity) for identity in identities[:21]]

    return {
        "schema_version": "v02-plan.v1",
        "lane": "sealed",
        "namespace": NAMESPACE,
        "seed": SEED_SEALED,
        "identities": identities,
        "identity_count": len(identities),
        "identities_per_option_count": 20,
        "option_counts": OPTION_COUNTS,
        "pilot_prefix": {
            "identities": pilot_identities,
            "count": 21,
            "per_option_count": 3,
        },
        "permutation_selection_seed": "saracura-v02-sealed-permutation-v1",
        "permutation_subset_size": 50,
        "author_target_hidden": True,
    }


def _validate_plan_integrity(plan: dict[str, Any]) -> None:
    if "schema_version" not in plan or plan["schema_version"] != "v02-plan.v1":
        raise ValueError("plan schema_version mismatch")
    if "lane" not in plan or plan["lane"] not in ("training", "sealed"):
        raise ValueError("plan lane mismatch")


def plan(lane: str, output_parent: Path) -> Path:
    if lane not in ("training", "sealed"):
        raise ValueError("lane must be training or sealed")
    _safe_output_parent(output_parent)
    _check_worktrees(output_parent)
    if not output_parent.is_dir():
        raise ValueError("output-parent must be an existing directory")

    plan_data = _generate_training_plan() if lane == "training" else _generate_sealed_plan()

    plan_bytes = canonical_json_bytes(cast(Any, plan_data)) + b"\n"

    output_path = output_parent / f"{lane}-plan.json"
    if output_path.is_symlink():
        raise ValueError(f"output path is a symlink: {output_path}")

    _atomic_write(output_path, plan_bytes)
    return output_path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _closed_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    """Parse one canonical, duplicate-key-free protocol object."""
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not valid closed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if raw != canonical_json_bytes(cast(Any, value)) + b"\n":
        raise ValueError(f"{name} bytes are not canonical")
    return cast(dict[str, Any], value)


def _require_exact_keys(value: dict[str, Any], expected: frozenset[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are closed")


def _model_identity(role: str) -> str:
    if role not in _MODEL_ROLE_NAMES:
        raise ValueError("unknown model role")
    model = MODEL_ROLES[role]
    return f"{model['model']}@{model['revision']}"


def _validate_gates(gates: Any) -> dict[str, bool]:
    if not isinstance(gates, dict) or set(gates) != LOCAL_GATES:
        raise ValueError("local gates are closed")
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError("local gates must be booleans")
    return cast(dict[str, bool], gates)


def _validate_identity(plan: dict[str, Any], lane: str, identity_id: Any) -> dict[str, Any]:
    if not isinstance(identity_id, str):
        raise ValueError("identity must be a string")
    if lane in {"training", "sealed"} and plan.get("lane") == lane:
        identity = _frozen_identity_index(lane).get(identity_id)
        if identity is not None:
            return identity
    raise ValueError("identity is not planned")


def _validate_envelope_base(
    envelope: dict[str, Any],
    *,
    expected: frozenset[str],
    role: str,
    lane: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(envelope, expected, name="envelope")
    if envelope.get("schema_version") != "v02-envelope.v1":
        raise ValueError("envelope schema_version mismatch")
    if envelope.get("role") != role or envelope.get("model") != _model_identity(role):
        raise ValueError("envelope role or model mismatch")
    if envelope.get("lane") != lane:
        raise ValueError("envelope lane mismatch")
    _validate_identity(plan, lane, envelope.get("identity_id"))
    if "content_digest" in envelope and not _is_digest(envelope["content_digest"]):
        raise ValueError("content digest must be SHA-256")
    return envelope


def _validate_option(value: Any, identity: dict[str, Any]) -> None:
    if not isinstance(value, str) or value not in {
        f"option_{index}" for index in range(identity["option_count"])
    }:
        raise ValueError("answer or label is not a planned option")


def _validate_semantic(value: Any, slot: dict[str, Any]) -> None:
    expected = {"scenario_code": slot["scenario_code"], "criterion_roles": slot["criterion_roles"]}
    if value != expected:
        raise ValueError("semantic attestation differs from the closed planned target")


def validate_training_author_output(
    envelope: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    result = _validate_envelope_base(
        envelope,
        expected=frozenset(
            {
                "schema_version",
                "role",
                "model",
                "lane",
                "identity_id",
                "content_digest",
                "answer",
                "semantic",
                "gates",
            }
        ),
        role="training_author",
        lane="training",
        plan=plan,
    )
    slot = _validate_identity(plan, "training", result["identity_id"])
    _validate_option(result["answer"], slot)
    _validate_semantic(result["semantic"], slot)
    _validate_gates(result["gates"])
    return result


def validate_training_reviewer_decision(
    envelope: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    result = _validate_envelope_base(
        envelope,
        expected=frozenset(
            {
                "schema_version",
                "role",
                "model",
                "lane",
                "identity_id",
                "content_digest",
                "answer",
                "semantic",
                "gates",
            }
        ),
        role="independent_reviewer",
        lane="training",
        plan=plan,
    )
    slot = _validate_identity(plan, "training", result["identity_id"])
    _validate_option(result["answer"], slot)
    _validate_semantic(result["semantic"], slot)
    _validate_gates(result["gates"])
    return result


def validate_sealed_author_output(envelope: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    result = _validate_envelope_base(
        envelope,
        expected=frozenset(
            {
                "schema_version",
                "role",
                "model",
                "lane",
                "identity_id",
                "content_digest",
                "target",
                "gates",
            }
        ),
        role="sealed_author",
        lane="sealed",
        plan=plan,
    )
    identity = _validate_identity(plan, "sealed", result["identity_id"])
    _validate_option(result["target"], identity)
    _validate_gates(result["gates"])
    return result


def validate_blind_annotator_decision(
    envelope: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    role = envelope.get("role")
    if role not in {"sealed_annotator_a", "sealed_annotator_b"}:
        raise ValueError("annotator role mismatch")
    result = _validate_envelope_base(
        envelope,
        expected=frozenset(
            {
                "schema_version",
                "role",
                "model",
                "lane",
                "identity_id",
                "content_digest",
                "label",
                "gates",
            }
        ),
        role=cast(str, role),
        lane="sealed",
        plan=plan,
    )
    identity = _validate_identity(plan, "sealed", result["identity_id"])
    _validate_option(result["label"], identity)
    _validate_gates(result["gates"])
    return result


def validate_adjudicator_decision(
    envelope: dict[str, Any], plan: dict[str, Any], label_a: str, label_b: str
) -> dict[str, Any]:
    result = _validate_envelope_base(
        envelope,
        expected=frozenset(
            {
                "schema_version",
                "role",
                "model",
                "lane",
                "identity_id",
                "content_digest",
                "choice",
                "gates",
            }
        ),
        role="sealed_adjudicator",
        lane="sealed",
        plan=plan,
    )
    if label_a == label_b:
        raise ValueError("adjudication is forbidden for agreement")
    if result["choice"] not in {label_a, label_b}:
        raise ValueError("adjudicator must choose label A or B")
    _validate_gates(result["gates"])
    return result


def blind_review_binding(author: dict[str, Any], plan_data: dict[str, Any]) -> dict[str, Any]:
    """Bind a private case for review without exposing the author's answer/target."""
    if plan_data["lane"] == "training":
        validate_training_author_output(author, plan_data)
    else:
        validate_sealed_author_output(author, plan_data)
    return {key: author[key] for key in ("lane", "identity_id", "content_digest")}


@lru_cache(maxsize=2)
def _frozen_plan_bytes(lane: str) -> bytes:
    plan_data = _generate_training_plan() if lane == "training" else _generate_sealed_plan()
    return canonical_json_bytes(cast(Any, plan_data))


@lru_cache(maxsize=2)
def _frozen_identity_index(lane: str) -> dict[str, dict[str, Any]]:
    plan_data = json.loads(_frozen_plan_bytes(lane))
    rows = plan_data["slots"] if lane == "training" else plan_data["identities"]
    key = "slot_id" if lane == "training" else "identity_id"
    return {row[key]: row for row in rows}


def _validate_frozen_plan(plan_data: dict[str, Any]) -> None:
    lane = plan_data.get("lane")
    if lane not in {"training", "sealed"} or canonical_json_bytes(
        cast(Any, plan_data)
    ) != _frozen_plan_bytes(lane):
        raise ValueError("ledger plan differs from the frozen plan")


def _validate_failure(envelope: dict[str, Any], plan_data: dict[str, Any], transition: str) -> None:
    role = transition.removesuffix("_failure")
    allowed = (
        {"training_author", "training_reviewer"}
        if plan_data["lane"] == "training"
        else {"sealed_author", "sealed_annotator_a", "sealed_annotator_b", "sealed_adjudicator"}
    )
    if role not in allowed:
        raise ValueError("failure transition lane mismatch")
    if role == "training_reviewer":
        role = "independent_reviewer"
    _validate_envelope_base(
        envelope,
        expected=frozenset(
            {"schema_version", "role", "model", "lane", "identity_id", "error_code"}
        ),
        role=role,
        lane=plan_data["lane"],
        plan=plan_data,
    )
    if not isinstance(envelope["error_code"], str) or envelope["error_code"] not in {
        "model_error",
        "invalid_output",
        "local_gate_failure",
    }:
        raise ValueError("failure code is closed")


def _event_filename(identity_id: str, transition: str) -> str:
    return f"{_sha256(f'{identity_id}:{transition}'.encode())}.json"


class OfflineLedger:
    """Immutable, local-only transition log over one deterministic B1 plan."""

    def __init__(
        self,
        plan_data: dict[str, Any],
        artifact_root: Path,
        *,
        other_ancestry: dict[str, Any] | None = None,
    ) -> None:
        _validate_frozen_plan(plan_data)
        self.plan = plan_data
        self.lane = cast(str, plan_data.get("lane"))
        if self.lane not in {"training", "sealed"}:
            raise ValueError("ledger plan lane mismatch")
        _safe_output_parent(artifact_root)
        self.ancestry = _root_ancestry(artifact_root, self.lane)
        if other_ancestry is not None:
            _validate_separation(self.ancestry, other_ancestry)
        self.root = artifact_root
        binding = {
            "schema_version": "v02-ledger-binding.v1",
            "ancestry": self.ancestry,
            "plan_digest": _sha256(canonical_json_bytes(cast(Any, plan_data))),
        }
        binding_bytes = canonical_json_bytes(cast(Any, binding)) + b"\n"
        binding_path = artifact_root / "ledger-binding.json"
        try:
            _atomic_write(binding_path, binding_bytes)
        except ValueError:
            if (
                binding_path.is_symlink()
                or not binding_path.is_file()
                or binding_path.read_bytes() != binding_bytes
            ):
                raise ValueError(
                    "ledger root already belongs to a different plan or lane"
                ) from None
        self.events_root = artifact_root / "ledger-events"
        if self.events_root.exists() and self.events_root.is_symlink():
            raise ValueError("ledger events path is a symlink")
        self.events_root.mkdir(mode=0o700, exist_ok=True)
        if (
            self.events_root.stat().st_uid != os.geteuid()
            or stat.S_IMODE(self.events_root.stat().st_mode) != 0o700
        ):
            raise ValueError("ledger events path must have exact permission mode 0700")

    def _read_existing(self, path: Path) -> dict[str, Any]:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_uid != os.geteuid()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise ValueError("ledger event must be an owned regular mode-0600 file")
        return _closed_json_object(path.read_bytes(), name="ledger event")

    def events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in sorted(self.events_root.glob("*.json")):
            event = self._read_existing(path)
            self._validate_event(event)
            if path.name != _event_filename(event["identity_id"], event["transition"]):
                raise ValueError("ledger event filename mismatch")
            events.append(event)
        _validated_events(self.plan, events)
        return events

    def _validate_event(self, event: dict[str, Any]) -> None:
        _require_exact_keys(
            event,
            frozenset(
                {
                    "schema_version",
                    "lane",
                    "identity_id",
                    "transition",
                    "envelope_digest",
                    "envelope",
                }
            ),
            name="ledger event",
        )
        if event["schema_version"] != LEDGER_SCHEMA or event["lane"] != self.lane:
            raise ValueError("ledger event schema or lane mismatch")
        if not isinstance(event["envelope"], dict):
            raise ValueError("ledger event envelope must be an object")
        if event["envelope_digest"] != _sha256(canonical_json_bytes(cast(Any, event["envelope"]))):
            raise ValueError("ledger event envelope digest mismatch")
        self._validate_transition(
            cast(str, event["transition"]), cast(dict[str, Any], event["envelope"])
        )
        if event["identity_id"] != event["envelope"]["identity_id"]:
            raise ValueError("ledger event identity mismatch")

    def _validate_transition(self, transition: str, envelope: dict[str, Any]) -> None:
        if transition.endswith("_failure"):
            _validate_failure(envelope, self.plan, transition)
            return
        validators = {
            "training_author": validate_training_author_output,
            "training_reviewer": validate_training_reviewer_decision,
            "sealed_author": validate_sealed_author_output,
            "sealed_annotator_a": validate_blind_annotator_decision,
            "sealed_annotator_b": validate_blind_annotator_decision,
        }
        if transition == "sealed_adjudicator":
            # The binding to committed A/B labels is checked at transition time.
            _validate_envelope_base(
                envelope,
                expected=frozenset(
                    {
                        "schema_version",
                        "role",
                        "model",
                        "lane",
                        "identity_id",
                        "content_digest",
                        "choice",
                        "gates",
                    }
                ),
                role="sealed_adjudicator",
                lane="sealed",
                plan=self.plan,
            )
            if not isinstance(envelope["choice"], str):
                raise ValueError("adjudicator choice must be a string")
            _validate_gates(envelope["gates"])
            return
        validator = validators.get(transition)
        if validator is None:
            raise ValueError("unknown ledger transition")
        validator(envelope, self.plan)
        required_role = transition if transition != "training_reviewer" else "independent_reviewer"
        if envelope.get("role") != required_role:
            raise ValueError("ledger transition role mismatch")

    def commit(self, transition: str, envelope: dict[str, Any]) -> bool:
        """Commit once; equal pre-existing bytes are a verified resume, never a rewrite."""
        identity_id = envelope.get("identity_id")
        if not isinstance(identity_id, str):
            raise ValueError("identity must be a string")
        self._validate_transition(transition, envelope)
        before = self.events()
        existing = _group_events(before).get(identity_id, {})
        self._validate_legal_transition(transition, envelope, existing)
        event = {
            "schema_version": LEDGER_SCHEMA,
            "lane": self.lane,
            "identity_id": identity_id,
            "transition": transition,
            "envelope_digest": _sha256(canonical_json_bytes(cast(Any, envelope))),
            "envelope": envelope,
        }
        data = canonical_json_bytes(cast(Any, event)) + b"\n"
        path = self.events_root / _event_filename(identity_id, transition)
        if path.is_symlink():
            raise ValueError("ledger event path is a symlink")
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("duplicate ledger commitment differs")
            self._validate_event(self._read_existing(path))
            return False
        _validated_events(self.plan, [*before, event])
        metrics = (
            reduce_training(self.plan, before)
            if self.lane == "training"
            else reduce_sealed(self.plan, before)
        )
        if metrics["status"] == "NO_GO" or metrics["pilot"]["status"] == "NO_GO":
            raise ValueError("ledger is terminal NO_GO")
        pilot_ids = (
            set(self.plan["pilot_prefix"]["slot_ids"])
            if self.lane == "training"
            else {item["identity_id"] for item in self.plan["pilot_prefix"]["identities"]}
        )
        if identity_id not in pilot_ids and metrics["pilot"]["status"] != "PASS":
            raise ValueError("pilot must settle and pass before full-lane commitments")
        try:
            _atomic_write(path, data)
        except ValueError as exc:
            if path.exists() and path.read_bytes() == data:
                self._validate_event(self._read_existing(path))
                return False
            raise exc
        return True

    def by_identity(self, identity_id: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for event in self.events():
            if event["identity_id"] == identity_id:
                transition = cast(str, event["transition"])
                if transition in result:
                    raise ValueError("duplicate ledger transition")
                result[transition] = event
        return result

    def _validate_legal_transition(
        self, transition: str, envelope: dict[str, Any], existing: dict[str, dict[str, Any]]
    ) -> None:
        if transition in existing:
            return
        failed = any(name.endswith("_failure") for name in existing)
        if failed:
            raise ValueError("identity already settled by a failure")
        successful_transition = transition.removesuffix("_failure")
        existing_roles = {name.removesuffix("_failure") for name in existing}
        if successful_transition in existing_roles:
            raise ValueError("identity role already committed")
        transition = successful_transition
        if self.lane == "training":
            if transition == "training_author" and not existing:
                return
            if transition == "training_reviewer" and set(existing) == {"training_author"}:
                return
            raise ValueError("training ledger transition is out of order")
        if transition == "sealed_author" and not existing:
            return
        if transition in {"sealed_annotator_a", "sealed_annotator_b"} and set(existing) in (
            {"sealed_author"},
            {"sealed_author", "sealed_annotator_a"},
            {"sealed_author", "sealed_annotator_b"},
        ):
            return
        if transition == "sealed_adjudicator" and set(existing) == {
            "sealed_author",
            "sealed_annotator_a",
            "sealed_annotator_b",
        }:
            a = cast(dict[str, Any], existing["sealed_annotator_a"]["envelope"])["label"]
            b = cast(dict[str, Any], existing["sealed_annotator_b"]["envelope"])["label"]
            if "error_code" in envelope:
                if a == b:
                    raise ValueError("adjudication is forbidden for agreement")
            else:
                validate_adjudicator_decision(envelope, self.plan, cast(str, a), cast(str, b))
            return
        raise ValueError("sealed ledger transition is out of order")


def _all_gates(envelope: dict[str, Any]) -> bool:
    return all(cast(dict[str, bool], envelope["gates"]).values())


def _validated_events(
    plan_data: dict[str, Any], events: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Revalidate the complete immutable history on resume and before reduction."""
    _validate_frozen_plan(plan_data)
    verifier = object.__new__(OfflineLedger)
    verifier.plan = plan_data
    verifier.lane = plan_data["lane"]
    event_list = list(events)
    for event in event_list:
        verifier._validate_event(event)
    grouped = _group_events(event_list)
    order = {
        name: index
        for index, name in enumerate(
            (
                "training_author",
                "training_reviewer",
                "sealed_author",
                "sealed_annotator_a",
                "sealed_annotator_b",
                "sealed_adjudicator",
            )
        )
    }
    for transitions in grouped.values():
        existing: dict[str, dict[str, Any]] = {}
        for name, event in sorted(
            transitions.items(), key=lambda item: order[item[0].removesuffix("_failure")]
        ):
            envelope = event["envelope"]
            verifier._validate_legal_transition(name, envelope, existing)
            author_event = existing.get("training_author") or existing.get("sealed_author")
            if (
                author_event is not None
                and "error_code" not in envelope
                and envelope["content_digest"] != author_event["envelope"]["content_digest"]
            ):
                raise ValueError("review or annotation content digest differs from author")
            existing[name] = event
    return event_list


def reduce_training(plan_data: dict[str, Any], events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    slots = cast(list[dict[str, Any]], plan_data["slots"])
    grouped = _group_events(_validated_events(plan_data, events))
    accepted: set[str] = set()
    resolved: set[str] = set()
    cohorts: dict[str, dict[str, int]] = {"split": {}, "locale": {}, "option_count": {}}
    for slot in slots:
        identity_id = cast(str, slot["slot_id"])
        transition = grouped.get(identity_id, {})
        author = _event_envelope(transition.get("training_author"))
        reviewer = _event_envelope(transition.get("training_reviewer"))
        if reviewer is not None or any(name.endswith("_failure") for name in transition):
            resolved.add(identity_id)
        if (
            author
            and reviewer
            and (
                author["answer"] == reviewer["answer"]
                and author["answer"] == f"option_{slot['gold_position']}"
                and author["semantic"] == reviewer["semantic"]
                and _all_gates(author)
                and _all_gates(reviewer)
            )
        ):
            accepted.add(identity_id)
            for key, value in (
                ("split", cast(str, slot["split"])),
                ("locale", cast(str, slot["locale"])),
                ("option_count", str(slot["option_count"])),
            ):
                cohorts[key][value] = cohorts[key].get(value, 0) + 1
    pilot_ids = {cast(str, slot_id) for slot_id in plan_data["pilot_prefix"]["slot_ids"]}
    pilot_slots = [slot for slot in slots if slot["slot_id"] in pilot_ids]
    pilot = _training_pilot_metrics(pilot_slots, accepted, resolved)
    return {
        "accepted": len(accepted),
        "resolved": len(resolved),
        "unresolved": len(slots) - len(resolved),
        "denominator": len(slots),
        "cohorts": cohorts,
        "pilot": pilot,
        "status": "NO_GO"
        if pilot["status"] == "NO_GO"
        else _training_full_status(slots, accepted, resolved),
    }


def _training_pilot_metrics(
    slots: list[dict[str, Any]], accepted: set[str], resolved: set[str]
) -> dict[str, Any]:
    accepted_ids = {cast(str, slot["slot_id"]) for slot in slots if slot["slot_id"] in accepted}
    unresolved_ids = {
        cast(str, slot["slot_id"]) for slot in slots if slot["slot_id"] not in resolved
    }
    locale_counts = {
        locale: sum(slot["slot_id"] in accepted for slot in slots if slot["locale"] == locale)
        for locale in ("pt_br", "english")
    }
    option_counts = {
        str(option): sum(
            slot["slot_id"] in accepted for slot in slots if slot["option_count"] == option
        )
        for option in OPTION_COUNTS
    }
    possible_locale = {
        locale: sum(
            slot["slot_id"] in accepted_ids | unresolved_ids
            for slot in slots
            if slot["locale"] == locale
        )
        for locale in ("pt_br", "english")
    }
    possible_option = {
        str(option): sum(
            slot["slot_id"] in accepted_ids | unresolved_ids
            for slot in slots
            if slot["option_count"] == option
        )
        for option in OPTION_COUNTS
    }
    passed = (
        not unresolved_ids
        and len(accepted_ids) >= 106
        and all(value >= 53 for value in locale_counts.values())
        and all(value >= 15 for value in option_counts.values())
    )
    impossible = (
        len(accepted_ids | unresolved_ids) < 106
        or any(value < 53 for value in possible_locale.values())
        or any(value < 15 for value in possible_option.values())
    )
    return {
        "accepted": len(accepted_ids),
        "denominator": 140,
        "locale_accepted": locale_counts,
        "option_count_accepted": option_counts,
        "status": "PASS" if passed else "NO_GO" if impossible else "PENDING",
    }


def _training_full_status(
    slots: list[dict[str, Any]], accepted: set[str], resolved: set[str]
) -> str:
    # The global, split, locale and cardinality minima are feasibility gates.
    requirements: list[tuple[int, list[dict[str, Any]]]] = [(1200, slots)]
    for split, minimum in (("train", 1020), ("internal_dev", 180)):
        requirements.append((minimum, [slot for slot in slots if slot["split"] == split]))
    for option in OPTION_COUNTS:
        for split, minimum in (("train", 20), ("internal_dev", 10)):
            for locale in ("pt_br", "english"):
                requirements.append(
                    (
                        minimum,
                        [
                            slot
                            for slot in slots
                            if slot["split"] == split
                            and slot["locale"] == locale
                            and slot["option_count"] == option
                        ],
                    )
                )
    for split, minimum in (("train", 60), ("internal_dev", 10)):
        for domain in DOMAINS:
            requirements.append(
                (
                    minimum,
                    [slot for slot in slots if slot["split"] == split and slot["domain"] == domain],
                )
            )

    def possible(group: list[dict[str, Any]]) -> int:
        return sum(slot["slot_id"] in accepted or slot["slot_id"] not in resolved for slot in group)

    def actual(group: list[dict[str, Any]]) -> int:
        return sum(slot["slot_id"] in accepted for slot in group)

    if any(possible(group) < minimum for minimum, group in requirements):
        return "NO_GO"
    pairs: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        pair_id = slot["bilingual_pair_id"]
        if pair_id is not None:
            pairs.setdefault(cast(str, pair_id), []).append(slot)
    accepted_pairs = sum(
        all(member["slot_id"] in accepted for member in members) for members in pairs.values()
    )
    possible_pairs = sum(
        all(
            member["slot_id"] in accepted or member["slot_id"] not in resolved for member in members
        )
        for members in pairs.values()
    )
    if possible_pairs < 120:
        return "NO_GO"
    percentages_pass = True
    for locale, numerator in (("pt_br", 60), ("english", 20)):
        group = [slot for slot in slots if slot["locale"] == locale]
        current = actual(group)
        available = sum(slot["slot_id"] not in resolved for slot in group)
        if 100 * (current + available) < numerator * max(1200, len(accepted) + available):
            return "NO_GO"
        percentages_pass = percentages_pass and 100 * current >= numerator * len(accepted)
    if (
        len(resolved) == len(slots)
        and percentages_pass
        and accepted_pairs >= 120
        and all(actual(group) >= minimum for minimum, group in requirements)
    ):
        return "READY"
    return "PENDING"


def reduce_sealed(plan_data: dict[str, Any], events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    identities = cast(list[dict[str, Any]], plan_data["identities"])
    grouped = _group_events(_validated_events(plan_data, events))
    accepted: set[str] = set()
    resolved: set[str] = set()
    direct_agreements: set[str] = set()
    adjudications: set[str] = set()
    for identity in identities:
        identity_id = cast(str, identity["identity_id"])
        transition = grouped.get(identity_id, {})
        author = _event_envelope(transition.get("sealed_author"))
        a = _event_envelope(transition.get("sealed_annotator_a"))
        b = _event_envelope(transition.get("sealed_annotator_b"))
        adjudication = _event_envelope(transition.get("sealed_adjudicator"))
        if any(name.endswith("_failure") for name in transition):
            resolved.add(identity_id)
            continue
        if a and b:
            if a["label"] == b["label"]:
                resolved.add(identity_id)
                direct_agreements.add(identity_id)
                chosen = a["label"]
            elif adjudication:
                resolved.add(identity_id)
                adjudications.add(identity_id)
                chosen = adjudication["choice"]
            else:
                continue
            if (
                author
                and chosen == author["target"]
                and all(_all_gates(envelope) for envelope in (author, a, b) if envelope is not None)
                and (adjudication is None or _all_gates(adjudication))
            ):
                accepted.add(identity_id)
    pilot_ids = {
        cast(str, identity["identity_id"]) for identity in plan_data["pilot_prefix"]["identities"]
    }
    full_ids = {cast(str, identity["identity_id"]) for identity in identities}
    pilot = _sealed_metrics(
        identities, pilot_ids, accepted, resolved, direct_agreements, adjudications, 15, 2, 14, 7
    )
    full = _sealed_metrics(
        identities, full_ids, accepted, resolved, direct_agreements, adjudications, 100, 14, 0, 140
    )
    return {
        "accepted": len(accepted),
        "resolved": len(resolved),
        "unresolved": len(identities) - len(resolved),
        "denominator": len(identities),
        "direct_agreements": len(direct_agreements),
        "adjudications": len(adjudications),
        "pilot": pilot,
        "status": "NO_GO"
        if pilot["status"] == "NO_GO"
        else "READY"
        if full["status"] == "PASS"
        else full["status"],
        "cohorts": full["option_count_accepted"],
    }


def _sealed_metrics(
    identities: list[dict[str, Any]],
    selection: set[str],
    accepted: set[str],
    resolved: set[str],
    direct_agreements: set[str],
    adjudications: set[str],
    minimum: int,
    per_option_minimum: int,
    direct_minimum: int,
    adjudication_maximum: int,
) -> dict[str, Any]:
    selected = [identity for identity in identities if identity["identity_id"] in selection]
    accepted_ids = {
        cast(str, identity["identity_id"])
        for identity in selected
        if identity["identity_id"] in accepted
    }
    possible_ids = {
        cast(str, identity["identity_id"])
        for identity in selected
        if identity["identity_id"] in accepted or identity["identity_id"] not in resolved
    }
    counts = {
        str(option): sum(
            identity["identity_id"] in accepted
            for identity in selected
            if identity["option_count"] == option
        )
        for option in OPTION_COUNTS
    }
    possible_counts = {
        str(option): sum(
            identity["identity_id"] in possible_ids
            for identity in selected
            if identity["option_count"] == option
        )
        for option in OPTION_COUNTS
    }
    direct = len(selection & direct_agreements)
    adjudicated = len(selection & adjudications)
    unresolved_ids = selection - resolved
    passed = (
        not (selection - resolved)
        and len(accepted_ids) >= minimum
        and all(value >= per_option_minimum for value in counts.values())
        and direct >= direct_minimum
        and adjudicated <= adjudication_maximum
    )
    impossible = len(possible_ids) < minimum or any(
        value < per_option_minimum for value in possible_counts.values()
    )
    impossible = (
        impossible
        or direct + len(unresolved_ids) < direct_minimum
        or adjudicated > adjudication_maximum
    )
    return {
        "accepted": len(accepted_ids),
        "denominator": len(selected),
        "option_count_accepted": counts,
        "status": "PASS" if passed else "NO_GO" if impossible else "PENDING",
    }


def _group_events(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events:
        identity_id = cast(str, event["identity_id"])
        transition = cast(str, event["transition"])
        if transition in grouped.setdefault(identity_id, {}):
            raise ValueError("duplicate event transition")
        grouped[identity_id][transition] = event
    return grouped


def _event_envelope(event: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if event is None else cast(dict[str, Any], event["envelope"])


def event_digest(events: Iterable[dict[str, Any]]) -> str:
    canonical_events = sorted(
        events, key=lambda event: (str(event["identity_id"]), str(event["transition"]))
    )
    return _sha256(canonical_json_bytes(cast(Any, canonical_events)))


def create_aggregate_receipt(
    output_parent: Path,
    *,
    artifact_id: str,
    plan_data: dict[str, Any],
    events: Iterable[dict[str, Any]],
    metrics: dict[str, Any],
    status: str,
    ancestry: dict[str, Any],
) -> Path:
    _safe_output_parent(output_parent)
    if not _logical_id(artifact_id):
        raise ValueError("artifact_id must be logical")
    if status not in {"PENDING", "NO_GO", "READY"}:
        raise ValueError("receipt status is closed")
    _validate_ancestry(ancestry, plan_data["lane"])
    if ancestry != _root_ancestry(output_parent, plan_data["lane"]):
        raise ValueError("receipt ancestry differs from actual artifact root")
    event_list = _validated_events(plan_data, events)
    derived = (
        reduce_training(plan_data, event_list)
        if plan_data["lane"] == "training"
        else reduce_sealed(plan_data, event_list)
    )
    if metrics != derived or status != derived["status"]:
        raise ValueError("receipt metrics or status differ from verified events")
    _validate_aggregate_metrics(metrics)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": artifact_id,
        "plan_digest": _sha256(canonical_json_bytes(cast(Any, plan_data))),
        "event_digest": event_digest(event_list),
        "ancestry": ancestry,
        "counts": {
            key: metrics[key] for key in ("accepted", "resolved", "unresolved", "denominator")
        },
        "cohort_metrics": metrics["cohorts"],
        "pilot_metrics": metrics["pilot"],
        "status": status,
    }
    path = output_parent / f"{artifact_id}.receipt.json"
    _atomic_write(path, canonical_json_bytes(cast(Any, receipt)) + b"\n")
    return path


def verify_receipt(
    path: Path,
    *,
    plan_data: dict[str, Any],
    events: Iterable[dict[str, Any]],
    expected_lane: str,
    other_root_binding: str | None = None,
    other_ancestry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _safe_output_parent(path.parent)
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.stat().st_uid != os.geteuid()
    ):
        raise ValueError("receipt must be an owned regular mode-0600 file")
    receipt = _closed_json_object(path.read_bytes(), name="aggregate receipt")
    _require_exact_keys(
        receipt,
        frozenset(
            {
                "schema_version",
                "artifact_id",
                "plan_digest",
                "event_digest",
                "ancestry",
                "counts",
                "cohort_metrics",
                "pilot_metrics",
                "status",
            }
        ),
        name="aggregate receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA or receipt["status"] not in {
        "PENDING",
        "NO_GO",
        "READY",
    }:
        raise ValueError("aggregate receipt schema or status mismatch")
    if receipt["plan_digest"] != _sha256(canonical_json_bytes(cast(Any, plan_data))):
        raise ValueError("aggregate receipt plan digest mismatch")
    event_list = _validated_events(plan_data, events)
    if receipt["event_digest"] != event_digest(event_list):
        raise ValueError("aggregate receipt event digest mismatch")
    ancestry = receipt["ancestry"]
    _validate_ancestry(ancestry, expected_lane)
    if expected_lane != plan_data["lane"] or ancestry != _root_ancestry(path.parent, expected_lane):
        raise ValueError("aggregate receipt ancestry mismatch")
    if (
        not _logical_id(receipt["artifact_id"])
        or path.name != f"{receipt['artifact_id']}.receipt.json"
    ):
        raise ValueError("aggregate receipt logical artifact ID mismatch")
    if other_ancestry is not None:
        _validate_separation(ancestry, other_ancestry)
    if other_root_binding is not None and ancestry["root_binding"] == other_root_binding:
        raise ValueError("aggregate receipt root separation mismatch")
    counts = receipt["counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"accepted", "resolved", "unresolved", "denominator"}
        or any(type(value) is not int or value < 0 for value in counts.values())
        or counts["resolved"] + counts["unresolved"] != counts["denominator"]
        or counts["accepted"] > counts["resolved"]
    ):
        raise ValueError("aggregate receipt count invariants mismatch")
    _validate_aggregate_value(receipt["cohort_metrics"])
    metrics = (
        reduce_training(plan_data, event_list)
        if expected_lane == "training"
        else reduce_sealed(plan_data, event_list)
    )
    if (
        counts
        != {key: metrics[key] for key in ("accepted", "resolved", "unresolved", "denominator")}
        or receipt["cohort_metrics"] != metrics["cohorts"]
        or receipt["pilot_metrics"] != metrics["pilot"]
        or receipt["status"] != metrics["status"]
    ):
        raise ValueError("aggregate receipt metrics or status differ from verified events")
    return receipt


def _logical_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 64
        and value[0].isalnum()
        and all(
            character.isascii() and (character.isalnum() or character in "_-")
            for character in value
        )
    )


def _inode_binding(path: Path) -> str:
    metadata = path.stat()
    return _sha256(canonical_json_bytes({"device": metadata.st_dev, "inode": metadata.st_ino}))


def _root_ancestry(root: Path, lane: str) -> dict[str, Any]:
    return {
        "lane": lane,
        "root_binding": _inode_binding(root),
        "ancestor_bindings": [_inode_binding(parent) for parent in root.parents],
    }


def _validate_ancestry(value: Any, lane: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"lane", "root_binding", "ancestor_bindings"}
        or value["lane"] != lane
        or lane not in {"training", "sealed"}
        or not _is_digest(value["root_binding"])
        or not isinstance(value["ancestor_bindings"], list)
        or not value["ancestor_bindings"]
        or any(not _is_digest(item) for item in value["ancestor_bindings"])
    ):
        raise ValueError("aggregate receipt ancestry is closed")


def _validate_separation(ancestry: dict[str, Any], other: dict[str, Any]) -> None:
    other_lane = "sealed" if ancestry["lane"] == "training" else "training"
    _validate_ancestry(other, other_lane)
    if (
        ancestry["root_binding"] in [other["root_binding"], *other["ancestor_bindings"]]
        or other["root_binding"] in ancestry["ancestor_bindings"]
    ):
        raise ValueError("aggregate receipt root separation mismatch")


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_aggregate_value(value: Any) -> None:
    if type(value) is int and value >= 0:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                not isinstance(key, str)
                or not key.replace("_", "").isalnum()
                or any(
                    word in key.lower() for word in ("raw", "prompt", "label", "path", "credential")
                )
            ):
                raise ValueError("aggregate receipt metrics contain a private field")
            _validate_aggregate_value(child)
        return
    raise ValueError("aggregate receipt metrics must contain only counts")


def _validate_aggregate_metrics(metrics: dict[str, Any]) -> None:
    required = {"accepted", "resolved", "unresolved", "denominator", "cohorts"}
    if not required <= set(metrics):
        raise ValueError("receipt metrics are closed")
    for key in required - {"cohorts"}:
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError("receipt counts must be non-negative integers")
    _validate_aggregate_value(metrics["cohorts"])


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: v02_corpus <command> [args]", file=sys.stderr)
        return 2

    command = args[0]

    if command == "validate-protocol":
        return validate_protocol()

    if command == "verify-receipt":
        parser = argparse.ArgumentParser(prog="v02_corpus verify-receipt")
        parser.add_argument("--artifact-root", type=Path, required=True)
        parser.add_argument("--lane", choices=("training", "sealed"), required=True)
        parser.add_argument("--receipt-id", required=True)
        parsed = parser.parse_args(args[1:])
        try:
            _safe_output_parent(parsed.artifact_root)
            if not _logical_id(parsed.receipt_id):
                raise ValueError("receipt ID must be logical")
            plan_path = parsed.artifact_root / f"{parsed.lane}-plan.json"
            if plan_path.is_symlink() or not plan_path.is_file():
                raise ValueError("plan must be a regular private artifact")
            if (
                not (parsed.artifact_root / "ledger-binding.json").is_file()
                or not (parsed.artifact_root / "ledger-events").is_dir()
            ):
                raise ValueError("ledger must already exist for verification")
            plan_data = _closed_json_object(plan_path.read_bytes(), name="plan")
            ledger = OfflineLedger(plan_data, parsed.artifact_root)
            verify_receipt(
                parsed.artifact_root / f"{parsed.receipt_id}.receipt.json",
                plan_data=plan_data,
                events=ledger.events(),
                expected_lane=parsed.lane,
            )
        except (ValueError, OSError):
            print("ERROR: receipt verification failed", file=sys.stderr)
            return 1
        print("receipt verified")
        return 0

    if command == "plan":
        lane = None
        output_parent = None
        i = 1
        while i < len(args):
            if args[i] == "--lane" and i + 1 < len(args):
                lane = args[i + 1]
                i += 2
            elif args[i] == "--output-parent" and i + 1 < len(args):
                output_parent = Path(args[i + 1])
                i += 2
            else:
                print(f"unknown argument: {args[i]}", file=sys.stderr)
                return 2
        if lane is None or output_parent is None:
            print("plan requires --lane and --output-parent", file=sys.stderr)
            return 2
        try:
            plan(lane, output_parent)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("plan written")
        return 0

    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
