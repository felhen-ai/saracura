"""Offline, deterministic corpus and sealed-plan planner for Phase B1."""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import stat
import subprocess
import sys
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


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: v02_corpus <command> [args]", file=sys.stderr)
        return 2

    command = args[0]

    if command == "validate-protocol":
        return validate_protocol()

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
